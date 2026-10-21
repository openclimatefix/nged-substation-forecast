"""Falsifier F2 of experiment 152: does the latent filter matter at D12 at all?

Replays the filter on experiment 110's stored L0 predictions at D12 (frozen signed flow `p_mw`
and frozen mean `mean - level - anomaly`), day-ahead, in each of the replicate and scored years:

- arm A: the additive filter of `studies.dp_assimilation` on the reading `|S|`, with symmetric
  quantiles (the mean plus the centred residual quantiles of the other year);
- arm F: the gated latent filter of `e00152_latent_filter`, with folded quantiles (the signed
  flow plus residuals drawn from the other year, folded to `|.|`);
- arm F-ungated: F without the gate, to show what the gate does;
- arm AF: arm A's corrected reading with its sign taken from the frozen flow, and folded
  quantiles (the fall-back if the latent filter is dropped).

The residual pools come from the other year (the scored year uses the replicate year's, which
is earlier; the replicate year uses the scored year's, which is later: a screening convention,
because F2 only decides whether the latent filter is worth building).

    uv run python ml-experiments/00152-*/e00152_f2.py
"""

import json

import numpy as np
import polars as pl
from e00152_common import EXPERIMENT_110_DIR, WORK_DIR, load_cache, quantile_score
from e00152_latent_filter import correction_at as latent_correction
from e00152_latent_filter import folded_quantiles, run_latent
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_gam import QUANTILE_LEVELS

SITE = "D12"
N_BOOTSTRAP = 10_000
N_SAMPLES = 200
SEED = 152
FOLDED_ARMS = ("f", "f_ungated", "af")
"""F, F without the gate, and the additive filter with folded quantiles (A plus folded)."""
LEVELS = np.array(QUANTILE_LEVELS)


def load_year(*, cut: str, cache) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Return D12's stored L0 rows of one year as arrays on the cache grid positions."""
    ts_id = cache.sites[SITE].table.time_series_id
    frame = (
        pl.read_parquet(EXPERIMENT_110_DIR / "predictions" / f"{cut}_l0" / "predictions.parquet")
        .filter((pl.col("time_series_id") == ts_id) & pl.col("filtered"))
        .sort("valid_time")
    )
    stamps = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    index = np.searchsorted(cache.end_times, stamps)
    assert (cache.end_times[index] == stamps).all()
    return {
        "index": index,
        "actual": frame["actual"].to_numpy().astype(np.float64),
        "p_hat": frame["p_mw"].to_numpy().astype(np.float64),
        "q_hat": frame["q_mvar"].to_numpy().astype(np.float64),
        "stored_mean": frame["mean"].to_numpy().astype(np.float64),
        "frozen": (frame["mean"] - frame["level"] - frame["anomaly"]).to_numpy().astype(np.float64),
        "masked": frame["switching_mask"].to_numpy(),
    }


def replay(*, cache, year: dict[str, np.ndarray], hyper: FilterHyperparameters) -> dict:  # noqa: ANN001
    """Run arms A, F, and F-ungated over one year and return their corrected flows."""
    band = cache.calendar.band
    n = len(cache.end_times)
    index = year["index"]
    valid = cache.end_times[index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=valid, days_ahead=1),
        delay_hours=DEFAULT_DELAY_HOURS,
    )
    residual = np.full(n, np.nan)
    residual[index] = year["actual"] - year["frozen"]
    additive = correction_at(
        trace=run_kalman(residual=residual, band=band, hyper=hyper),
        cutoff=cutoff,
        target=index,
        band=band[index],
    ).total
    reading = np.full(n, np.nan)
    reading[index] = year["actual"]
    p_hat = np.full(n, np.nan)
    p_hat[index] = year["p_hat"]
    q_hat = np.zeros(n)
    q_hat[index] = year["q_hat"]
    out = {"additive": additive, "mean_a": year["frozen"] + additive}
    for name, gate in (("f", True), ("f_ungated", False)):
        trace = run_latent(
            reading=reading, p_hat=p_hat, q_hat=q_hat, band=band, hyper=hyper, gate=gate
        )
        flow = year["p_hat"] + latent_correction(
            trace=trace, cutoff=cutoff, target=index, band=band[index]
        )
        out[f"flow_{name}"] = flow
        out[f"updated_{name}"] = float(np.mean(np.concatenate(trace.updated)))
    out["flow_af"] = np.where(year["p_hat"] < 0, -1.0, 1.0) * out["mean_a"]
    return out


def score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row, in points."""
    return quantile_score(actual=actual, quantiles=quantiles)


def month_block_interval(*, diff: np.ndarray, months: np.ndarray) -> tuple[float, float, float]:
    """Return the mean paired difference and its 95% month-block bootstrap interval."""
    unique = np.unique(months)
    block_sum = np.array([diff[months == m].sum() for m in unique])
    block_n = np.array([(months == m).sum() for m in unique])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(unique), size=(N_BOOTSTRAP, len(unique)))
    boot = block_sum[draws].sum(axis=1) / block_n[draws].sum(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:  # noqa: PLR0915
    """Run F2 and write `work/f2.json`."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    tuned = json.loads((EXPERIMENT_110_DIR / "work" / "tuned.json").read_text())
    hyper = FilterHyperparameters.from_array(values=np.array(tuned["filter_hyper"]["l0"][SITE]))
    years = {cut: load_year(cut=cut, cache=cache) for cut in ("replicate", "scored")}
    sunny_rows = pl.read_parquet(WORK_DIR / "step0_rows.parquet").filter(
        (pl.col("site") == SITE) & pl.col("filtered")
    )
    sunny = {}
    for cut in years:
        stamps = sunny_rows.filter(pl.col("cut") == cut).sort("valid_time")
        sunny[cut] = stamps["sunny_midday"].to_numpy()
    replays = {cut: replay(cache=cache, year=years[cut], hyper=hyper) for cut in years}
    band, quarter = cache.calendar.band, cache.calendar.quarter
    out: dict[str, object] = {"hyper": hyper.as_array().tolist()}
    for cut, other in (("replicate", "scored"), ("scored", "replicate")):
        year, run, ref = years[cut], replays[cut], years[other]
        ref_run = replays[other]
        index = year["index"]
        # Check: the additive replay reproduces experiment 110's stored mean.
        out[f"{cut}_replay_max_abs_diff_points"] = float(
            np.max(np.abs(run["mean_a"] - year["stored_mean"]))
        )
        # Residual pools from the other year.
        pool_a: dict[tuple[int, int], np.ndarray] = {}
        pools: dict[str, dict[tuple[int, int], np.ndarray]] = {name: {} for name in FOLDED_ARMS}
        r_index = ref["index"]
        keep_ref = ~ref["masked"]
        res_a = ref["actual"] - ref_run["mean_a"]
        signed_ref = {}
        for name in FOLDED_ARMS:
            flow_ref = ref_run[f"flow_{name}"]
            sign_ref = np.where(flow_ref < 0, -1.0, 1.0)
            reading_ref = np.sqrt(np.maximum(ref["actual"] ** 2 - ref["q_hat"] ** 2, 0.0))
            signed_ref[name] = sign_ref * reading_ref - flow_ref
        for b in range(4):
            for q in range(4):
                cell = keep_ref & (band[r_index] == b) & (quarter[r_index] == q)
                if cell.sum() < 30:
                    cell = keep_ref & (band[r_index] == b)
                pool_a[(b, q)] = res_a[cell] - res_a[cell].mean()
                for name in FOLDED_ARMS:
                    pools[name][(b, q)] = signed_ref[name][cell] - signed_ref[name][cell].mean()
        keep = ~year["masked"]
        arm_q: dict[str, np.ndarray] = {}
        mean_a = run["mean_a"]
        qa = np.empty((len(index), len(LEVELS)))
        for (b, q), pool in pool_a.items():
            rows = np.flatnonzero((band[index] == b) & (quarter[index] == q))
            qa[rows] = mean_a[rows, None] + np.quantile(pool, LEVELS)[None, :]
        arm_q["a"] = np.maximum(qa, 0.0)
        for name in FOLDED_ARMS:
            arm_q[name] = folded_quantiles(
                flow=run[f"flow_{name}"],
                reactive=year["q_hat"],
                residual_pool=pools[name],
                band=band[index],
                quarter=quarter[index],
                levels=LEVELS,
                n_samples=N_SAMPLES,
                seed=SEED,
            )
        months = cache.end_times[index].astype("datetime64[M]").astype(int)
        scores = {k: score(actual=year["actual"], quantiles=v) for k, v in arm_q.items()}
        sunny_mask = sunny[cut] & keep
        for slice_name, mask in (("sunny_midday", sunny_mask), ("all", keep)):
            for arm in FOLDED_ARMS:
                mean, lo, hi = month_block_interval(
                    diff=(scores[arm] - scores["a"])[mask], months=months[mask]
                )
                out[f"{cut}_{slice_name}_{arm}_minus_a"] = [mean, lo, hi]
            out[f"{cut}_{slice_name}_score_a"] = float(scores["a"][mask].mean())
            out[f"{cut}_{slice_name}_score_f"] = float(scores["f"][mask].mean())
            # Error of the mean forecast beside the quantile contrast (audit trigger 3).
            err_a = np.abs(year["actual"] - mean_a)[mask].mean()
            err_f = np.abs(year["actual"] - np.hypot(run["flow_f"], year["q_hat"]))[mask].mean()
            err_u = np.abs(year["actual"] - np.hypot(run["flow_f_ungated"], year["q_hat"]))[
                mask
            ].mean()
            out[f"{cut}_{slice_name}_mean_abs_error_a_f_ungated"] = [
                float(err_a),
                float(err_f),
                float(err_u),
            ]
            for label, column in (("q05", 1), ("q95", 9)):
                out[f"{cut}_{slice_name}_coverage_below_{label}_a_f"] = [
                    float((year["actual"][mask] < arm_q[arm][mask, column]).mean())
                    for arm in ("a", "f")
                ]
        out[f"{cut}_share_updated_f"] = run["updated_f"]
        out[f"{cut}_share_updated_f_ungated"] = run["updated_f_ungated"]
    (WORK_DIR / "f2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
