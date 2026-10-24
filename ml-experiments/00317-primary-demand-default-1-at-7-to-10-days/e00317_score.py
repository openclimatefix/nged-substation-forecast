# ruff: noqa: E501, PLR0912, PLR0915
"""Score arm A out to 240 hours against the manual heuristic and climatology, and write the report.

Rows: the scored year (runs from 2025-07-02 to 2026-06-30, origin 09:00 UTC), six Primaries, rows
with a reading outside the hand-filtered events, and a stored forecast of both benchmarks. The
quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Intervals resample the 12 months of the forecast run's initialisation time, 20,000
draws. "A" is the convex chain of experiments 136, 155, 160, and 164 rebuilt to 240 hours by
`e00317_chain.py`. "A+H+S" is experiment 251's arm 1 (A plus the network's median correction and
the two-number solar summary), which exists below 168 hours only.

    uv run --no-sync python ml-experiments/00317-*/e00317_score.py
"""

import itertools
from typing import Final

import e00317_common as common
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
BAND_EDGES: Final[np.ndarray] = np.arange(0.0, 241.0, 24.0)
BAND_LABELS: Final[tuple[str, ...]] = tuple(
    f"{int(a)}-{int(b)} h" for a, b in itertools.pairwise(BAND_EDGES)
)
TAILS: Final[tuple[tuple[str, int], ...]] = (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
SCALE_E87: Final[float] = 100.0
"""Experiment 87 stores quantiles as fractions of effective capacity; the chain stores points."""
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})


def pin(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the pinball loss of each quantile, shape (rows, 11)."""
    error = actual[:, None] - q
    return np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)


def load_chain() -> pl.DataFrame:
    """Return the chain's scored-year rows at the six Primaries."""
    pieces = [
        pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet")
        .filter(pl.col("period") == "scored")
        .with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    frame = pl.concat(pieces)
    return frame.filter(~pl.col("switching_mask") & pl.col("actual").is_finite())


def read_reference(*, name: str, tag: str) -> pl.DataFrame:
    """Return a stored benchmark's quantiles, in points, with columns suffixed by `tag`."""
    path = common.E87_PREDICTIONS / name / "predictions.parquet"
    frame = pl.read_parquet(path, columns=[*common.KEYS, *common.QUANTILES, "actual"]).unique(
        subset=common.KEYS
    )
    return frame.select(
        *common.KEYS,
        pl.col("actual").alias(f"actual_{tag}"),
        *[(pl.col(q) * SCALE_E87).alias(f"{tag}_{q}") for q in common.QUANTILES],
    )


def flags(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add sunny, daylight, and cold-day flags, which depend only on the site and valid time."""
    sunny = pl.read_parquet(common.E178_ROWS, columns=["site", "valid_time", "sunny"]).unique(
        subset=["site", "valid_time"]
    )
    other = pl.read_parquet(
        common.E220_ARMS / "rows.parquet", columns=["site", "valid_time", "daylight", "cold2"]
    ).unique(subset=["site", "valid_time"])
    joined = rows.join(sunny, on=["site", "valid_time"], how="left", validate="m:1").join(
        other, on=["site", "valid_time"], how="left", validate="m:1"
    )
    local_hour = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour()
    return joined.with_columns(
        sunny_daylight=(pl.col("sunny") & pl.col("daylight")).fill_null(False),
        cold_winter_night=(
            pl.col("cold2").fill_null(False)
            & ~pl.col("daylight").fill_null(True)
            & pl.col("valid_time").dt.month().is_in(list(WINTER_MONTHS))
            & ((local_hour >= 18) | (local_hour < 7))
        ),
        known_flags=pl.col("daylight").is_not_null(),
    )


def interval_mean(
    *, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> dict:
    """Mean of `values` over `mask` with a block-resampled 95% interval."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    sums = np.bincount(codes, weights=values[mask], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {
        "est": float(sums.sum() / counts.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(mask.sum()),
    }


def interval_gain(
    *,
    treat: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Relative gain `1 - mean(treat) / mean(ref)` over `mask`, plus the difference, with intervals."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    if n == 0:
        return {"gain": float("nan")}
    st = np.bincount(codes, weights=treat[mask], minlength=n)
    sr = np.bincount(codes, weights=ref[mask], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    c = counts[draws].sum(axis=1)
    mt, mr = st[draws].sum(axis=1) / c, sr[draws].sum(axis=1) / c
    gain = 1.0 - mt / mr
    diff = mt - mr
    return {
        "gain": float(1.0 - st.sum() / sr.sum()),
        "gain_lo": float(np.percentile(gain, 2.5)),
        "gain_hi": float(np.percentile(gain, 97.5)),
        "diff": float((st.sum() - sr.sum()) / counts.sum()),
        "diff_lo": float(np.percentile(diff, 2.5)),
        "diff_hi": float(np.percentile(diff, 97.5)),
        "treat": float(st.sum() / counts.sum()),
        "ref": float(sr.sum() / counts.sum()),
        "n": int(mask.sum()),
    }


def main() -> None:
    """Join the arms, run the gates, score every lead band and group, and write the report."""
    chain = load_chain()
    manual = read_reference(name="ref_manual_heuristic", tag="mh")
    clim = read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=common.KEYS, how="inner", validate="1:1").join(
        clim, on=common.KEYS, how="inner", validate="1:1"
    )
    actual_gap = float(
        np.abs(rows["actual"].to_numpy() - rows["actual_mh"].to_numpy() * SCALE_E87).max()
    )
    clim_gap = float(
        np.abs(rows["actual"].to_numpy() - rows["actual_cl"].to_numpy() * SCALE_E87).max()
    )
    rows = rows.filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
    stored = (
        pl.read_parquet(common.E251_ARMS / "rows.parquet")
        .select(["site", "power_fcst_init_time", "valid_time"])
        .with_row_index("i251")
    )
    arms251 = {
        n: np.load(common.E251_ARMS / f"{n}.npy")
        for n in ("arm1", "d", "bg2", "manual_heuristic", "climatology")
    }
    rows = rows.join(
        stored, on=["site", "power_fcst_init_time", "valid_time"], how="left", validate="1:1"
    )
    rows = flags(rows=rows)
    n = rows.height
    actual = rows["actual"].to_numpy().astype(np.float64)
    lead = rows["lead_hours"].to_numpy()
    band = np.searchsorted(BAND_EDGES, lead, side="left") - 1
    band = np.clip(band, 0, len(BAND_LABELS) - 1)
    qa = rows.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    qm = rows.select([f"mh_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    qc = rows.select([f"cl_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    in251 = rows["i251"].is_not_null().to_numpy()
    idx = rows["i251"].fill_null(0).to_numpy()
    q_hs = arms251["arm1"][idx].astype(np.float64)
    q_a251 = arms251["d"][idx].astype(np.float64)
    q_bg2 = arms251["bg2"][idx].astype(np.float64)
    arms = {"A": qa, "manual_heuristic": qm, "climatology": qc, "A+H+S": q_hs, "BG2": q_bg2}
    loss = {k: 2.0 * pin(actual=actual, q=v).mean(axis=1) for k, v in arms.items()}
    rep: dict = {
        "n_rows": n,
        "n_rows_in_251": int(in251.sum()),
        "actual_gap_manual": actual_gap,
        "actual_gap_clim": clim_gap,
    }
    # Reproduction gate: A on experiment 251's rows against experiment 220's stored A, and the stored A+H+S.
    sel = in251
    rep["gate"] = {
        "a_vs_stored_max_abs_diff": float(np.abs(qa[sel] - q_a251[sel]).max()),
        "a_vs_stored_score_diff": float(
            loss["A"][sel].mean()
            - (2.0 * pin(actual=actual[sel], q=q_a251[sel]).mean(axis=1)).mean()
        ),
        "a_hs_six": float(loss["A+H+S"][sel].mean()),
        "a_six_on_251_rows": float(loss["A"][sel].mean()),
        "a_hs_five": float(
            loss["A+H+S"][sel & np.isin(rows["site"].to_numpy(), common.DECISION_SITES)].mean()
        ),
        "rows_missing_in_251": int((~in251 & (lead <= common.STORED_HORIZON_HOURS)).sum()),
        "rows_251_missing_here": int(stored.height - in251.sum()),
    }
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    site = rows["site"].to_numpy()
    month_of_valid = rows["valid_time"].dt.month().to_numpy()
    groups: dict[str, np.ndarray] = {
        "all rows": np.ones(n, dtype=bool),
        "cold winter nights": rows["cold_winter_night"].to_numpy(),
        "sunny daylight": rows["sunny_daylight"].to_numpy(),
        "daylight": rows["daylight"].fill_null(False).to_numpy(),
        "winter (Nov-Mar)": np.isin(month_of_valid, list(WINTER_MONTHS)),
    }
    rep["group_rows"] = {k: int(v.sum()) for k, v in groups.items()}
    windows = {
        "0-168 h": (lead <= 168.0),
        "168-240 h": (lead > 168.0),
        "168-216 h": (lead > 168.0) & (lead <= 216.0),
        "216-240 h": (lead > 216.0),
    }
    # Per-band scores, contrasts, and tails.
    by_band: dict = {}
    for i, label in enumerate(BAND_LABELS):
        mask = band == i
        entry: dict = {"n": int(mask.sum())}
        for arm in ("A", "manual_heuristic", "climatology"):
            entry[arm] = interval_mean(values=loss[arm], mask=mask, blocks=month, rng=rng)
        for ref in ("manual_heuristic", "climatology"):
            entry[f"A_vs_{ref}"] = interval_gain(
                treat=loss["A"], ref=loss[ref], mask=mask, blocks=month, rng=rng
            )
        entry["manual_vs_climatology"] = interval_gain(
            treat=loss["manual_heuristic"],
            ref=loss["climatology"],
            mask=mask,
            blocks=month,
            rng=rng,
        )
        both = mask & in251
        if both.any():
            for arm in ("A+H+S", "BG2", "A"):
                entry[f"{arm}_on_251_rows"] = interval_mean(
                    values=loss[arm], mask=both, blocks=month, rng=rng
                )
            for ref in ("manual_heuristic", "climatology"):
                entry[f"A+H+S_vs_{ref}"] = interval_gain(
                    treat=loss["A+H+S"], ref=loss[ref], mask=both, blocks=month, rng=rng
                )
        tails: dict = {}
        for arm in ("A", "manual_heuristic", "climatology"):
            q = arms[arm]
            err = actual[:, None] - q
            p = pin(actual=actual, q=q)
            tails[arm] = {
                **{f"{t}_pin": float(p[mask, pos].mean()) for t, pos in TAILS},
                **{f"{t}_cov": float((err[mask, pos] <= 0).mean()) for t, pos in TAILS},
                "interval90_cov": float(((actual >= q[:, 1]) & (actual <= q[:, 9]))[mask].mean()),
                "mean_forecast_error_median": float(np.abs(err[mask, 5]).mean()),
                "median_minus_mean_q": float((q[mask, 5] - q[mask].mean(axis=1)).mean()),
            }
        entry["tails"] = tails
        entry["mae"] = {
            arm: float(np.abs(actual - arms[arm][:, 5])[mask].mean())
            for arm in ("A", "manual_heuristic", "climatology")
        }
        by_band[label] = entry
    rep["by_band"] = by_band
    # Windows by protected group.
    by_group: dict = {}
    for gname, gmask in groups.items():
        by_group[gname] = {}
        for wname, wmask in windows.items():
            mask = gmask & wmask
            if mask.sum() < 50:
                continue
            e: dict = {"n": int(mask.sum())}
            for arm in ("A", "manual_heuristic", "climatology"):
                e[arm] = interval_mean(values=loss[arm], mask=mask, blocks=month, rng=rng)
            for ref in ("manual_heuristic", "climatology"):
                e[f"A_vs_{ref}"] = interval_gain(
                    treat=loss["A"], ref=loss[ref], mask=mask, blocks=month, rng=rng
                )
            e["mae_A"] = float(np.abs(actual - qa[:, 5])[mask].mean())
            e["mae_manual"] = float(np.abs(actual - qm[:, 5])[mask].mean())
            e["mae_clim"] = float(np.abs(actual - qc[:, 5])[mask].mean())
            by_group[gname][wname] = e
    rep["by_group"] = by_group
    # Per substation, 168-240 h, for the leave-one-out check.
    rep["by_site_168_240"] = {
        s: {
            ref: interval_gain(
                treat=loss["A"],
                ref=loss[ref],
                mask=(site == s) & windows["168-240 h"],
                blocks=month,
                rng=rng,
            )
            for ref in ("manual_heuristic", "climatology")
        }
        for s in common.SCORED_SITES
    }
    # Per band, one site left out, for the 168-240 h gain over each benchmark.
    rep["leave_one_out"] = {
        s: {
            label: {
                ref: interval_gain(
                    treat=loss["A"],
                    ref=loss[ref],
                    mask=(site != s) & (band == i),
                    blocks=month,
                    rng=rng,
                )["gain"]
                for ref in ("manual_heuristic", "climatology")
            }
            for i, label in enumerate(BAND_LABELS)
            if i >= 6
        }
        for s in common.SCORED_SITES
    }
    # Effect of H and S below 168 h: A+H+S minus A on experiment 251's rows, by band.
    rep["h_and_s_effect"] = {
        label: interval_mean(
            values=loss["A+H+S"] - loss["A"], mask=(band == i) & in251, blocks=month, rng=rng
        )
        for i, label in enumerate(BAND_LABELS)
        if i < 7
    }
    # Median forecast minus mean forecast, per site (the 11 quantiles' mean), scored year, all leads.
    rep["median_minus_mean_by_site"] = {
        s: float((qa[site == s, 5] - qa[site == s].mean(axis=1)).mean())
        for s in common.SCORED_SITES
    }
    # Experiment 251's headline contrast, reproduced: A+H+S minus the manual heuristic, leads to 168 h.
    both = in251
    rep["a_hs_vs_manual_all_leads_to_168"] = interval_gain(
        treat=loss["A+H+S"], ref=loss["manual_heuristic"], mask=both, blocks=month, rng=rng
    )
    rep["a_hs_vs_bg2_all_leads_to_168"] = interval_gain(
        treat=loss["A+H+S"], ref=loss["BG2"], mask=both, blocks=month, rng=rng
    )
    # Rows per band, to show how many origins reach past 168 h.
    rep["rows_per_band"] = {label: int((band == i).sum()) for i, label in enumerate(BAND_LABELS)}
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=rep)
    rows.select(
        "site",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "actual",
        "cold_winter_night",
        "sunny_daylight",
        "daylight",
        *[pl.Series(f"A_{q}", qa[:, k].astype(np.float32)) for k, q in enumerate(common.QUANTILES)],
        *[pl.Series(f"M_{q}", qm[:, k].astype(np.float32)) for k, q in enumerate(common.QUANTILES)],
        *[pl.Series(f"C_{q}", qc[:, k].astype(np.float32)) for k, q in enumerate(common.QUANTILES)],
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")
    print("gate", rep["gate"], flush=True)
    for label, e in by_band.items():
        print(
            label,
            e["n"],
            "A",
            round(e["A"]["est"], 3),
            "MH",
            round(e["manual_heuristic"]["est"], 3),
            "CL",
            round(e["climatology"]["est"], 3),
            "gain MH",
            round(e["A_vs_manual_heuristic"]["gain"], 3),
            "gain CL",
            round(e["A_vs_climatology"]["gain"], 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
