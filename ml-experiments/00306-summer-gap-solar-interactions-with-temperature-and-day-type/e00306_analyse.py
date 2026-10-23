# ruff: noqa: E501
"""Score the control and the four solar-interaction arms against BG2 and write the report.

Reads experiment 266's master frame (BG2's and the stored arm D's per-row scores on 561,207 rows)
and this experiment's forecast frames (`work/dp_preds/<arm>_scored_<site>.parquet`), joins on
(site, run, valid_time), and writes `work/report.json` and `work/rows.parquet`.

    uv run --no-sync python ml-experiments/00306-*/e00306_analyse.py
"""

import json
from typing import Any, Final

import e00306_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E266: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
ARMS: Final = ("a2", "s_both", "s_heat", "s_wknd", "s_placebo")
NEW_ARMS: Final = ARMS[1:]
SITES_ALL: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
TARGET: Final = ("D5", "D9")
PLACEBO_SITES: Final = ("D10", "D16")
AMJ: Final = ("2026-04", "2026-05", "2026-06")
JAS: Final = ("2025-07", "2025-08", "2025-09")
LEVELS: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
TAILS: Final = (0, 1, 9, 10)
N_RESAMPLES: Final = 20_000
GATE_TOLERANCE: Final = 1e-3


def qs(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 quantiles, per row."""
    err = actual[:, None] - q
    return 2.0 * np.maximum(LEVELS * err, (LEVELS - 1.0) * err).mean(axis=1)


def site_week_draws(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    site: np.ndarray,
    week: np.ndarray,
    sites: tuple[str, ...],
    seed: int = 0,
) -> dict:
    """Mean of `values` over `mask` averaged over `sites`, with its week-block bootstrap draws.

    Each draw resamples whole ISO weeks of initialisation time, the same weeks at every
    substation, and averages the per-substation means. The draws depend only on `mask`, `site`,
    `week`, and `seed`, so two value arrays on one subset share their draws.
    """
    keep = mask & np.isfinite(values) & np.isin(site, sites)
    weeks = np.unique(week[keep])
    index = {w: i for i, w in enumerate(weeks)}
    codes = np.array([index[w] for w in week[keep]])
    sums = np.zeros((len(sites), len(weeks)))
    counts = np.zeros((len(sites), len(weeks)))
    for k, s in enumerate(sites):
        pick = site[keep] == s
        sums[k] = np.bincount(codes[pick], weights=values[keep][pick], minlength=len(weeks))
        counts[k] = np.bincount(codes[pick], minlength=len(weeks))
    draws = np.random.default_rng(seed).integers(0, len(weeks), size=(N_RESAMPLES, len(weeks)))
    num = sums[:, draws].sum(axis=2)
    den = counts[:, draws].sum(axis=2)
    per_site = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    est_sites = np.divide(
        sums.sum(axis=1),
        counts.sum(axis=1),
        out=np.full(len(sites), np.nan),
        where=counts.sum(axis=1) > 0,
    )
    return {
        "est": float(np.nanmean(est_sites)),
        "draws": np.nanmean(per_site, axis=0),
        "n": int(keep.sum()),
        "weeks": len(weeks),
    }


def summary(*, draws: np.ndarray, est: float, extra: dict | None = None) -> dict:
    """Return the estimate with its 95% interval."""
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi), **(extra or {})}


def load() -> pl.DataFrame:
    """Return the master frame joined to every arm's per-row scores and medians."""
    master = pl.read_parquet(E266).with_columns(
        week=pl.col("power_fcst_init_time").dt.truncate("1w").dt.strftime("%Y-%m-%d")
    )
    out = master
    for arm in ARMS:
        frames = []
        for site in SITES_ALL:
            f = pl.read_parquet(
                common.WORK_DIR / "dp_preds" / f"{arm}_scored_{site}.parquet",
            ).with_columns(site=pl.lit(site))
            frames.append(f)
        arm_frame = pl.concat(frames)
        q = arm_frame.select(list(QCOLS)).to_numpy().astype(np.float64)
        scored = (
            arm_frame.select("site", "run", "valid_time", "mean", "actual")
            .with_columns(
                pl.Series(
                    f"qs_{arm}", qs(actual=arm_frame["actual"].to_numpy().astype(np.float64), q=q)
                ),
                pl.Series(f"med_{arm}", q[:, 5]),
                *(
                    pl.Series(f"{c}_{arm}", q[:, i])
                    for c, i in (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
                ),
            )
            .rename({"mean": f"mean_{arm}"})
            .drop("actual")
        )
        out = out.join(scored, on=["site", "run", "valid_time"], how="left", validate="1:1")
        assert out[f"qs_{arm}"].null_count() == 0, f"{arm} lacks master rows"
    return out


E251_ARMS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work" / "arms" / "scored"
)  # fmt: skip


def add_shifted_stored_arm_d(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Post hoc: shift stored arm D's quantiles by each arm's change in the engine mean.

    Stored arm D (experiment 220) is a close relative of this experiment's control, not the
    control itself. Shifting its quantiles by `mean_arm - mean_a2` follows experiments 251 and 289.
    """
    d = np.load(E251_ARMS / "d.npy").astype(np.float64)
    assert d.shape[0] == frame.height, "stored arm D and the master rows differ"
    actual = frame["actual"].to_numpy().astype(np.float64)
    new = {"qs_ds_a2": qs(actual=actual, q=d)}
    for arm in NEW_ARMS:
        shift = (frame[f"mean_{arm}"] - frame["mean_a2"]).to_numpy().astype(np.float64)
        new[f"qs_ds_{arm}"] = qs(actual=actual, q=d + shift[:, None])
    return frame.with_columns([pl.Series(k, v) for k, v in new.items()])


def shifted_contrasts(
    *, frame: pl.DataFrame, mask: np.ndarray, sites: tuple[str, ...], seed: int
) -> dict:
    """Gap of stored arm D and each shifted arm to BG2, with the closure."""
    args = {
        "mask": mask,
        "site": frame["site"].to_numpy(),
        "week": frame["week"].to_numpy(),
        "sites": sites,
        "seed": seed,
    }
    bg2 = frame["qs_bg2"].to_numpy()
    base = frame["qs_ds_a2"].to_numpy()
    gap = site_week_draws(values=base - bg2, **args)
    out: dict[str, Any] = {"excess_stored_d": summary(draws=gap["draws"], est=gap["est"])}
    for arm in NEW_ARMS:
        red = site_week_draws(values=base - frame[f"qs_ds_{arm}"].to_numpy(), **args)
        out[arm] = {
            "reduction": summary(draws=red["draws"], est=red["est"]),
            "closure": summary(draws=red["draws"] / gap["draws"], est=red["est"] / gap["est"]),
        }
    return out


def subset(
    *, frame: pl.DataFrame, months: tuple[str, ...], lead: float | None, daylight: bool
) -> np.ndarray:
    """Return the boolean mask of a season, lead limit, and daylight choice."""
    mask = frame["month"].is_in(list(months)).to_numpy()
    if lead is not None:
        mask &= (frame["lead_hours"] <= lead).to_numpy()
    if daylight:
        mask &= frame["daylight"].to_numpy()
    return mask


def contrasts(*, frame: pl.DataFrame, mask: np.ndarray, sites: tuple[str, ...], seed: int) -> dict:
    """Score, gap, reduction, and closure of every new arm on one subset."""
    args = {
        "mask": mask,
        "site": frame["site"].to_numpy(),
        "week": frame["week"].to_numpy(),
        "sites": sites,
        "seed": seed,
    }
    bg2 = frame["qs_bg2"].to_numpy()
    control = frame["qs_a2"].to_numpy()
    gap = site_week_draws(values=control - bg2, **args)
    out: dict[str, Any] = {
        "n": gap["n"],
        "weeks": gap["weeks"],
        "score_bg2": site_week_draws(values=bg2, **args)["est"],
        "score_control": site_week_draws(values=control, **args)["est"],
        "excess_control": summary(draws=gap["draws"], est=gap["est"]),
    }
    for arm in NEW_ARMS:
        score = frame[f"qs_{arm}"].to_numpy()
        red = site_week_draws(values=control - score, **args)
        ratio = red["draws"] / gap["draws"]
        out[arm] = {
            "score": site_week_draws(values=score, **args)["est"],
            "reduction": summary(draws=red["draws"], est=red["est"]),
            "closure": summary(
                draws=ratio, est=red["est"] / gap["est"] if gap["est"] else float("nan")
            ),
            "excess": gap["est"] - red["est"],
        }
    return out


def tail_table(*, frame: pl.DataFrame, mask: np.ndarray, sites: tuple[str, ...]) -> dict:
    """Coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles, control and s_both."""
    keep = mask & np.isin(frame["site"].to_numpy(), sites)
    actual = frame["actual"].to_numpy().astype(np.float64)[keep]
    out = {}
    for arm in ("a2", "s_both"):
        row = {}
        for label, col, level in (
            ("q01", "q01", 0.01),
            ("q05", "q05", 0.05),
            ("q95", "q95", 0.95),
            ("q99", "q99", 0.99),
        ):
            q = frame[f"{col}_{arm}"].to_numpy().astype(np.float64)[keep]
            err = actual - q
            row[label] = {
                "coverage_below": float((actual <= q).mean()),
                "pinball": float(np.maximum(level * err, (level - 1) * err).mean()),
            }
        out[arm] = row
    out["bg2_note"] = (
        "BG2 quantiles are not in the master frame; tail rows compare the control and the interaction arm"
    )
    return out


def main() -> None:
    """Write the report."""
    frame = add_shifted_stored_arm_d(frame=load())
    report: dict[str, Any] = {"rows": frame.height}
    gate = 0.0
    for site in SITES_ALL:
        new = pl.read_parquet(common.WORK_DIR / "dp_preds" / f"a2_scored_{site}.parquet")
        old = pl.read_parquet(common.E136_WORK_DIR / "dp_preds" / f"a2_scored_{site}.parquet")
        joined = new.join(old, on=["run", "valid_time"], suffix="_old", validate="1:1")
        assert joined.height == old.height == new.height, site
        for c in QCOLS:
            gap = (joined[c] - joined[f"{c}_old"]).abs().to_numpy()
            gate = max(gate, float(gap.max()))
    assert gate < GATE_TOLERANCE, f"control differs from experiment 136's stored A2: {gate}"
    report["gate_control_vs_experiment_136_a2_max_abs_quantile_diff"] = gate
    report["stored_arm_d_vs_control_max_abs_per_row_score_diff"] = float(
        np.abs(frame["qs_a2"].to_numpy() - frame["qs_d"].to_numpy()).max()
    )
    report["stored_arm_d_vs_control_mean_abs_per_row_score_diff"] = float(
        np.abs(frame["qs_a2"].to_numpy() - frame["qs_d"].to_numpy()).mean()
    )
    amj_day36 = subset(frame=frame, months=AMJ, lead=36.0, daylight=True)
    jas_day36 = subset(frame=frame, months=JAS, lead=36.0, daylight=True)
    report["headline"] = {}
    for label, sites in (
        ("D5", ("D5",)),
        ("D9", ("D9",)),
        ("D5_D9", TARGET),
        ("placebo_D10_D16", PLACEBO_SITES),
        ("D10", ("D10",)),
        ("D16", ("D16",)),
        ("D11", ("D11",)),
        ("D14", ("D14",)),
    ):
        report["headline"][f"amj_{label}"] = contrasts(
            frame=frame, mask=amj_day36, sites=sites, seed=1
        )
        report["headline"][f"jas_{label}"] = contrasts(
            frame=frame, mask=jas_day36, sites=sites, seed=2
        )
    report["post_hoc_shifted_stored_arm_d"] = {
        f"{season}_{label}": shifted_contrasts(frame=frame, mask=mask, sites=sites, seed=seed)
        for season, mask, seed in (("amj", amj_day36, 1), ("jas", jas_day36, 2))
        for label, sites in (
            ("D5", ("D5",)),
            ("D9", ("D9",)),
            ("D5_D9", TARGET),
            ("placebo_D10_D16", PLACEBO_SITES),
        )
    }
    all_hours_amj = subset(frame=frame, months=AMJ, lead=None, daylight=True)
    report["amj_daylight_all_leads"] = {
        s: contrasts(frame=frame, mask=all_hours_amj, sites=(s,), seed=3) for s in SITES_ALL
    }
    clear = (
        amj_day36
        & (frame["realised_csi"].fill_null(0.0).to_numpy() > 0.8)
        & frame["hour"].is_between(10, 14).to_numpy()
    )
    report["amj_clear_midday_D5"] = contrasts(frame=frame, mask=clear, sites=("D5",), seed=4)
    report["tails_amj_daylight_D5_D9"] = tail_table(frame=frame, mask=amj_day36, sites=TARGET)
    # Checks: median minus mean forecast per site, and mean-forecast error beside the score.
    checks = {}
    for arm in ("a2", "s_both"):
        checks[arm] = {
            s: float(
                np.mean(
                    (frame[f"med_{arm}"] - frame[f"mean_{arm}"])
                    .filter(frame["site"] == s)
                    .to_numpy()
                )
            )
            for s in SITES_ALL
        }
    report["median_minus_mean_by_site"] = checks
    actual = frame["actual"].to_numpy().astype(np.float64)
    mae = {a: np.abs(actual - frame[f"mean_{a}"].to_numpy()) for a in ARMS}
    report["mean_forecast_mae_amj_daylight"] = {
        a: {s: float(mae[a][amj_day36 & (frame["site"].to_numpy() == s)].mean()) for s in SITES_ALL}
        for a in ARMS
    }
    # Monthly excess for the chart: control and each arm against BG2, daylight, lead <= 36, D5 and D9.
    months = sorted(frame["month"].unique().to_list())
    monthly: dict[str, Any] = {}
    for s in (*TARGET, *PLACEBO_SITES):
        monthly[s] = {}
        for m in months:
            sel = (
                (frame["site"].to_numpy() == s)
                & (frame["month"].to_numpy() == m)
                & frame["daylight"].to_numpy()
                & (frame["lead_hours"].to_numpy() <= 36.0)
            )
            if not sel.any():
                continue
            bg2 = frame["qs_bg2"].to_numpy()[sel].mean()
            monthly[s][m] = {a: float(frame[f"qs_{a}"].to_numpy()[sel].mean() - bg2) for a in ARMS}
    report["monthly_excess"] = monthly
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "report.json").write_text(json.dumps(report, indent=2))
    frame.write_parquet(common.WORK_DIR / "rows.parquet")
    h = report["headline"]
    print(
        "gate",
        gate,
        "stored arm D vs control, mean abs per-row score diff",
        report["stored_arm_d_vs_control_mean_abs_per_row_score_diff"],
    )
    for key in ("amj_D5", "amj_D9", "amj_D5_D9", "amj_placebo_D10_D16", "jas_D5_D9"):
        c = h[key]
        print(key, "n", c["n"], "excess_control", round(c["excess_control"]["est"], 3))
        for arm in NEW_ARMS:
            a = c[arm]
            print(
                "  ",
                arm,
                "reduction",
                round(a["reduction"]["est"], 3),
                (round(a["reduction"]["lo"], 3), round(a["reduction"]["hi"], 3)),
                "closure",
                round(a["closure"]["est"], 3),
                (round(a["closure"]["lo"], 2), round(a["closure"]["hi"], 2)),
            )


if __name__ == "__main__":
    main()
