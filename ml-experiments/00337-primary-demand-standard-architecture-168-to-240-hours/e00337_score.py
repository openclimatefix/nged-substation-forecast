# ruff: noqa: E501, PLR0915
"""Build the arms of experiment 337 from the stored fits and score them at 168 to 240 hours.

Arms: default (Primary demand research default 3 at 168 to 240 hours, from experiment 335's stored
file), STD (the adapter plus mixture network, sorted, so its median replaces the default's median),
and STD-A (the adapter network's median with the default's offsets about its own median), with the
manual heuristic and climatology as references.

    source /tmp/pp337.sh   # or the PYTHONPATH in e00337_common.py
    uv run --no-sync python ml-experiments/00337-*/e00337_score.py
"""

from typing import Final

import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00311_score as base  # isort: skip  # ty: ignore[unresolved-import]
import e00330_common as c330  # isort: skip  # ty: ignore[unresolved-import]
import e00337_common as own  # isort: skip
import numpy as np
import polars as pl

KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
K_LEVEL: Final[float] = 1.0 - 0.01 / 2.0
GATE_168_240: Final[float] = 3.077
CONTRASTS: Final[tuple[str, ...]] = ("STD", "STDA")


def read_fits(*, head: str) -> pl.DataFrame:
    """Return the seed-averaged forecasts of one head per row."""
    columns = list(common.QUANTILES) if head == "std" else ["q50"]
    pieces = [
        pl.read_parquet(own.FIT_DIR / f"{head}_cut{label}_seed{seed}.parquet", columns=[*KEYS, *columns])
        for label in common.SCORED_CUTS
        for seed in c330.SEEDS
    ]  # fmt: skip
    out = pl.concat(pieces).group_by(KEYS).agg([pl.col(c).mean() for c in columns] + [pl.len().alias("n_seeds")])  # fmt: skip
    assert (out["n_seeds"] == len(c330.SEEDS)).all(), "a row is missing a seed"
    return out.drop("n_seeds")


def main() -> None:
    """Score the scored year and write the report."""
    quantiles = list(common.QUANTILES)
    rows = pl.read_parquet(own.E335_SCORED).filter(pl.col("lead_hours") > own.STORED_HORIZON_HOURS)
    prep = pl.read_parquet(own.PREP_DIR / "rows.parquet").select(KEYS).with_columns(
        mean_d=pl.Series(np.load(own.PREP_DIR / "mean.npy"))
    )  # fmt: skip
    std = read_fits(head="std")
    stda = read_fits(head="stda").rename({"q50": "stda_q50"})
    joined = rows.join(std, on=KEYS, how="left", validate="1:1", suffix="_std").join(stda, on=KEYS, how="left", validate="1:1").join(prep, on=KEYS, how="left", validate="1:1")  # fmt: skip
    missing = int(joined["q50_std"].is_null().sum()) if "q50_std" in joined.columns else int(joined["q50"].is_null().sum())  # fmt: skip
    assert missing == 0, f"{missing} scored rows have no STD forecast"
    std_cols = [f"{q}_std" if f"{q}_std" in joined.columns else q for q in quantiles]
    default_q = joined.select([f"d3_{q}" for q in quantiles]).to_numpy().astype(np.float64)
    median = default_q[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
    arms = {
        "default": default_q.astype(np.float32),
        "STD": np.sort(joined.select(std_cols).to_numpy().astype(np.float64), axis=1).astype(np.float32),
        "STDA": np.sort(joined["stda_q50"].to_numpy()[:, None] + (default_q - median), axis=1).astype(np.float32),
        "manual_heuristic": joined.select([f"mh_{q}" for q in quantiles]).to_numpy().astype(np.float32),
        "climatology": joined.select([f"cl_{q}" for q in quantiles]).to_numpy().astype(np.float32),
    }  # fmt: skip
    out_dir = own.WORK_DIR / "arms"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, q in arms.items():
        np.save(out_dir / f"{name}.npy", q)
    joined.select(KEYS).write_parquet(out_dir / "rows.parquet")

    site = joined["site"].to_numpy()
    th = base.thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    actual = joined["actual"].to_numpy().astype(np.float64)
    m = {a: base.per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = base.Blocks(blocks=joined["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(), seed=own.EXPERIMENT_ID)  # fmt: skip
    local = joined["valid_time"].dt.convert_time_zone("Europe/London")
    local_hh = local.dt.hour().to_numpy() * 2 + local.dt.minute().to_numpy() // 30
    lead = joined["lead_hours"].to_numpy()
    every = np.ones(joined.height, dtype=bool)
    cold = joined["cold_winter_night"].to_numpy()
    sunny = joined["sunny_daylight"].to_numpy()
    no_d10 = site != "D10"
    base_masks: dict[str, np.ndarray] = {
        "all rows": every, "cold winter nights": cold, "sunny daylight": sunny,
        "ordinary rows": ~cold & ~sunny, "Economy 7 night (00:30-07:30)": (local_hh >= 1) & (local_hh <= 14),
    }  # fmt: skip
    masks: dict[str, np.ndarray] = {
        **base_masks,
        **{f"{k} (without D10)": v & no_d10 for k, v in base_masks.items()},
        **{f"site {s}": site == s for s in common.SCORED_SITES},
        **{f"lead {int(lo)}-{int(hi)} h": (lead > lo) & (lead <= hi) for lo, hi in own.LEAD_BANDS},
    }  # fmt: skip
    keys_abs = ("qs", "mae", "tw95", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q01", "pin_q05", "pin_q95", "pin_q99")  # fmt: skip
    rep: dict = {"n_rows": joined.height, "n_cold_rows": int(cold.sum()), "n_sunny_rows": int(sunny.sum())}  # fmt: skip
    rep["by_mask"] = {a: {cut: {k: float(m[a][k][mask].mean()) for k in keys_abs} for cut, mask in masks.items() if mask.any()} for a in arms}  # fmt: skip
    rep["gate"] = {"default_168_240": float(m["default"]["qs"].mean()), "expected": GATE_168_240}
    rep["gate"]["pass"] = bool(
        abs(rep["gate"]["default_168_240"] - GATE_168_240) <= base.GATE_TOLERANCE
    )
    contrasts: dict = {}
    for treat in CONTRASTS:
        diff = m[treat]["qs"] - m["default"]["qs"]
        c: dict = {
            "gain_k2": boot.mean(values=diff, mask=every, level=K_LEVEL),
            "gain_95": boot.mean(values=diff, mask=every, level=0.95),
            "gain_without_d10_k2": boot.mean(values=diff, mask=no_d10, level=K_LEVEL),
            "gain_without_d10_95": boot.mean(values=diff, mask=no_d10, level=0.95),
            "relative_all": float(diff.mean() / m["default"]["qs"].mean()),
            "relative_without_d10": float(diff[no_d10].mean() / m["default"]["qs"][no_d10].mean()),
            "groups": {cut: boot.mean(values=diff, mask=mask) for cut, mask in masks.items() if mask.any()},
            "noninferiority": {},
        }  # fmt: skip
        month_means = [float(diff[boot.labels[boot.codes] == lab].mean()) for lab in boot.labels]
        c["months_won"] = int(sum(v < 0 for v in month_means))
        for metric, cut in (("qs", "all rows"), ("pin_q01", "all rows"), ("pin_q99", "all rows"), ("qs", "cold winter nights"), ("qs", "sunny daylight"), ("qs", "all rows (without D10)")):  # fmt: skip
            c["noninferiority"][f"{metric} on {cut}"] = boot.mean(values=m[treat][metric] - m["default"][metric], mask=masks[cut])  # fmt: skip
        c["noninferiority"]["tw95 relative"] = boot.relative(treat=m[treat]["tw95"], base=m["default"]["tw95"], mask=every)  # fmt: skip
        c["leave_one_site_out"] = {s: {"k2": boot.mean(values=diff, mask=site != s, level=K_LEVEL), "ci99": boot.mean(values=diff, mask=site != s, level=0.99)} for s in common.SCORED_SITES}  # fmt: skip
        c["site_alone"] = {s: boot.mean(values=diff, mask=site == s) for s in common.SCORED_SITES}
        for ref in ("manual_heuristic", "climatology"):
            c[f"minus_{ref}"] = boot.mean(values=m[treat]["qs"] - m[ref]["qs"], mask=every)
        contrasts[treat] = c
    contrasts["STD_minus_STDA"] = boot.mean(values=m["STD"]["qs"] - m["STDA"]["qs"], mask=every, level=0.95)  # fmt: skip
    rep["contrasts"] = contrasts
    mean_d = joined["mean_d"].to_numpy()
    rep["checks"] = {
        "median_minus_mean_forecast_per_site": {
            a: {
                s: float(
                    (
                        arms[a][site == s, common.MEDIAN_INDEX].astype(np.float64)
                        - mean_d[site == s]
                    ).mean()
                )
                for s in common.SCORED_SITES
            }
            for a in ("default", "STD", "STDA")
        },
        "mean_forecast_mae": {
            a: float(np.abs(arms[a].astype(np.float64).mean(axis=1) - actual).mean())
            for a in ("default", "STD", "STDA")
        },
        "median_mae": {a: float(m[a]["mae"].mean()) for a in ("default", "STD", "STDA")},
        "mean_forecast_mae_without_d10": {
            a: float(np.abs(arms[a].astype(np.float64).mean(axis=1) - actual)[no_d10].mean())
            for a in ("default", "STD", "STDA")
        },
    }
    own.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=own.REPORT_DIR / "report_scored.json", content=rep)
    print("gate", rep["gate"])
    print({a: round(rep["by_mask"][a]["all rows"]["qs"], 4) for a in arms})
    for treat in CONTRASTS:
        c = contrasts[treat]
        print(treat, "k2", c["gain_k2"], "without D10", c["gain_without_d10_k2"], "sunny", c["noninferiority"]["qs on sunny daylight"], "cold", c["noninferiority"]["qs on cold winter nights"])  # fmt: skip
    print("STD-STDA", contrasts["STD_minus_STDA"])


if __name__ == "__main__":
    main()
