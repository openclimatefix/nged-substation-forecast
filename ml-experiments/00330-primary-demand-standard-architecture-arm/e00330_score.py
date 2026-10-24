# ruff: noqa: E501, PLR0915
"""Build the arms of experiment 330 from the stored fits and score them at 0.5 to 168 hours.

Arms: default (Primary demand research default 3 below 168 hours, which is experiment 251's arm 1),
G (experiment 314's gated mixture head), STD (the adapter plus mixture network, uncentred, so its
median replaces the default's median), and STD-A (the adapter network's median with the default's
offsets about its own median). The replicate year compares with experiment 169's arm A bands
(experiment 314's replicate reference), centred the same way.

    export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*:ml-experiments/00330-*
    uv run --no-sync python ml-experiments/00330-*/e00330_score.py
"""

from typing import Final

import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00311_score as base  # isort: skip  # ty: ignore[unresolved-import]
import e00330_common as own  # isort: skip
import numpy as np
import polars as pl

CONTRASTS: Final[tuple[str, ...]] = ("STD", "STDA", "G")
K_LEVEL: Final[float] = 1.0 - 0.01 / 2.0
LEAD_BANDS: Final[tuple[tuple[float, float], ...]] = tuple((24.0 * d, 24.0 * (d + 1)) for d in range(7))  # fmt: skip
KEYS: Final[list[str]] = ["site", "run", "valid_time"]


def read_fits(*, head: str, cuts: tuple[str, ...]) -> pl.DataFrame:
    """Return the seed-averaged forecasts of one head per (site, run, valid_time)."""
    columns = list(common.QUANTILES) if head == "std" else ["q50"]
    pieces = [
        pl.read_parquet(
            own.FIT_DIR / f"{head}_cut{label}_seed{seed}.parquet", columns=[*KEYS, *columns]
        )
        for label in cuts
        for seed in own.SEEDS
    ]
    out = (
        pl.concat(pieces)
        .group_by(KEYS)
        .agg([pl.col(c).mean() for c in columns] + [pl.len().alias("n_seeds")])
    )
    assert (out["n_seeds"] == len(own.SEEDS)).all(), "a row is missing a seed"
    return out.drop("n_seeds")


def build_arms(
    *, default_q: np.ndarray, std_q: np.ndarray, stda_median: np.ndarray
) -> dict[str, np.ndarray]:
    """Return STD (sorted mixture quantiles) and STD-A (the default's offsets about the new median)."""
    median = default_q[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1].astype(np.float64)
    offsets = default_q.astype(np.float64) - median
    return {
        "STD": np.sort(std_q.astype(np.float64), axis=1).astype(np.float32),
        "STDA": np.sort(stda_median[:, None] + offsets, axis=1).astype(np.float32),
    }


def main() -> None:
    """Score the scored year and the replicate year and write the report."""
    rows = pl.read_parquet(base.SCORED_START / "rows.parquet")
    std = read_fits(head="std", cuts=common.SCORED_CUTS)
    stda = read_fits(head="stda", cuts=common.SCORED_CUTS).rename({"q50": "stda_q50"})
    joined = rows.select("row", *KEYS).join(std, on=KEYS, how="left", validate="1:1").join(stda, on=KEYS, how="left", validate="1:1")  # fmt: skip
    assert joined["q50"].null_count() == 0, "a scored row has no STD forecast"
    assert joined["stda_q50"].null_count() == 0, "a scored row has no STD-A forecast"
    default_q = np.load(common.E251_SCORED / "arm1.npy")
    arms = {"default": default_q.astype(np.float32)}
    arms["G"] = np.load(own.WORK_DIR.parent.parent / "00314-primary-demand-gated-or-pinball-band-head" / "work" / "arms" / "scored" / "G.npy")  # fmt: skip
    arms.update(build_arms(default_q=default_q, std_q=joined.select(list(common.QUANTILES)).to_numpy(), stda_median=joined["stda_q50"].to_numpy()))  # fmt: skip
    for name in base.BENCHMARKS:
        arms[name] = np.load(common.E251_SCORED / f"{name}.npy")
    out_dir = own.WORK_DIR / "arms" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, q in arms.items():
        np.save(out_dir / f"{name}.npy", q)

    daylight = rows["daylight"].to_numpy()
    sunny = rows["sunny"].to_numpy() & daylight
    cold_night = rows["cold2"].to_numpy() & ~daylight
    site = rows["site"].to_numpy()
    th = base.thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64)
    m = {a: base.per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = base.Blocks(
        blocks=rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(), seed=own.EXPERIMENT_ID
    )
    local = rows["valid_time"].dt.convert_time_zone("Europe/London")
    local_hh = local.dt.hour().to_numpy() * 2 + local.dt.minute().to_numpy() // 30
    lead = rows["lead_hours"].to_numpy()
    every = np.ones(rows.height, dtype=bool)
    masks: dict[str, np.ndarray] = {
        "all rows": every, "cold winter nights": cold_night, "sunny daylight": sunny,
        "ordinary rows": ~cold_night & ~sunny, "Christmas": rows["christmas"].to_numpy(),
        "bank holidays": rows["bank_holiday"].to_numpy(),
        "Economy 7 night (00:30-07:30)": (local_hh >= 1) & (local_hh <= 14),
        "leads 12-36 h": (lead >= 12) & (lead < 36),
        **{f"site {s}": site == s for s in common.SCORED_SITES},
        **{f"lead {int(lo)}-{int(hi)} h": (lead >= lo) & (lead < hi) for lo, hi in LEAD_BANDS},
    }  # fmt: skip
    keys_abs = ("qs", "mae", "tw95", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q01", "pin_q05", "pin_q95", "pin_q99")  # fmt: skip
    rep: dict = {"n_rows": rows.height, "n_sunny_daylight_rows": int(sunny.sum())}
    rep["by_mask"] = {a: {cut: {k: float(m[a][k][mask].mean()) for k in keys_abs} for cut, mask in masks.items() if mask.any()} for a in arms}  # fmt: skip
    five = np.isin(site, ["D5", "D10", "D11", "D14", "D16"])
    rep["gate"] = {"default_six": float(m["default"]["qs"].mean()), "default_five": float(m["default"]["qs"][five].mean())}  # fmt: skip
    rep["gate"]["pass"] = bool(abs(rep["gate"]["default_six"] - base.GATE_SIX) <= base.GATE_TOLERANCE and abs(rep["gate"]["default_five"] - base.GATE_FIVE) <= base.GATE_TOLERANCE)  # fmt: skip
    contrasts: dict = {}
    for treat in CONTRASTS:
        diff = m[treat]["qs"] - m["default"]["qs"]
        c: dict = {
            "gain_k2": boot.mean(values=diff, mask=every, level=K_LEVEL),
            "gain_95": boot.mean(values=diff, mask=every, level=0.95),
            "groups": {cut: boot.mean(values=diff, mask=mask) for cut, mask in masks.items() if mask.any()},
            "noninferiority": {},
        }  # fmt: skip
        month_means = [float(diff[boot.labels[boot.codes] == lab].mean()) for lab in boot.labels]
        c["months_won"] = int(sum(v < 0 for v in month_means))
        for metric, cut in (("qs", "all rows"), ("pin_q01", "all rows"), ("pin_q99", "all rows"), ("qs", "cold winter nights"), ("qs", "sunny daylight")):  # fmt: skip
            c["noninferiority"][f"{metric} on {cut}"] = boot.mean(values=m[treat][metric] - m["default"][metric], mask=masks[cut])  # fmt: skip
        c["noninferiority"]["tw95 relative"] = boot.relative(treat=m[treat]["tw95"], base=m["default"]["tw95"], mask=every)  # fmt: skip
        c["by_level"] = {name: float((m[treat][f"lvl_{name}"] - m["default"][f"lvl_{name}"]).mean()) for name in common.QUANTILES}  # fmt: skip
        c["leave_one_site_out"] = {s: {"k2": boot.mean(values=diff, mask=site != s, level=K_LEVEL), "ci99": boot.mean(values=diff, mask=site != s, level=0.99)} for s in common.SCORED_SITES}  # fmt: skip
        c["site_alone"] = {s: boot.mean(values=diff, mask=site == s) for s in common.SCORED_SITES}
        for ref in base.BENCHMARKS:
            c[f"minus_{ref}"] = boot.mean(values=m[treat]["qs"] - m[ref]["qs"], mask=every)
        contrasts[treat] = c
    contrasts["STD_minus_STDA"] = boot.mean(
        values=m["STD"]["qs"] - m["STDA"]["qs"], mask=every, level=0.95
    )
    contrasts["STD_minus_G"] = boot.mean(
        values=m["STD"]["qs"] - m["G"]["qs"], mask=every, level=0.95
    )
    rep["contrasts"] = contrasts
    mean_d = rows["mean_d"].to_numpy() if "mean_d" in rows.columns else rows["mean"].to_numpy()
    rep["checks"] = {
        "median_minus_mean_forecast_per_site": {
            a: {
                s: float((arms[a][site == s, 5].astype(np.float64) - mean_d[site == s]).mean())
                for s in common.SCORED_SITES
            }
            for a in ("default", "STD", "STDA")
        },
        "mean_forecast_mae": {
            a: float(np.abs(arms[a].astype(np.float64).mean(axis=1) - actual).mean())
            for a in ("default", "G", "STD", "STDA")
        },
        "median_mae": {a: float(m[a]["mae"].mean()) for a in ("default", "G", "STD", "STDA")},
    }
    rep["replicate"] = replicate(th=th)
    own.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=own.REPORT_DIR / "report_scored.json", content=rep)
    print("gate", rep["gate"])
    print({a: round(rep["by_mask"][a]["all rows"]["qs"], 4) for a in arms})
    for treat in CONTRASTS:
        print(treat, contrasts[treat]["gain_k2"], "sunny", contrasts[treat]["noninferiority"]["qs on sunny daylight"], "lso", {s: round(v["k2"]["est"], 3) for s, v in contrasts[treat]["leave_one_site_out"].items()})  # fmt: skip
    print("STD-STDA", contrasts["STD_minus_STDA"], "STD-G", contrasts["STD_minus_G"], "replicate", rep["replicate"])  # fmt: skip


def replicate(*, th: dict[str, float]) -> dict:
    """Score the even months of 2024-25 for STD and STD-A against experiment 169's arm A bands."""
    pieces = [
        pl.read_parquet(common.E169_PREDS / f"a_replicate_{s}.parquet").with_columns(site=pl.lit(s))
        for s in common.SCORED_SITES
    ]
    frame = pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    frame = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")).filter(pl.col("month").is_in(list(common.REPLICATE_CUTS)))  # fmt: skip
    std = read_fits(head="std", cuts=common.REPLICATE_CUTS)
    stda = read_fits(head="stda", cuts=common.REPLICATE_CUTS).rename({"q50": "stda_q50"})
    joined = frame.join(std, on=KEYS, how="left", validate="1:1", suffix="_std").join(stda, on=KEYS, how="left", validate="1:1")  # fmt: skip
    default_q = joined.select(list(common.QUANTILES)).to_numpy()
    std_q = joined.select([f"{q}_std" for q in common.QUANTILES]).to_numpy()
    assert not np.isnan(std_q).any(), "a replicate row has no STD forecast"
    assert joined["stda_q50"].null_count() == 0, "a replicate row has no STD-A forecast"
    arms = {"default": default_q.astype(np.float32), **build_arms(default_q=default_q, std_q=std_q, stda_median=joined["stda_q50"].to_numpy())}  # fmt: skip
    actual = joined["actual"].to_numpy().astype(np.float64)
    edge = np.array([th[s] for s in joined["site"].to_numpy()], dtype=np.float64)
    m = {a: base.per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = base.Blocks(blocks=joined["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy(), seed=own.EXPERIMENT_ID + 1)  # fmt: skip
    every = np.ones(joined.height, dtype=bool)
    out: dict = {"n_rows": joined.height, "absolute": {a: float(m[a]["qs"].mean()) for a in arms}}
    for treat in ("STD", "STDA"):
        out[treat] = boot.mean(values=m[treat]["qs"] - m["default"]["qs"], mask=every)
    return out


if __name__ == "__main__":
    main()
