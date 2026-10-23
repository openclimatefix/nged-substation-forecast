# ruff: noqa: E501, PLR0915
"""Build the arms of experiment 314 and score them with experiment 311's machinery.

Arms: R (Primary demand research default 1), M (experiment 311's stored mixture head, for
reference), G (M with the default's bands kept on sunny daylight half-hours), P (the pinball
quantile network around the default's median), GP (P gated the same way). The gate flag is
`sunny & daylight` from the stored rows, experiment 311's protected group. The three pre-registered
contrasts (G, P, GP minus R) are read at 99.667% (k = 3). The replicate year has no sunny flag, so
only the ungated arms M and P can be scored there.

    export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*
    uv run --no-sync python ml-experiments/00314-*/e00314_score.py
"""

from typing import Final

import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00311_score as base  # isort: skip  # ty: ignore[unresolved-import]
import e00314_common as own  # isort: skip
import numpy as np
import polars as pl

CONTRASTS: Final[tuple[str, ...]] = ("G", "P", "GP")
K_LEVEL: Final[float] = 1.0 - 0.01 / 3.0


def read_fits(*, cuts: tuple[str, ...]) -> pl.DataFrame:
    """Return the seed-averaged pinball-head quantiles per (site, run, valid_time)."""
    pieces = [
        pl.read_parquet(
            own.FIT_DIR / f"cut{label}_seed{seed}.parquet",
            columns=["site", "run", "valid_time", *common.QUANTILES],
        )
        for label in cuts
        for seed in common.SEEDS
    ]
    out = (
        pl.concat(pieces)
        .group_by(["site", "run", "valid_time"])
        .agg([pl.col(q).mean() for q in common.QUANTILES] + [pl.len().alias("n_seeds")])
    )
    assert (out["n_seeds"] == len(common.SEEDS)).all(), "a row is missing a seed"
    return out.drop("n_seeds")


def gate(*, protected: np.ndarray, kept: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Return `kept` on protected rows and `other` elsewhere."""
    return np.where(protected[:, None], kept, other).astype(np.float32)


def main() -> None:
    """Score the scored year and the replicate year and write the report."""
    rows = pl.read_parquet(base.SCORED_START / "rows.parquet")
    keys = ["site", "run", "valid_time"]
    joined = rows.select("row", *keys).join(
        read_fits(cuts=common.SCORED_CUTS), on=keys, how="left", validate="1:1"
    )
    assert joined["q50"].null_count() == 0, "a scored row has no head forecast"
    default_q = np.load(common.E251_SCORED / "arm1.npy")
    arm_m = np.load(own.E311_DATA_DIR / "arms" / "scored" / "M.npy")
    arm_p = base.centred_arms(
        default_q=default_q,
        head_q=joined.select(list(common.QUANTILES)).to_numpy(),
        ens_mean_c=rows["ens_mean_c"].to_numpy(),
    )["M"]
    daylight = rows["daylight"].to_numpy()
    sunny = rows["sunny"].to_numpy() & daylight
    cold_night = rows["cold2"].to_numpy() & ~daylight
    arms = {
        "default": default_q.astype(np.float32),
        "M": arm_m,
        "G": gate(protected=sunny, kept=default_q, other=arm_m),
        "P": arm_p,
        "GP": gate(protected=sunny, kept=default_q, other=arm_p),
    }
    for name in base.BENCHMARKS:
        arms[name] = np.load(common.E251_SCORED / f"{name}.npy")
    out_dir = own.WORK_DIR / "arms" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, q in arms.items():
        np.save(out_dir / f"{name}.npy", q)

    site = rows["site"].to_numpy()
    th = base.thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64)
    m = {a: base.per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = base.Blocks(
        blocks=rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        seed=own.EXPERIMENT_ID,
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
    }  # fmt: skip
    keys_abs = ("qs", "mae", "tw95", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q01", "pin_q05", "pin_q95", "pin_q99")  # fmt: skip
    rep: dict = {"n_rows": rows.height, "n_sunny_daylight_rows": int(sunny.sum())}
    rep["by_mask"] = {a: {cut: {k: float(m[a][k][mask].mean()) for k in keys_abs} for cut, mask in masks.items() if mask.any()} for a in arms}  # fmt: skip
    five = np.isin(site, ["D5", "D10", "D11", "D14", "D16"])
    rep["gate"] = {"default_six": float(m["default"]["qs"].mean()), "default_five": float(m["default"]["qs"][five].mean())}  # fmt: skip
    rep["gate"]["pass"] = bool(abs(rep["gate"]["default_six"] - base.GATE_SIX) <= base.GATE_TOLERANCE and abs(rep["gate"]["default_five"] - base.GATE_FIVE) <= base.GATE_TOLERANCE)  # fmt: skip
    contrasts: dict = {}
    for treat in ("M", *CONTRASTS):
        diff = m[treat]["qs"] - m["default"]["qs"]
        c: dict = {
            "gain_k3": boot.mean(values=diff, mask=every, level=K_LEVEL),
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
        c["leave_one_site_out"] = {s: {"k3": boot.mean(values=diff, mask=site != s, level=K_LEVEL), "ci99": boot.mean(values=diff, mask=site != s, level=0.99)} for s in common.SCORED_SITES}  # fmt: skip
        c["site_alone"] = {s: boot.mean(values=diff, mask=site == s) for s in common.SCORED_SITES}
        for ref in base.BENCHMARKS:
            c[f"minus_{ref}"] = boot.mean(values=m[treat]["qs"] - m[ref]["qs"], mask=every)
        contrasts[treat] = c
    m_gain = float((m["M"]["qs"] - m["default"]["qs"]).mean())
    contrasts["G_share_of_M_gain"] = float((m["G"]["qs"] - m["default"]["qs"]).mean()) / m_gain
    contrasts["P_minus_M"] = boot.mean(values=m["P"]["qs"] - m["M"]["qs"], mask=every, level=0.95)
    contrasts["GP_minus_G"] = boot.mean(values=m["GP"]["qs"] - m["G"]["qs"], mask=every, level=0.95)
    rep["contrasts"] = contrasts
    mean_d = rows["mean_d"].to_numpy() if "mean_d" in rows.columns else rows["mean"].to_numpy()
    rep["checks"] = {
        "median_equals_default": {
            a: bool(np.array_equal(arms[a][:, 5], arms["default"][:, 5]))
            for a in ("M", "G", "P", "GP")
        },
        "G_equals_default_on_sunny": bool(np.array_equal(arms["G"][sunny], arms["default"][sunny])),
        "G_equals_M_elsewhere": bool(np.array_equal(arms["G"][~sunny], arms["M"][~sunny])),
        "GP_equals_P_elsewhere": bool(np.array_equal(arms["GP"][~sunny], arms["P"][~sunny])),
        "median_minus_mean_per_site": {
            s: float((arms["default"][site == s, 5].astype(np.float64) - mean_d[site == s]).mean())
            for s in common.SCORED_SITES
        },
        "mean_forecast_mean_minus_median_P": float(
            (arms["P"].astype(np.float64).mean(axis=1) - arms["P"][:, 5]).mean()
        ),
        "mean_forecast_mean_minus_median_M": float(
            (arms["M"].astype(np.float64).mean(axis=1) - arms["M"][:, 5]).mean()
        ),
        "mean_forecast_mean_minus_median_default": float(
            (arms["default"].astype(np.float64).mean(axis=1) - arms["default"][:, 5]).mean()
        ),
    }
    rep["replicate"] = replicate(th=th)
    own.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=own.REPORT_DIR / "report_scored.json", content=rep)
    print("gate", rep["gate"], "checks", {k: v for k, v in rep["checks"].items() if "per_site" not in k})  # fmt: skip
    print({a: round(rep["by_mask"][a]["all rows"]["qs"], 4) for a in arms})
    for treat in ("M", *CONTRASTS):
        print(treat, contrasts[treat]["gain_k3"], "sunny", contrasts[treat]["noninferiority"]["qs on sunny daylight"], "lso", {s: round(v["k3"]["est"], 3) for s, v in contrasts[treat]["leave_one_site_out"].items()})  # fmt: skip
    print("G share", contrasts["G_share_of_M_gain"], "P-M", contrasts["P_minus_M"], "replicate", rep["replicate"])  # fmt: skip


def replicate(*, th: dict[str, float]) -> dict:
    """Score the even months of 2024-25 for the ungated arms M and P against arm A's bands."""
    pieces = [
        pl.read_parquet(common.E169_PREDS / f"a_replicate_{s}.parquet").with_columns(site=pl.lit(s))
        for s in common.SCORED_SITES
    ]
    frame = pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    frame = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")).filter(pl.col("month").is_in(list(common.REPLICATE_CUTS)))  # fmt: skip
    keys = ["site", "run", "valid_time"]
    m_fits = base.read_fits(cuts=common.REPLICATE_CUTS)
    p_fits = read_fits(cuts=common.REPLICATE_CUTS)
    joined = frame.join(m_fits, on=keys, how="left", validate="1:1", suffix="_m").join(p_fits, on=keys, how="left", validate="1:1", suffix="_p")  # fmt: skip
    default_q = joined.select(list(common.QUANTILES)).to_numpy()
    ens = joined["ens_mean_c"].to_numpy()
    arms = {"default": default_q.astype(np.float32)}
    for tag, suffix in (("M", "_m"), ("P", "_p")):
        head = joined.select([f"{q}{suffix}" for q in common.QUANTILES]).to_numpy()
        assert not np.isnan(head).any(), "a replicate row has no head forecast"
        arms[tag] = base.centred_arms(default_q=default_q, head_q=head, ens_mean_c=ens)["M"]
    actual = joined["actual"].to_numpy().astype(np.float64)
    edge = np.array([th[s] for s in joined["site"].to_numpy()], dtype=np.float64)
    m = {a: base.per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = base.Blocks(blocks=joined["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy(), seed=own.EXPERIMENT_ID + 1)  # fmt: skip
    every = np.ones(joined.height, dtype=bool)
    out: dict = {"n_rows": joined.height, "absolute": {a: float(m[a]["qs"].mean()) for a in arms}}
    for treat in ("M", "P"):
        out[treat] = boot.mean(values=m[treat]["qs"] - m["default"]["qs"], mask=every)
    return out


if __name__ == "__main__":
    main()
