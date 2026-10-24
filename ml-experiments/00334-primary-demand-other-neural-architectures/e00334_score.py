# ruff: noqa: E501, PLR0915
"""Score the four networks against Primary demand research default 3 at 0.5 to 240 hours.

Rows, default, and benchmarks are rebuilt as in experiment 322 (the chain's scored-year rows with a
reading outside the hand-filtered events; the default is experiment 251's arm 1 up to 168 hours and
experiment 322's `d1` above). The seed-averaged quantiles of each network are joined by (site,
forecast origin, valid time). Intervals resample the 12 forecast-origin months (20,000 draws).

    data/ml-experiments/00334-*/run.sh ml-experiments/00334-*/e00334_score.py
"""

import itertools
from typing import Final

import e00311_common as c311  # ty: ignore[unresolved-import]
import e00311_score as base  # ty: ignore[unresolved-import]
import e00317_score as s317  # ty: ignore[unresolved-import]
import e00320_common as c320  # ty: ignore[unresolved-import]
import e00322_common as c322  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

import e00334_common as common  # isort: skip

KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
NEURAL: Final[tuple[str, ...]] = ("m1", "m2", "r", "t")
LABELS: Final[dict[str, str]] = {"m1": "M1", "m2": "M2", "r": "R", "t": "T"}
K_LEVEL: Final[float] = 0.95
GATE_VALUE: Final[float] = 2.727
GATE_TOLERANCE: Final[float] = 0.01
BAND_EDGES: Final[np.ndarray] = np.arange(0.0, 241.0, 24.0)
STD_FIT_DIR: Final = (
    c311.ML_EXPERIMENTS_DATA_DIR
    / "00330-primary-demand-standard-architecture-arm"
    / "work"
    / "fits"
)


def read_fits(
    *, arm: str, cuts: tuple[str, ...], seeds: tuple[int, ...], raw: bool = False
) -> pl.DataFrame:
    """Return the seed-averaged quantiles of one arm per key, asserting every seed is present."""
    names = [f"raw_{q}" for q in common.QUANTILES] if raw else list(common.QUANTILES)
    pieces = [
        pl.read_parquet(
            common.FIT_DIR / f"{arm}_cut{label}_seed{seed}.parquet", columns=[*KEYS, *names]
        )
        for label in cuts
        for seed in seeds
    ]
    out = (
        pl.concat(pieces)
        .group_by(KEYS)
        .agg([pl.col(c).mean() for c in names] + [pl.len().alias("n_seeds")])
    )
    assert (out["n_seeds"] == len(seeds)).all(), f"{arm}: a row is missing a seed"
    return out.drop("n_seeds").rename({f"raw_{q}": q for q in common.QUANTILES} if raw else {})


def assemble_rows() -> pl.DataFrame:
    """Return the scored rows with the default's quantiles (`default_q01`..) and the benchmarks'."""
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    stored = (
        pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet").select(KEYS).with_row_index("i251")
    )
    rows = rows.join(stored, on=KEYS, how="left", validate="1:1")
    long_arms = pl.read_parquet(
        c322.ARMS_DIR / "scored_long_lead.parquet",
        columns=[*KEYS, "kind", *[f"d1_{q}" for q in common.QUANTILES]],
    ).rename({"kind": "kind_long"})
    rows = rows.join(long_arms, on=KEYS, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    long = rows["lead_hours"].to_numpy() > c320.STORED_HORIZON_HOURS
    keep = long | rows["i251"].is_not_null().to_numpy()
    rows = rows.filter(pl.Series(keep))
    long = rows["lead_hours"].to_numpy() > c320.STORED_HORIZON_HOURS
    arm1 = np.load(c320.c319.E251_ARMS / "arm1.npy")
    bg2 = np.load(c320.c319.E251_ARMS / "bg2.npy")
    idx = rows["i251"].fill_null(0).to_numpy()
    long_q = rows.select([f"d1_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    default = np.where(long[:, None], long_q, arm1[idx].astype(np.float64))
    return rows.with_columns(
        *[pl.Series(f"default_{q}", default[:, i]) for i, q in enumerate(common.QUANTILES)],
        *[
            pl.Series(f"bg2_{q}", np.where(long, np.nan, bg2[idx][:, i].astype(np.float64)))
            for i, q in enumerate(common.QUANTILES)
        ],
    )


def quantile_matrix(*, rows: pl.DataFrame, prefix: str) -> np.ndarray:
    """Return the (rows, 11) quantile matrix of a prefix."""
    return rows.select([f"{prefix}_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)


def replicate(*, th: dict[str, float]) -> dict:
    """Score the even months of 2024-25 to 168 hours against experiment 169's arm A bands."""
    pieces = [
        pl.read_parquet(c311.E169_PREDS / f"a_replicate_{s}.parquet").with_columns(site=pl.lit(s))
        for s in common.SITES
    ]
    frame = pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    frame = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")).filter(
        pl.col("month").is_in(list(common.REPLICATE_CUTS))
    )
    ref = frame.select(*KEYS, "actual", *[pl.col(q).alias(f"ref_{q}") for q in common.QUANTILES])
    for arm in NEURAL:
        fits = read_fits(arm=arm, cuts=common.REPLICATE_CUTS, seeds=common.REPLICATE_SEEDS).rename(
            {q: f"{arm}_{q}" for q in common.QUANTILES}
        )
        ref = ref.join(fits, on=KEYS, how="left", validate="1:1")
    covered = np.ones(ref.height, dtype=bool)
    for arm in NEURAL:
        covered &= ref[f"{arm}_q50"].is_not_null().to_numpy()
    ref = ref.filter(pl.Series(covered))
    actual = ref["actual"].to_numpy().astype(np.float64)
    edge = np.array([th[s] for s in ref["site"].to_numpy()], dtype=np.float64)
    arms = {"default": quantile_matrix(rows=ref, prefix="ref")}
    arms.update({LABELS[a]: np.sort(quantile_matrix(rows=ref, prefix=a), axis=1) for a in NEURAL})
    m = {a: base.per_row(actual=actual, q=q, edge=edge) for a, q in arms.items()}
    boot = base.Blocks(
        blocks=ref["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy(),
        seed=common.EXPERIMENT_ID + 1,
    )
    every = np.ones(ref.height, dtype=bool)
    out: dict = {"n_rows": ref.height, "absolute": {a: float(m[a]["qs"].mean()) for a in arms}}
    for treat in LABELS.values():
        out[treat] = boot.mean(values=m[treat]["qs"] - m["default"]["qs"], mask=every)
    return out


def main() -> None:
    """Join the arms, run the gate, score every contrast, and write the report."""
    rows = assemble_rows()
    n_reference = rows.height
    for arm in NEURAL:
        fits = read_fits(arm=arm, cuts=common.SCORED_CUTS, seeds=common.SEEDS).rename(
            {q: f"{arm}_{q}" for q in common.QUANTILES}
        )
        rows = rows.join(fits, on=KEYS, how="left", validate="1:1")
    present = np.ones(rows.height, dtype=bool)
    for arm in NEURAL:
        present &= rows[f"{arm}_q50"].is_not_null().to_numpy()
    coverage = {"reference_rows": n_reference, "rows_with_all_four_arms": int(present.sum())}
    lead_all = rows["lead_hours"].to_numpy()
    coverage["missing_by_band"] = {
        f"{int(lo)}-{int(hi)}": int(((lead_all > lo) & (lead_all <= hi) & ~present).sum())
        for lo, hi in itertools.pairwise(BAND_EDGES)
    }
    rows = rows.filter(pl.Series(present))
    # Experiment 330's STD (0.5 to 168 hours), joined through experiment 311's run column.
    run_map = pl.read_parquet(
        c311.E311_ROWS if hasattr(c311, "E311_ROWS") else common.E311_PREP / "rows.parquet",
        columns=["site", "run", "power_fcst_init_time", "valid_time"],
    )
    std_pieces = [
        pl.read_parquet(
            STD_FIT_DIR / f"std_cut{label}_seed{seed}.parquet",
            columns=["site", "run", "valid_time", *c311.QUANTILES],
        )
        for label in c311.SCORED_CUTS
        for seed in (0, 1, 2, 3, 4, 5)
    ]
    std = (
        pl.concat(std_pieces)
        .group_by(["site", "run", "valid_time"])
        .agg([pl.col(q).mean() for q in c311.QUANTILES])
    )
    std = std.join(run_map, on=["site", "run", "valid_time"], how="inner", validate="1:1").drop(
        "run"
    )
    rows = rows.join(
        std.rename({q: f"std_{q}" for q in c311.QUANTILES}), on=KEYS, how="left", validate="1:1"
    )

    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    site = rows["site"].to_numpy()
    actual = rows["actual"].to_numpy().astype(np.float64)
    th = base.thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    arms: dict[str, np.ndarray] = {"default": quantile_matrix(rows=rows, prefix="default")}
    for arm in NEURAL:
        arms[LABELS[arm]] = np.sort(quantile_matrix(rows=rows, prefix=arm), axis=1)
    arms["STD"] = np.sort(quantile_matrix(rows=rows, prefix="std"), axis=1)
    arms["BG2"] = quantile_matrix(rows=rows, prefix="bg2")
    arms["manual_heuristic"] = quantile_matrix(rows=rows, prefix="mh")
    arms["climatology"] = quantile_matrix(rows=rows, prefix="cl")
    m = {a: base.per_row(actual=actual, q=q, edge=edge) for a, q in arms.items()}
    every = np.ones(rows.height, dtype=bool)
    gate_value = float(m["default"]["qs"].mean())
    gate = {
        "default_0p5_to_240": gate_value,
        "stored": GATE_VALUE,
        "pass": bool(abs(gate_value - GATE_VALUE) <= GATE_TOLERANCE),
    }
    boot = base.Blocks(
        blocks=rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        seed=common.EXPERIMENT_ID,
    )

    sunny = rows["sunny_daylight"].to_numpy()
    cold_night = rows["cold_winter_night"].to_numpy()
    local = rows["valid_time"].dt.convert_time_zone("Europe/London")
    local_hh = local.dt.hour().to_numpy() * 2 + local.dt.minute().to_numpy() // 30
    masks: dict[str, np.ndarray] = {
        "all rows": every,
        "0.5-168 h": ~long,
        "168-240 h": long,
        "cold winter nights": cold_night,
        "sunny daylight": sunny,
        "ordinary rows": ~cold_night & ~sunny,
        "Economy 7 night (00:30-07:30)": (local_hh >= 1) & (local_hh <= 14),
        **{f"site {s}": site == s for s in common.SITES},
        **{
            f"lead {int(lo)}-{int(hi)} h": (lead > lo) & (lead <= hi)
            for lo, hi in itertools.pairwise(BAND_EDGES)
        },
    }
    keys_abs = (
        "qs",
        "mae",
        "tw95",
        "cov_q01",
        "cov_q05",
        "cov_q95",
        "cov_q99",
        "pin_q01",
        "pin_q05",
        "pin_q95",
        "pin_q99",
    )
    rep: dict = {
        "n_rows": rows.height,
        "n_long": int(long.sum()),
        "coverage": coverage,
        "gate": gate,
    }
    rep["by_mask"] = {
        a: {
            cut: {k: float(np.nanmean(m[a][k][mask])) for k in keys_abs}
            for cut, mask in masks.items()
            if mask.any()
        }
        for a in arms
    }
    rep["by_mask_without_d10"] = {
        a: {
            cut: float(np.nanmean(m[a]["qs"][mask & (site != "D10")]))
            for cut, mask in masks.items()
            if (mask & (site != "D10")).any()
        }
        for a in arms
    }
    contrasts: dict = {}
    windows = {"0.5-240 h": every, "0.5-168 h": ~long, "168-240 h": long}
    for treat in (*LABELS.values(), "STD"):
        use_windows = {k: v for k, v in windows.items() if treat != "STD" or k == "0.5-168 h"}
        diff = m[treat]["qs"] - m["default"]["qs"]
        c: dict = {
            "window": {
                w: boot.mean(values=diff, mask=mask, level=K_LEVEL)
                for w, mask in use_windows.items()
            },
            "window_without_d10": {
                w: boot.mean(values=diff, mask=mask & (site != "D10"), level=K_LEVEL)
                for w, mask in use_windows.items()
            },
            "groups": {
                cut: boot.mean(values=diff, mask=mask & ~long if treat == "STD" else mask)
                for cut, mask in masks.items()
                if (mask & ~long if treat == "STD" else mask).any()
            },
            "noninferiority": {},
            "months_won": int(
                sum(float(diff[boot.labels[boot.codes] == lab].mean()) < 0 for lab in boot.labels)
            ),
            "by_level": {
                name: float(
                    np.nanmean(
                        (m[treat][f"lvl_{name}"] - m["default"][f"lvl_{name}"])[
                            ~long if treat == "STD" else every
                        ]
                    )
                )
                for name in common.QUANTILES
            },
            "leave_one_site_out": {
                s: boot.mean(
                    values=diff,
                    mask=(site != s) & (~long if treat == "STD" else every),
                    level=K_LEVEL,
                )
                for s in common.SITES
            },
            "site_alone": {
                s: boot.mean(values=diff, mask=(site == s) & (~long if treat == "STD" else every))
                for s in common.SITES
            },
        }
        base_mask = ~long if treat == "STD" else every
        for metric, cut in (
            ("qs", "all rows"),
            ("pin_q01", "all rows"),
            ("pin_q99", "all rows"),
            ("qs", "cold winter nights"),
            ("qs", "sunny daylight"),
        ):
            c["noninferiority"][f"{metric} on {cut}"] = boot.mean(
                values=m[treat][metric] - m["default"][metric], mask=masks[cut] & base_mask
            )
        c["noninferiority"]["tw95 relative"] = boot.relative(
            treat=m[treat]["tw95"], base=m["default"]["tw95"], mask=base_mask
        )
        for ref in ("manual_heuristic", "climatology"):
            c[f"minus_{ref}"] = boot.mean(values=m[treat]["qs"] - m[ref]["qs"], mask=base_mask)
        contrasts[treat] = c
    pairwise = {}
    for a, b in (("M2", "M1"), ("R", "M1"), ("T", "M1"), ("T", "R")):
        d = m[a]["qs"] - m[b]["qs"]
        pairwise[f"{a}_minus_{b}"] = {
            w: boot.mean(values=d, mask=mask, level=K_LEVEL) for w, mask in windows.items()
        }
    best = min(LABELS.values(), key=lambda a: float(m[a]["qs"][~long].mean()))
    d = m[best]["qs"] - m["STD"]["qs"]
    pairwise["best_neural_minus_STD_0p5_to_168"] = {
        "arm": best,
        **boot.mean(values=d, mask=~long, level=K_LEVEL),
        "relative": float(m[best]["qs"][~long].mean() / m["STD"]["qs"][~long].mean() - 1.0),
    }
    rep["contrasts"], rep["pairwise"] = contrasts, pairwise
    # Seed noise (P4): pooled score of each seed alone.
    seed_scores: dict = {}
    for arm in NEURAL:
        per_seed = []
        for seed in common.SEEDS:
            single = pl.concat(
                [
                    pl.read_parquet(
                        common.FIT_DIR / f"{arm}_cut{label}_seed{seed}.parquet",
                        columns=[*KEYS, *common.QUANTILES],
                    )
                    for label in common.SCORED_CUTS
                ]
            )
            joined = rows.select(KEYS).join(single, on=KEYS, how="left", validate="1:1")
            per_seed.append(
                float(
                    base.per_row(
                        actual=actual,
                        q=np.sort(
                            joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64),
                            axis=1,
                        ),
                        edge=edge,
                    )["qs"].mean()
                )
            )
        seed_scores[LABELS[arm]] = {
            "single_seed_scores": per_seed,
            "sd": float(np.std(per_seed, ddof=1)),
            "seed_average": float(m[LABELS[arm]]["qs"].mean()),
        }
    rep["seed_noise"] = seed_scores
    # Raw (uncalibrated) quantiles: what the as-of calibration of experiment 164 changed.
    raw_scores = {}
    for arm in NEURAL:
        raw = read_fits(arm=arm, cuts=common.SCORED_CUTS, seeds=common.SEEDS, raw=True)
        joined = rows.select(KEYS).join(raw, on=KEYS, how="left", validate="1:1")
        q = np.sort(joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64), axis=1)
        pr = base.per_row(actual=actual, q=q, edge=edge)
        raw_scores[LABELS[arm]] = {
            "qs": float(pr["qs"].mean()),
            **{k: float(pr[k][every].mean()) for k in ("cov_q01", "cov_q05", "cov_q95", "cov_q99")},
            "qs_cold_winter_nights": float(pr["qs"][cold_night].mean()),
        }
    rep["raw_uncalibrated"] = raw_scores
    mean_proxy = {a: arms[a].mean(axis=1) for a in ("default", *LABELS.values())}
    rep["checks"] = {
        "median_minus_mean_of_quantiles_per_site": {
            a: {
                s: float(
                    (arms[a][site == s, common.MEDIAN_INDEX] - mean_proxy[a][site == s]).mean()
                )
                for s in common.SITES
            }
            for a in mean_proxy
        },
        "mean_of_quantiles_mae": {
            a: float(np.abs(mean_proxy[a] - actual).mean()) for a in mean_proxy
        },
        "median_mae": {a: float(m[a]["mae"].mean()) for a in ("default", *LABELS.values())},
    }
    rep["gain_by_hour"] = {
        a: [float(np.mean((m[a]["qs"] - m["default"]["qs"])[local_hh == h])) for h in range(48)]
        for a in LABELS.values()
    }
    rep["replicate"] = replicate(th=th)
    common.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=common.REPORT_DIR / "report_scored.json", content=rep)
    np.savez(
        common.REPORT_DIR / "scored_arrays.npz",
        lead=lead,
        site=site,
        valid_time=rows["valid_time"].dt.epoch("us").to_numpy(),
        **{f"qs_{a}": m[a]["qs"] for a in arms},
    )
    print("gate", gate, "coverage", coverage)
    print({a: round(float(m[a]["qs"].mean()), 4) for a in arms}, "n", rows.height)
    for t, c in contrasts.items():
        print(
            t,
            c["window"],
            "without D10",
            {w: round(v["est"], 3) for w, v in c["window_without_d10"].items()},
        )


if __name__ == "__main__":
    main()
