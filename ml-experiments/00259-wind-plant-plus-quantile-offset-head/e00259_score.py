"""Score experiment 00259's arms and read the planned contrasts.

Reads the base arms from `work/base/` (R0, R1), the heads from `work/heads/`, and experiment 230's
stored arms (A0r, B1, CLIM) as reference rows. Scores every arm on the quantile score of experiment
33 (nine levels, points of capacity), the pinball loss at the named tail levels, the tail score T
(the mean of the 0.01 and 0.99 pinball losses), coverage of the 5% to 95% and 1% to 99% bands, and
the error of the median and mean forecasts, by lead-day band and row set, and bootstraps
contrasts by resampling whole months (of the ENS run) and the five seeds. The farms are not
resampled.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00259-*/e00259_score.py
"""

import json
import os
from collections.abc import Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00259_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BASE_DIR = common.WORK_DIR / "base"
HEAD_DIR = common.WORK_DIR / "heads"
REFERENCE_ARM_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "arms"
)
"""Experiment 230's stored arms, read only."""
REFERENCE_ARMS = {"A0r": "A0r", "B1-230": "B1", "CLIM": "CLIM"}
SEEDS = (0, 1, 2, 3, 4)
RESAMPLES = 50_000
BONFERRONI_LEVEL = 1 - 0.05 / 4
LEVEL_95 = 0.95
NON_NEURAL = ("N5-lin", "N5-xgb")
HEAD_ARMS = ("H5c", "K5", "BL5", "H5c-Tue", "N5-lin", "N5-xgb", "T5")
CONTRAST_METRICS = ("qs", "tail_t", "pinball_q99", "mean_abs_error", "median_abs_error")


def read_arm(*, name: str, cases: pl.DataFrame) -> pl.DataFrame | None:
    """Return one arm's quantiles joined to `cases` with seeds 0 to 4, or None if absent."""
    if name in REFERENCE_ARMS:
        path = REFERENCE_ARM_DIR / f"{REFERENCE_ARMS[name]}_spread.parquet"
        frame = pl.read_parquet(path) if path.exists() else None
    elif name == "R0":
        paths = [BASE_DIR / f"R0_scored_seed{s}.parquet" for s in SEEDS]
        frame = (
            pl.concat([pl.read_parquet(p) for p in paths])
            if all(p.exists() for p in paths)
            else None
        )
    elif name == "R1":
        path = BASE_DIR / "R1_scored_seed0.parquet"
        frame = pl.read_parquet(path) if path.exists() else None
    else:
        path = HEAD_DIR / f"{name}.parquet"
        frame = pl.read_parquet(path) if path.exists() else None
    if frame is None:
        return None
    keep = [*common.KEY_COLUMNS, "seed", *common.LEVEL_COLUMNS]
    frame = frame.select(keep)
    if frame["seed"].n_unique() == 1 and name not in REFERENCE_ARMS:
        frame = pl.concat([frame.with_columns(seed=pl.lit(s, dtype=pl.Int32)) for s in SEEDS])
    joined = cases.join(frame, on=list(common.KEY_COLUMNS), how="inner", validate="1:m")
    return joined.sort("seed", *common.KEY_COLUMNS)


def case_scores(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row scores and errors of one arm (points of capacity for the scores)."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    scale = common.POINTS_PER_FRACTION
    mean_forecast = quantiles[:, 1:10] @ common.NINE_MASSES
    out = {
        "qs": common.quantile_score(quantiles=quantiles, actual=actual) * scale,
        "covered": common.covered(quantiles=quantiles, actual=actual).astype(np.float64),
        "covered_1_99": ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 10])).astype(
            np.float64
        ),
        "median_abs_error": np.abs(actual - quantiles[:, common.MEDIAN_INDEX]) * scale,
        "median_minus_mean": (quantiles[:, common.MEDIAN_INDEX] - mean_forecast) * scale,
        "mean_abs_error": np.abs(actual - mean_forecast) * scale,
    }
    for name in common.NAMED_LEVELS:
        out[f"pinball_{name}"] = (
            common.pinball_at(quantiles=quantiles, actual=actual, column=name) * scale
        )
    out["tail_t"] = (out["pinball_q01"] + out["pinball_q99"]) / 2.0
    return out


def bootstrap(
    *, difference: np.ndarray, month: np.ndarray, seed: np.ndarray, levels: tuple[float, ...]
) -> list[float]:
    """Return the mean difference and its interval at each of `levels`, resampling months, seeds."""
    months, seeds = np.unique(month), np.unique(seed)
    sums = np.zeros((months.size, seeds.size))
    counts = np.zeros_like(sums)
    for i, m in enumerate(months):
        for j, s in enumerate(seeds):
            keep = (month == m) & (seed == s)
            sums[i, j], counts[i, j] = difference[keep].sum(), keep.sum()
    rng = np.random.default_rng(0)
    draw_m = rng.integers(0, months.size, size=(RESAMPLES, months.size))
    draw_s = rng.integers(0, seeds.size, size=(RESAMPLES, seeds.size))
    picked = (draw_m[:, :, None], draw_s[:, None, :])
    means = sums[picked].sum(axis=(1, 2)) / counts[picked].sum(axis=(1, 2))
    out = [float(sums.sum() / counts.sum())]
    for level in levels:
        tail = (1 - level) / 2
        out += [float(x) for x in np.quantile(means, [tail, 1 - tail])]
    return out


def better_non_neural() -> str:
    """Return the non-neural arm with the lower mean loss on the early-stopping runs."""
    losses = json.loads((HEAD_DIR / "validation_loss.json").read_text())
    return min(NON_NEURAL, key=lambda arm: float(np.mean(list(losses[arm].values()))))


def hindcast_quarters() -> dict[str, float]:
    """Return R1's mean quantile score by quarter of valid time on the kept hindcast cases."""
    summaries = pl.read_parquet(BASE_DIR / "summaries_hindcast.parquet")
    base = pl.read_parquet(BASE_DIR / "R1_hindcast_seed0.parquet").drop("seed")
    frame = summaries.join(base, on=list(common.KEY_COLUMNS), how="inner", validate="1:1").filter(
        ~pl.col("masked")
    )
    score = (
        common.quantile_score(
            quantiles=frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64),
            actual=frame["actual_norm"].to_numpy().astype(np.float64),
        )
        * common.POINTS_PER_FRACTION
    )
    quarter = frame["valid_time"].dt.to_string("%Y-Q") + (
        (frame["valid_time"].dt.month() - 1) // 3 + 1
    ).cast(pl.String)
    return {
        str(q): float(score[(quarter == q).to_numpy()].mean()) for q in sorted(quarter.unique())
    }


def main() -> None:  # noqa: PLR0912
    """Score every arm present and write the report."""
    sites = common.site_table()
    cases = common.cases_table(sites=sites).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        quarter=pl.col("valid_time").dt.to_string("%Y-Q")
        + ((pl.col("valid_time").dt.month() - 1) // 3 + 1).cast(pl.String),
        spring=pl.col("valid_time").dt.month().is_in([4, 5, 6]),
    )
    arms = {}
    for name in (*REFERENCE_ARMS, "R0", "R1", *HEAD_ARMS):
        frame = read_arm(name=name, cases=cases)
        if frame is not None:
            arms[name] = (frame, case_scores(frame=frame))
    row_sets: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
        "all rows": lambda f: np.ones(f.height, dtype=bool),
        "kept rows": lambda f: ~f["masked"].to_numpy(),
        "kept April-June": lambda f: ~f["masked"].to_numpy() & f["spring"].to_numpy(),
    }
    band_masks: dict[str, Callable[..., np.ndarray]] = {
        name: lambda f, lo=lo, hi=hi: (
            (f["lead_day"].to_numpy() >= lo) & (f["lead_day"].to_numpy() <= hi)
        )
        for name, lo, hi in common.LEAD_DAY_BANDS
    }
    band_masks["hours 12-36"] = lambda f: (
        (f["lead_hours"].to_numpy() >= 12.0) & (f["lead_hours"].to_numpy() <= 36.0)
    )
    report: dict = {"scores": {}, "contrasts": {}, "per_farm": {}, "quarters": {}, "checks": {}}
    for name, (frame, scores) in arms.items():
        report["scores"][name] = {}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame) & bm(frame)
                report["scores"][name][f"{rs_name} | {band}"] = {
                    metric: float(values[keep].mean()) if keep.any() else None
                    for metric, values in scores.items()
                } | {"rows": int(keep.sum())}
        report["per_farm"][name] = {
            farm: {
                "qs_all": float(scores["qs"][(frame["site"] == farm).to_numpy()].mean()),
                "median_minus_mean": float(
                    scores["median_minus_mean"][(frame["site"] == farm).to_numpy()].mean()
                ),
            }
            for farm in sites["site"].to_list()
        }
        report["quarters"][name] = {
            str(q): float(
                scores["qs"][((frame["quarter"] == q) & ~frame["masked"]).to_numpy()].mean()
            )
            for q in sorted(frame["quarter"].unique())
        }
    report["quarters"]["R1 on the hindcast"] = hindcast_quarters()
    chosen = (
        better_non_neural()
        if all((HEAD_DIR / f"{a}.parquet").exists() for a in NON_NEURAL)
        else None
    )
    report["checks"]["better_non_neural"] = chosen
    contrasts = [
        ("C1", "H5c", "R1", True),
        ("C2", "H5c", "R0", True),
        ("C3", "H5c", chosen, True),
        ("C4", "H5c", "K5", True),
        ("C5", "H5c", "H5c-Tue", False),
        ("D1", "H5c", "BL5", False),
        ("D2", "BL5", "R1", False),
        ("D3", "H5c", "T5", False),
        ("D4", "R0", "R1", False),
        ("D5", "H5c", "N5-lin", False),
        ("D6", "H5c", "N5-xgb", False),
    ]
    for label, first, second, decisive in contrasts:
        if second is None or first not in arms or second not in arms:
            continue
        (frame_a, sa), (frame_b, sb) = arms[first], arms[second]
        keys = [*common.KEY_COLUMNS, "seed"]
        if not frame_a.select(keys).equals(frame_b.select(keys)):
            msg = f"{first} and {second} do not cover the same rows"
            raise ValueError(msg)
        report["contrasts"][label] = {"first": first, "second": second}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame_a) & bm(frame_a)
                levels = (LEVEL_95, BONFERRONI_LEVEL) if decisive else (LEVEL_95,)
                for metric in CONTRAST_METRICS:
                    report["contrasts"][label][f"{rs_name} | {band} | {metric}"] = bootstrap(
                        difference=(sa[metric] - sb[metric])[keep],
                        month=frame_a["month"].to_numpy()[keep],
                        seed=frame_a["seed"].to_numpy()[keep],
                        levels=levels,
                    )
    report["leave_one_farm_out"] = {}
    for label in ("C1", "C2"):
        if label not in report["contrasts"]:
            continue
        first, second = report["contrasts"][label]["first"], report["contrasts"][label]["second"]
        (frame_a, sa), (_, sb) = arms[first], arms[second]
        for farm in sites["site"].to_list():
            keep = ~frame_a["masked"].to_numpy() & (frame_a["site"] != farm).to_numpy()
            report["leave_one_farm_out"][f"{label} without {farm}"] = bootstrap(
                difference=(sa["qs"] - sb["qs"])[keep],
                month=frame_a["month"].to_numpy()[keep],
                seed=frame_a["seed"].to_numpy()[keep],
                levels=(LEVEL_95,),
            )
    if {"H5c", "R1", "BL5"} <= arms.keys():
        for rs_name in ("kept rows", "kept April-June"):
            key = f"{rs_name} | all leads"
            gain = report["scores"]["H5c"][key]["qs"] - report["scores"]["R1"][key]["qs"]
            blind = report["scores"]["BL5"][key]["qs"] - report["scores"]["R1"][key]["qs"]
            report["checks"][f"blind_share {rs_name}"] = blind / gain if gain < 0 else None
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        print(
            name,
            {
                k: round(v["qs"], 3)
                for k, v in report["scores"][name].items()
                if k in ("kept rows | all leads", "all rows | all leads")
            },
        )
    for label, content in report["contrasts"].items():
        print(
            label,
            content["first"],
            "-",
            content["second"],
            "kept | all leads | qs:",
            [round(x, 3) for x in content.get("kept rows | all leads | qs", [])],
            "| tail_t:",
            [round(x, 3) for x in content.get("kept rows | all leads | tail_t", [])],
        )
    print(report["checks"])


if __name__ == "__main__":
    main()
