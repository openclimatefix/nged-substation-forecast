"""Score experiment 00256's arms and read the planned contrasts.

Reads the stored best guess (A0, nine levels) and `work/arms/<arm>_spread.parquet`, scores every arm
on the quantile score of experiment 33 (nine levels, points of capacity), the pinball loss at the
named tail levels, coverage of the 5% to 95% band, and the error of the median and mean forecasts,
by lead-day band and row set (all rows, rows kept by experiment 210's mask, and the kept night,
day, winter, and summer rows), and bootstraps contrasts by resampling whole months (of the ENS run)
and the three seeds. The farms are not resampled.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00256-*/e00256_score.py
"""

import json
import os
from collections.abc import Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00256_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from studies import solar  # noqa: E402

ARM_DIR = common.WORK_DIR / "arms"
REFERENCE_ARM_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "arms"
)
"""Experiment 230's stored arms (A0r, A0rT, CLIM), read only."""
REFERENCE_ARMS = ("A0", "A0r", "A0rT", "CLIM")
RESAMPLES = 50_000
CONTRASTS = (
    ("C1", "T1", "T0"),
    ("C2", "T2", "T1"),
    ("C3", "T3", "T1"),
    ("C4", "P1", "T0"),
    ("D1", "T0", "A0r"),
    ("D2", "T1", "A0rT"),
    ("D3", "T2", "T0"),
    ("D4", "T3", "T0"),
    ("D5", "T1", "P1"),
    ("D6", "T2p", "T1"),
    ("D7", "T3p", "T1"),
    ("D8", "T4", "T1"),
    ("D9", "T5", "T1"),
    ("D10", "T6", "T1"),
    ("D11", "T3", "T6"),
)
DECISIVE: tuple[str, ...] = ("C1", "C2", "C3", "C4")
BONFERRONI_LEVEL = 1 - 0.05 / 4
SCORE_BANDS = ("days 1-2", "days 3-6", "days 7+", "all leads")


def read_arm(*, name: str, cases: pl.DataFrame) -> pl.DataFrame | None:
    """Return one arm's quantiles joined to `cases`, or None if the arm has not been written."""
    if name == "A0":
        frame = pl.read_parquet(common.BEST_GUESS_PATH).filter(pl.col("technology") == "wind")
        nine = [c for c in common.LEVEL_COLUMNS if c in frame.columns]
        frame = frame.select(*common.KEY_COLUMNS, "seed", *nine)
        for missing in set(common.LEVEL_COLUMNS) - set(nine):
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float32).alias(missing))
        frame = frame.select(*common.KEY_COLUMNS, "seed", *common.LEVEL_COLUMNS)
    else:
        base = REFERENCE_ARM_DIR if name in REFERENCE_ARMS else ARM_DIR
        path = base / f"{name}_spread.parquet"
        if not path.exists():
            return None
        frame = pl.read_parquet(path)
    joined = cases.join(frame, on=list(common.KEY_COLUMNS), how="inner", validate="1:m")
    return joined.sort("seed", *common.KEY_COLUMNS)


def case_scores(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row scores and errors of one arm (points of capacity for the scores)."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    scale = common.POINTS_PER_FRACTION
    out = {
        "qs": common.quantile_score(quantiles=quantiles, actual=actual) * scale,
        "covered": common.covered(quantiles=quantiles, actual=actual).astype(np.float64),
        "median_abs_error": np.abs(actual - quantiles[:, common.MEDIAN_INDEX]) * scale,
        "median_minus_mean": (
            quantiles[:, common.MEDIAN_INDEX] - quantiles[:, 1:10] @ common.NINE_MASSES
        )
        * scale,
    }
    mean_forecast = quantiles[:, 1:10] @ common.NINE_MASSES
    out["mean_abs_error"] = np.abs(actual - mean_forecast) * scale
    if not np.isnan(quantiles[:, 0]).any():
        for name in common.NAMED_LEVELS:
            out[f"pinball_{name}"] = (
                common.pinball_at(quantiles=quantiles, actual=actual, column=name) * scale
            )
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


def main() -> None:  # noqa: PLR0912
    """Score every arm present and write the report."""
    sites = common.site_table()
    cases = common.cases_table(sites=sites)
    pieces = []
    for row in sites.iter_rows(named=True):
        times = (
            cases.filter(pl.col("time_series_id") == row["time_series_id"])
            .select("time_series_id", "valid_time")
            .unique()
            .sort("valid_time")
        )
        zenith = solar.zenith(
            stamps=times["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(times.with_columns(elevation_deg=pl.Series(90.0 - zenith)))
    elevation = pl.concat(pieces)
    cases = cases.join(
        elevation, on=["time_series_id", "valid_time"], how="left", validate="m:1"
    ).with_columns(
        night=pl.col("elevation_deg") < 0,
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        season=pl.col("valid_time")
        .dt.month()
        .replace_strict(
            {12: "winter", 1: "winter", 2: "winter", 6: "summer", 7: "summer", 8: "summer"},
            default="other",
        ),
    )
    arms = {}
    for name in (*REFERENCE_ARMS, *common.ARM_FEATURES):
        frame = read_arm(name=name, cases=cases)
        if frame is not None:
            arms[name] = (frame, case_scores(frame=frame))
    row_sets = {
        "all rows": lambda f: np.ones(f.height, dtype=bool),
        "kept rows": lambda f: ~f["masked"].to_numpy(),
        "kept night": lambda f: ~f["masked"].to_numpy() & f["night"].to_numpy(),
        "kept day": lambda f: ~f["masked"].to_numpy() & ~f["night"].to_numpy(),
        "kept winter": lambda f: ~f["masked"].to_numpy() & (f["season"] == "winter").to_numpy(),
        "kept summer": lambda f: ~f["masked"].to_numpy() & (f["season"] == "summer").to_numpy(),
    }
    band_masks: dict[str, Callable[..., np.ndarray]] = {
        name: lambda f, lo=lo, hi=hi: (
            (f["lead_day"].to_numpy() >= lo) & (f["lead_day"].to_numpy() <= hi)
        )
        for name, lo, hi in common.LEAD_DAY_BANDS
    }
    band_masks["hours 0-6"] = lambda f: f["lead_hours"].to_numpy() <= 6.0
    report: dict = {"scores": {}, "contrasts": {}, "per_farm": {}, "checks": {}}
    for name, (frame, scores) in arms.items():
        report["scores"][name] = {}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame) & bm(frame)
                report["scores"][name][f"{rs_name} | {band}"] = {
                    metric: float(values[keep].mean()) if keep.any() else None
                    for metric, values in scores.items()
                } | {"rows": int(keep.sum())}
        by_farm = {}
        for farm in sites["site"].to_list():
            keep = (frame["site"] == farm).to_numpy()
            by_farm[farm] = {
                "qs_all": float(scores["qs"][keep].mean()),
                "median_minus_mean": float(scores["median_minus_mean"][keep].mean()),
            }
        report["per_farm"][name] = by_farm
    for label, first, second in CONTRASTS:
        if first not in arms or second not in arms:
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
                levels = (0.95, BONFERRONI_LEVEL) if label in DECISIVE else (0.95,)
                for metric in ("qs", "median_abs_error", "mean_abs_error"):
                    report["contrasts"][label][f"{rs_name} | {band} | {metric}"] = bootstrap(
                        difference=(sa[metric] - sb[metric])[keep],
                        month=frame_a["month"].to_numpy()[keep],
                        seed=frame_a["seed"].to_numpy()[keep],
                        levels=levels,
                    )
    report["leave_one_farm_out"] = {}
    for label, first, second in CONTRASTS:
        if label not in report["contrasts"]:
            continue
        (frame_a, sa), (_, sb) = arms[first], arms[second]
        for farm in sites["site"].to_list():
            keep = ~frame_a["masked"].to_numpy() & (frame_a["site"] != farm).to_numpy()
            report["leave_one_farm_out"][f"{label} without {farm}"] = bootstrap(
                difference=(sa["qs"] - sb["qs"])[keep],
                month=frame_a["month"].to_numpy()[keep],
                seed=frame_a["seed"].to_numpy()[keep],
                levels=(0.95,),
            )
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        print(
            name,
            {
                k: v["qs"] and round(v["qs"], 3)
                for k, v in report["scores"][name].items()
                if k.startswith(("all rows", "kept rows"))
            },
        )
    for label, content in report["contrasts"].items():
        print(
            label,
            content["first"],
            "-",
            content["second"],
            "kept | all leads:",
            [round(x, 3) for x in content.get("kept rows | all leads | qs", [])],
        )


if __name__ == "__main__":
    main()
