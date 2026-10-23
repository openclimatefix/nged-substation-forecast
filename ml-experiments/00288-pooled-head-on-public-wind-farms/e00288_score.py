"""Score experiment 00288's arms on W1 to W3 and read the planned contrasts.

Reads R0 (experiment 259's stored rows, seeds 0 to 2) as P0, the heads from `work/heads/`, and
experiment 230's stored A0r and climatology as reference rows. Every arm is scored on the quantile
score of experiment 33 (nine levels, points of capacity), the pinball loss at the named tail
levels, the tail score T (the mean of the 0.01 and 0.99 pinball losses), coverage of the 5% to 95%
and 1% to 99% bands, and the error of the median and mean forecasts, by lead band and row set.
Contrasts resample whole months of the run and the three seeds, 50,000 times; the farms are not
resampled. The three decisive contrasts (C1 to C3, on kept rows at lead days 1 and 2) are read at
98.33%, and every other contrast at 95%.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00288-*/e00288_score.py
"""

import json
import os
from collections.abc import Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00288_common as common  # noqa: E402
import e00288_tables as tables  # noqa: E402
import e00288_wsites as wsites  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HEAD_DIR = common.WORK_DIR / "heads"
REFERENCE_ARM_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "arms"
)
"""Experiment 230's stored arms, read only."""
REFERENCE_ARMS = {"A0r-230": "A0r", "CLIM-230": "CLIM"}
SEEDS = (0, 1, 2)
KEYS = ("farm", "power_fcst_init_time", "valid_time")
RESAMPLES = 50_000
DECISIVE_LEVEL = 1 - 0.05 / 3
LEVEL_95 = 0.95
HEAD_ARMS = ("T1", "P1", "P2", "P3", "X2", "X3", "BL2")
CONTRAST_METRICS = ("qs", "tail_t", "pinball_q99", "mean_abs_error", "median_abs_error")
DECISIVE = (("C1", "P2", "P1"), ("C2", "P3", "P0"), ("C3", "P2", "X2"))
DESCRIPTIVE = (
    ("D1", "P3", "P1"),
    ("D2", "P3", "T1"),
    ("D3", "P2", "T1"),
    ("D4", "P2", "P0"),
    ("D5", "X3", "P3"),
    ("D6", "P1", "P0"),
    ("D7", "T1", "P0"),
    ("D8", "X2", "P0"),
    ("D9", "BL2", "P0"),
    ("D10", "P2", "BL2"),
    ("D11", "P3", "A0r-230"),
    ("D12", "X3", "P0"),
)


def read_arm(*, name: str, cases: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame | None:
    """Return one arm's quantiles joined to `cases` with seeds 0 to 2, or None if absent."""
    if name == "P0":
        frames = []
        farms = pl.read_parquet(tables.CLEAN_DIR / "farms.parquet")
        for seed in SEEDS:
            table = tables.w_table(set_name="scored", seed=seed, pool_farms=farms)
            frames.append(
                table.select(*KEYS, *common.LEVEL_COLUMNS).with_columns(
                    seed=pl.lit(seed, dtype=pl.Int32)
                )
            )
        frame = pl.concat(frames)
    elif name in REFERENCE_ARMS:
        path = REFERENCE_ARM_DIR / f"{REFERENCE_ARMS[name]}_spread.parquet"
        if not path.exists():
            return None
        frame = (
            pl.read_parquet(path)
            .join(sites.select("time_series_id", pl.col("site").alias("farm")), on="time_series_id")
            .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
        )
    else:
        path = HEAD_DIR / f"{name}.parquet"
        if not path.exists():
            return None
        frame = pl.read_parquet(path).select(*KEYS, "seed", *common.LEVEL_COLUMNS)
    if name == "CLIM-230":
        frame = pl.concat([frame.with_columns(seed=pl.lit(s, dtype=pl.Int32)) for s in SEEDS])
    joined = cases.join(frame, on=list(KEYS), how="inner", validate="1:m")
    return joined.sort("seed", *KEYS)


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


def main() -> None:  # noqa: PLR0912
    """Score every arm present and write the report."""
    sites = wsites.site_table().select("time_series_id", "site")
    farms = pl.read_parquet(tables.CLEAN_DIR / "farms.parquet")
    cases = (
        tables.w_table(set_name="scored", seed=0, pool_farms=farms)
        .select(*KEYS, "lead_hours", "lead_day", "actual_norm", "masked")
        .with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            quarter=pl.col("valid_time").dt.to_string("%Y-Q")
            + ((pl.col("valid_time").dt.month() - 1) // 3 + 1).cast(pl.String),
        )
    )
    arms = {}
    for name in ("P0", *REFERENCE_ARMS, *HEAD_ARMS):
        frame = read_arm(name=name, cases=cases, sites=sites.rename({"site": "site"}))
        if frame is not None:
            arms[name] = (frame, case_scores(frame=frame))
    row_sets: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
        "all rows": lambda f: np.ones(f.height, dtype=bool),
        "kept rows": lambda f: ~f["masked"].to_numpy(),
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
        report["per_farm"][name] = {
            farm: {
                "qs_kept": float(
                    scores["qs"][((frame["farm"] == farm) & ~frame["masked"]).to_numpy()].mean()
                ),
                "median_minus_mean": float(
                    scores["median_minus_mean"][(frame["farm"] == farm).to_numpy()].mean()
                ),
            }
            for farm in sites["site"].to_list()
        }
    for label, first, second in (*DECISIVE, *DESCRIPTIVE):
        if first not in arms or second not in arms:
            continue
        (frame_a, sa), (frame_b, sb) = arms[first], arms[second]
        keys = [*KEYS, "seed"]
        if not frame_a.select(keys).equals(frame_b.select(keys)):
            msg = f"{first} and {second} do not cover the same rows"
            raise ValueError(msg)
        decisive = label in {d[0] for d in DECISIVE}
        report["contrasts"][label] = {"first": first, "second": second, "decisive": decisive}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame_a) & bm(frame_a)
                is_decisive_cell = decisive and rs_name == "kept rows" and band == "days 1-2"
                levels = (LEVEL_95, DECISIVE_LEVEL) if is_decisive_cell else (LEVEL_95,)
                for metric in CONTRAST_METRICS:
                    report["contrasts"][label][f"{rs_name} | {band} | {metric}"] = bootstrap(
                        difference=(sa[metric] - sb[metric])[keep],
                        month=frame_a["month"].to_numpy()[keep],
                        seed=frame_a["seed"].to_numpy()[keep],
                        levels=levels,
                    )
    if {"P0", "P2", "BL2"} <= arms.keys():
        for band in band_masks:
            key = f"kept rows | {band}"
            gain = report["scores"]["P2"][key]["qs"] - report["scores"]["P0"][key]["qs"]
            blind = report["scores"]["BL2"][key]["qs"] - report["scores"]["P0"][key]["qs"]
            report["checks"][f"blind_share {band}"] = blind / gain if gain < 0 else None
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        print(
            name,
            {
                k: round(v["qs"], 3)
                for k, v in report["scores"][name].items()
                if k in ("kept rows | all leads", "kept rows | days 1-2")
            },
        )
    for label, content in report["contrasts"].items():
        key = "kept rows | days 1-2 | qs"
        print(
            label,
            content["first"],
            "-",
            content["second"],
            "kept | days 1-2 | qs:",
            [round(x, 3) for x in content.get(key, [])],
        )
    print(report["checks"])


if __name__ == "__main__":
    main()
