"""Score experiment 00242's arms and read the planned contrasts.

Reads `work/arms/<arm>_spread.parquet` and `<arm>_raw.parquet`, scores every arm on the tail score
T (the mean of the 0.01 and 0.99 pinball losses), the quantile score of experiment 33 (nine levels),
the pinball loss at the five named levels, the coverage of the 5% to 95% band, and the error of the
median and mean forecasts, by lead-day band and row set (all rows, rows kept by experiment 210's
mask), and bootstraps contrasts by resampling whole months (of the ENS run) and the three seeds. The
farms are not resampled. Scores are in points of capacity.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00242-*/e00242_score.py
"""

import json
import os
from collections.abc import Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00242_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ARM_DIR = common.WORK_DIR / "arms"
RESAMPLES = 50_000
ARM_NAMES = ("B1_members", "B1_mean", "B1d_members", "A0r_members", "A0r_mean")
CONTRASTS = (
    ("C1", "B1_members", "B1_mean"),
    ("C2", "A0r_members", "A0r_mean"),
    ("C3", "B1_members", "B1d_members"),
    ("C4", "B1_members", "A0r_members"),
    ("D1", "B1d_members", "B1_mean"),
)
DECISIVE: tuple[str, ...] = ("C1", "C2", "C3", "C4")
BONFERRONI_LEVEL = 1 - 0.05 / 4
FORMS = ("spread", "raw")


def read_arm(*, name: str, form: str, cases: pl.DataFrame) -> pl.DataFrame:
    """Return one arm's quantiles joined to `cases`, sorted by seed then case."""
    frame = pl.read_parquet(ARM_DIR / f"{name}_{form}.parquet")
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
        "median_abs_error": np.abs(actual - quantiles[:, common.MEDIAN_INDEX]) * scale,
        "mean_abs_error": np.abs(actual - mean_forecast) * scale,
        "median_minus_mean": (quantiles[:, common.MEDIAN_INDEX] - mean_forecast) * scale,
    }
    for name in common.NAMED_LEVELS:
        out[f"pinball_{name}"] = (
            common.pinball_at(quantiles=quantiles, actual=actual, column=name) * scale
        )
    out["tail"] = (out["pinball_q01"] + out["pinball_q99"]) / 2
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
    """Score every arm and write the report."""
    sites = common.site_table()
    cases = (
        common.cases_table(sites=sites)
        .join(
            sites.select("time_series_id", "site"), on="time_series_id", validate="m:1", suffix="_x"
        )
        .with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    )
    arms: dict[str, tuple[pl.DataFrame, dict[str, np.ndarray]]] = {}
    for form in FORMS:
        for name in ARM_NAMES:
            frame = read_arm(name=name, form=form, cases=cases)
            arms[f"{name}|{form}"] = (frame, case_scores(frame=frame))
    row_sets: dict[str, Callable[..., np.ndarray]] = {
        "all rows": lambda f: np.ones(f.height, dtype=bool),
        "kept rows": lambda f: ~f["masked"].to_numpy(),
    }
    band_masks: dict[str, Callable[..., np.ndarray]] = {
        name: lambda f, lo=lo, hi=hi: (
            (f["lead_day"].to_numpy() >= lo) & (f["lead_day"].to_numpy() <= hi)
        )
        for name, lo, hi in common.LEAD_DAY_BANDS
    }
    report: dict = {"scores": {}, "contrasts": {}, "per_farm": {}}
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
                metric: float(scores[metric][(frame["site"] == farm).to_numpy()].mean())
                for metric in ("qs", "tail", "median_minus_mean", "covered")
            }
            for farm in sites["site"].to_list()
        }
    for form in FORMS:
        for label, first, second in CONTRASTS:
            key = f"{label}|{form}"
            (frame_a, sa), (_, sb) = arms[f"{first}|{form}"], arms[f"{second}|{form}"]
            report["contrasts"][key] = {"first": first, "second": second, "form": form}
            for rs_name, rs in row_sets.items():
                for band, bm in band_masks.items():
                    keep = rs(frame_a) & bm(frame_a)
                    levels = (0.95, BONFERRONI_LEVEL) if label in DECISIVE else (0.95,)
                    for metric in ("tail", "qs", "pinball_q01", "pinball_q99", "mean_abs_error"):
                        report["contrasts"][key][f"{rs_name} | {band} | {metric}"] = bootstrap(
                            difference=(sa[metric] - sb[metric])[keep],
                            month=frame_a["month"].to_numpy()[keep],
                            seed=frame_a["seed"].to_numpy()[keep],
                            levels=levels,
                        )
    report["leave_one_farm_out"] = {}
    for label, first, second in CONTRASTS:
        (frame_a, sa), (_, sb) = arms[f"{first}|spread"], arms[f"{second}|spread"]
        for farm in sites["site"].to_list():
            keep = ~frame_a["masked"].to_numpy() & (frame_a["site"] != farm).to_numpy()
            report["leave_one_farm_out"][f"{label} without {farm}"] = bootstrap(
                difference=(sa["tail"] - sb["tail"])[keep],
                month=frame_a["month"].to_numpy()[keep],
                seed=frame_a["seed"].to_numpy()[keep],
                levels=(0.95,),
            )
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        cell = report["scores"][name]["kept rows | all leads"]
        print(name, {k: round(cell[k], 3) for k in ("tail", "qs", "covered")})
    for key, content in report["contrasts"].items():
        print(
            key,
            content["first"],
            "-",
            content["second"],
            "kept | all leads | tail:",
            [round(x, 3) for x in content["kept rows | all leads | tail"]],
        )


if __name__ == "__main__":
    main()
