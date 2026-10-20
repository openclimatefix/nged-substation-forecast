"""Print experiment 120's numbers: scores, the planned contrasts, the decision rule, and checks.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<cut>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row.

**The headline seed.** A library arm has five seeds. The headline fit is the seed with the lowest
training objective, and a contrast's estimate is that seed's. The other seeds' estimates and the
interval from pooling all five seeds (the reference arms' rows repeated once per seed) are printed
beside it.

- **Planned contrasts** (Bonferroni across the three, so 98.33% intervals resampling whole calendar
  months): C1 = SH - N, C2 = SH - G, C3 = SH - SH-K1, for the scored year and the replicate year,
  with the count of decision sites that improve, the estimate without each site, the winter
  (November to March), evening-peak, and cold-snap estimates, and the same contrast on the error of
  the mean forecast.
- **The decision rule** as the plan states it, applied to those numbers with the gate's outcome.
- **Exploratory contrasts** at 95%.
- **Checks**: the median forecast minus the mean forecast per site and arm, leave-one-site-out
  estimates, the audit triggers, and tail calibration (coverage and pinball at q01, q05, q95, q99).
- **Stability and shape mix**: matched cosines across seeds and across years, and each
  substation's weight shares.

    uv run python ml-experiments/00120-*/e00120_report.py > docs/ml-experiments/00120-*/report.txt
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import e00120_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.dp_shape_library import largest_pair_cosine, match_shapes
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
N_PLANNED: Final[int] = 3
PLANNED_LEVEL: Final[float] = 100.0 - 5.0 / N_PLANNED
EXPLORATORY_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20000
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
NIGHT_HOURS: Final[tuple[int, ...]] = (23, 0, 1, 2, 3, 4, 5, 6)
"""Valid-time hours of a winter night, where the Economy 7 step sits."""
EVENING_HOURS: Final[tuple[int, ...]] = (16, 17, 18, 19)
"""Valid-time hours (UTC; local time in winter) whose half-hours count as the evening peak."""
COLD_SNAP_C: Final[float] = 0.0
"""A cold-snap day has a mean 6-hour-smoothed ERA5 temperature below this at the site."""
MIN_COLD_SNAP_DAYS: Final[int] = 5
"""Fewer cold-snap site-days than this in a scored year give a "too few days" panel."""
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
SMALLEST_GAIN: Final[float] = 0.05
SCORED_ARMS: Final[tuple[str, ...]] = (
    "g",
    "n",
    "n_full",
    "sh",
    "sh_k1",
    "sh_g0",
    "sh_nodev",
    "sh_wd",
    "sh_all",
)
EXPERIMENT_106: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
EXTERNAL_SETS: Final[dict[str, Path]] = {
    "climatology": EXPERIMENT_106 / "predictions" / "{label}_climatology",
    "manual_heuristic": EXPERIMENT_106 / "predictions" / "{label}_manual_heuristic",
    "x_2y": common.EXPERIMENT_109_DIR / "predictions" / "{label}_x_2y",
    "x_2yamq": common.EXPERIMENT_109_DIR / "predictions" / "{label}_x_2yamq",
}
"""Reference sets stored by earlier experiments, read where they cover the same rows."""
N_109: Final[str] = "n_109"
G_109: Final[str] = "g_109"
"""Experiment 109's stored N and G, for the reproduction check."""


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment set's arm name without the cut label.
        reference: The reference set's arm name without the cut label.
        planned: True for the three pre-registered contrasts.
        sites: The sites the contrast is read on, the five decision sites if not given.
        years: The years the contrast exists in.
    """

    name: str
    treatment: str
    reference: str
    planned: bool = False
    sites: tuple[str, ...] | None = None
    years: tuple[str, ...] = ("scored", "replicate")


@cache
def site_of_id() -> dict[int, str]:
    """Return the label of each time series id in the feature cache."""
    return {f.table.time_series_id: label for label, f in common.load_cache().sites.items()}


def is_library(*, arm: str) -> bool:
    """Return True for an arm with five seeds."""
    return arm.startswith("sh")


def headline_seed(*, label: str, arm: str) -> int:
    """Return the seed with the lowest training objective for a library arm at a cut."""
    best_seed, best = 1, float("inf")
    for path in sorted((common.WORK_DIR / "fits").glob(f"{label}_{arm}_s*.json")):
        facts = json.loads(path.read_text())
        if facts["objective"] < best:
            best_seed, best = int(facts["seed"]), float(facts["objective"])
    return best_seed


def _folder(*, label: str, arm: str) -> Path:
    if arm in EXTERNAL_SETS:
        return Path(str(EXTERNAL_SETS[arm]).format(label=label))
    if arm in (N_109, G_109):
        return common.EXPERIMENT_109_DIR / "predictions" / f"{label}_{arm.removesuffix('_109')}"
    return common.PREDICTIONS_DIR / f"{label}_{arm}"


@cache
def load_arm(*, label: str, arm: str, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set with its per-row quantile score and absolute error of the mean.

    Args:
        label: The cut label: `tuning`, `replicate`, or `scored`.
        arm: The arm's name.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The set with columns `site_name`, `qs`, and `ae` added.
    """
    frame = pl.read_parquet(_folder(label=label, arm=arm) / "predictions.parquet")
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    mean = pl.col("mean") if "mean" in frame.columns else pl.col("q50")
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - mean).abs(),
        site_name=pl.col("time_series_id").replace_strict(site_of_id()),
    )


def seeded(*, label: str, arm: str, pooled: bool, drop_switching: bool = True) -> pl.DataFrame:
    """Return an arm's rows with the seed column the pairing needs.

    Args:
        label: The cut label.
        arm: The arm's name.
        pooled: True to keep all five seeds of a library arm (and repeat a reference arm once per
            seed); false to keep the headline seed only, with every arm on seed 0.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The rows.
    """
    frame = load_arm(label=label, arm=arm, drop_switching=drop_switching)
    if is_library(arm=arm):
        if pooled:
            return frame
        seed = headline_seed(label=label, arm=arm)
        return frame.filter(pl.col("seed") == seed).with_columns(seed=pl.lit(0, dtype=pl.Int8))
    if not pooled:
        return frame.with_columns(seed=pl.lit(0, dtype=pl.Int8))
    return pl.concat([frame.with_columns(seed=pl.lit(seed, dtype=pl.Int8)) for seed in range(1, 6)])


def paired(
    *,
    label: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    pooled: bool = False,
    drop_switching: bool = True,
    keep: pl.Expr | None = None,
) -> pl.DataFrame:
    """Return both arms' rows on the sites, paired, as the frame `studies.bootstrap` reads.

    Args:
        label: The cut label.
        treatment: The treatment arm.
        reference: The reference arm.
        sites: The sites to keep.
        pooled: True to pool the seeds.
        drop_switching: True to leave out the rows inside the switching masks.
        keep: An extra filter on the rows (a season or a cold-snap rule), or `None`.

    Returns:
        The paired loss frame.
    """
    arms = {}
    for name in (treatment, reference):
        frame = seeded(label=label, arm=name, pooled=pooled, drop_switching=drop_switching)
        frame = frame.filter(pl.col("site_name").is_in(sites))
        if keep is not None:
            frame = frame.filter(keep)
        arms[name] = frame.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "seed", "qs", "ae"
        )
    first, second = list(arms.values())
    keys = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    shared = first.select(keys).join(second.select(keys), on=keys, how="inner")
    arms = {name: frame.join(shared, on=keys, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *,
    label: str,
    contrast: Contrast,
    sites: tuple[str, ...],
    keep: pl.Expr | None = None,
    drop_switching: bool = True,
    pooled: bool = False,
) -> tuple[float, float, int]:
    """Return the mean paired difference in quantile score, in the mean absolute error, and rows.

    Args:
        label: The cut label.
        contrast: The contrast.
        sites: The sites to include.
        keep: An extra row filter, or `None`.
        drop_switching: True to leave out the rows inside the switching masks.
        pooled: True to pool the seeds.

    Returns:
        The treatment-minus-reference means of `qs` and `ae`, and the number of rows.

    Raises:
        ValueError: If the two arms share no rows.
    """
    losses = paired(
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=sites,
        pooled=pooled,
        drop_switching=drop_switching,
        keep=keep,
    )
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    if means.height < 2:
        msg = f"no shared rows for {contrast.name} at {sites}"
        raise ValueError(msg)
    treatment = means.filter(pl.col("arm") == contrast.treatment).row(0, named=True)
    reference = means.filter(pl.col("arm") == contrast.reference).row(0, named=True)
    return treatment["qs"] - reference["qs"], treatment["ae"] - reference["ae"], int(treatment["n"])


def interval(
    *, label: str, contrast: Contrast, level: float, pooled: bool = False
) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast on the contrast's sites."""
    losses = paired(
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=contrast.sites or DECISION,
        pooled=pooled,
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="qs",
        level=level,
        n_resamples=N_RESAMPLES,
    )


def sites_improving(*, label: str, contrast: Contrast) -> tuple[int, int]:
    """Return how many sites have a negative difference, and how many sites have rows."""
    better, present = 0, 0
    for site in contrast.sites or DECISION:
        try:
            difference, _, _ = estimate(label=label, contrast=contrast, sites=(site,))
        except ValueError:
            continue
        present += 1
        better += int(difference < 0.0)
    return better, present


@cache
def cold_days(*, label: str) -> pl.DataFrame:
    """Return the cold-snap site-days of a scored year: (site_name, date).

    A cold-snap day has a mean 6-hour-smoothed ERA5 temperature below `COLD_SNAP_C` at the site.
    The rule is fixed in the plan before any score is read.
    """
    cache_ = common.load_cache()
    cut = common.CUTS[label]
    end = common.year_end(cut=cut)
    times = cache_.end_times
    rows = []
    for site, features in cache_.sites.items():
        keep = (times > np.datetime64(cut.replace(tzinfo=None), "us")) & (
            times <= np.datetime64(end.replace(tzinfo=None), "us")
        )
        temperature = features.smoothed_temperature_c[:, 0]
        frame = pl.DataFrame(
            {
                "date": (times[keep] - np.timedelta64(30, "m")).astype("datetime64[D]"),
                "t": temperature[keep],
            }
        )
        daily = frame.group_by("date").agg(pl.col("t").mean()).filter(pl.col("t") < COLD_SNAP_C)
        rows.append(daily.select(site_name=pl.lit(site), date=pl.col("date")))
    return pl.concat(rows)


def season_filters(*, label: str) -> dict[str, pl.Expr]:
    """Return the row filters of the winter breakdown for a scored year."""
    valid = pl.col("valid_time")
    day = (valid - pl.duration(minutes=30)).dt.date()
    cold = cold_days(label=label)
    in_cold = pl.struct(["site_name", day.alias("date")]).is_in(
        cold.select(pl.struct("site_name", "date")).to_series()
    )
    month = valid.dt.month().is_in(WINTER_MONTHS)
    return {
        "all": pl.lit(True),
        "Nov-Mar": month,
        "Apr-Oct": ~month,
        "Nov-Mar evening peak": month & (valid.dt.hour().is_in(EVENING_HOURS)),
        "Nov-Mar nights (23:00-07:00)": month & valid.dt.hour().is_in(NIGHT_HOURS),
        "cold-snap days": in_cold,
    }


def contrasts() -> tuple[Contrast, ...]:
    """Return every contrast of the plan."""
    bar = x_bar()
    return (
        Contrast("C1 SH - N", "sh", "n", planned=True),
        Contrast("C2 SH - G", "sh", "g", planned=True),
        Contrast("C3 SH - SH-K1", "sh", "sh_k1", planned=True),
        Contrast("SH - SH-g0", "sh", "sh_g0"),
        Contrast("SH - SH-nodev", "sh", "sh_nodev"),
        Contrast("SH - N-full", "sh", "n_full"),
        Contrast(f"SH - {bar}", "sh", bar),
        Contrast("SH-wd - SH", "sh_wd", "sh"),
        Contrast("SH-all - SH", "sh_all", "sh"),
        Contrast("N - G", "n", "g"),
        Contrast("SH - climatology", "sh", "climatology"),
        Contrast("SH - manual_heuristic", "sh", "manual_heuristic"),
    )


def x_bar() -> str:
    """Return the lagless XGBoost bar: the better of x_2y and x_2yamq on the tuning year."""
    scores = {}
    for arm in ("x_2y", "x_2yamq"):
        frame = load_arm(label="tuning", arm=arm)
        scores[arm] = float(
            frame.filter(pl.col("site_name").is_in(DECISION))["qs"].to_numpy().mean()
        )
    return min(scores, key=lambda arm: scores[arm])


def contrast_row(*, contrast: Contrast, level: float) -> dict[str, object]:
    """Return one contrast's row: each year's estimate, intervals, and leave-one-site-out values."""
    sites = contrast.sites or DECISION
    row: dict[str, object] = {"contrast": contrast.name}
    for label in contrast.years:
        difference, mean_error, n_rows = estimate(label=label, contrast=contrast, sites=sites)
        row[f"{label} QS"] = difference
        row[f"{label} MAE of mean"] = mean_error
        row[f"{label} rows"] = n_rows
        better, present = sites_improving(label=label, contrast=contrast)
        row[f"{label} sites better"], row[f"{label} sites with rows"] = better, present
        row[f"{label} all rows"] = estimate(
            label=label, contrast=contrast, sites=sites, drop_switching=False
        )[0]
        row[f"{label} low"], row[f"{label} high"] = interval(
            label=label, contrast=contrast, level=level
        )
        if is_library(arm=contrast.treatment):
            row[f"{label} pooled-seed low"], row[f"{label} pooled-seed high"] = interval(
                label=label, contrast=contrast, level=level, pooled=True
            )
    label = contrast.years[0]
    for left_out in DECISION:
        if left_out in sites and len(sites) > 1:
            row[f"without {left_out}"] = estimate(
                label=label, contrast=contrast, sites=tuple(s for s in sites if s != left_out)
            )[0]
    return row


def winter_table(*, contrast: Contrast, level: float) -> pl.DataFrame:
    """Return a contrast broken down by season, with paired intervals and the row count.

    Args:
        contrast: The contrast.
        level: The interval level.

    Returns:
        One row per year and season: the estimate, the interval, the rows, and the cold-snap day
        count; a season with too few rows or days says so in `note`.
    """
    rows = []
    for label in contrast.years:
        filters = season_filters(label=label)
        n_cold = cold_days(label=label).filter(pl.col("site_name").is_in(DECISION)).height
        for season, keep in filters.items():
            try:
                difference, mean_error, n_rows = estimate(
                    label=label, contrast=contrast, sites=DECISION, keep=keep
                )
                losses = paired(
                    label=label,
                    treatment=contrast.treatment,
                    reference=contrast.reference,
                    sites=DECISION,
                    keep=keep,
                )
                low, high = bootstrap_difference_at_level(
                    losses=losses,
                    treatment=contrast.treatment,
                    reference=contrast.reference,
                    metric="qs",
                    level=level,
                    n_resamples=N_RESAMPLES,
                )
            except ValueError:
                difference = mean_error = low = high = float("nan")
                n_rows = 0
            note = ""
            if season == "cold-snap days" and n_cold < MIN_COLD_SNAP_DAYS:
                note = f"too few days ({n_cold} site-days)"
            rows.append(
                {
                    "contrast": contrast.name,
                    "year": label,
                    "season": season,
                    "QS": difference,
                    "MAE of mean": mean_error,
                    "low": low,
                    "high": high,
                    "rows": n_rows,
                    "cold site-days": n_cold if season == "cold-snap days" else None,
                    "note": note,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def absolute_table() -> pl.DataFrame:
    """Return every arm's quantile score by site for each cut, on rows outside the masks."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in (*SCORED_ARMS, "climatology", "manual_heuristic", "x_2y", "x_2yamq"):
            try:
                frame = seeded(label=label, arm=arm, pooled=False)
            except FileNotFoundError:
                continue
            decision = frame.filter(pl.col("site_name").is_in(DECISION))
            by_site = frame.group_by("site_name").agg(pl.col("qs").mean())
            row: dict[str, object] = {
                "cut": label,
                "arm": arm,
                "five sites QS": decision["qs"].mean(),
                "five sites MAE of mean": decision["ae"].mean(),
            }
            rows.append({**row, **dict(by_site.select("site_name", "qs").iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None).sort("cut", "five sites QS")


def seed_spread_table() -> pl.DataFrame:
    """Return SH's score on the five sites for each seed, and each seed's training objective."""
    rows = []
    for label in ("scored", "replicate"):
        frame = load_arm(label=label, arm="sh").filter(pl.col("site_name").is_in(DECISION))
        for seed, part in frame.group_by("seed", maintain_order=True):
            path = common.WORK_DIR / "fits" / f"{label}_sh_s{seed[0]}.json"
            facts = json.loads(path.read_text())
            rows.append(
                {
                    "cut": label,
                    "seed": seed[0],
                    "five sites QS": part["qs"].mean(),
                    "objective": facts["objective"],
                    "converged": facts["converged"],
                    "sweeps": facts["n_sweeps"],
                    "headline": seed[0] == headline_seed(label=label, arm="sh"),
                }
            )
    return pl.DataFrame(rows)


def median_minus_mean() -> pl.DataFrame:
    """Return the mean of (q50 - mean forecast) per site for the headline arms, in points."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in SCORED_ARMS:
            frame = seeded(label=label, arm=arm, pooled=False)
            by_site = frame.group_by("site_name").agg((pl.col("q50") - pl.col("mean")).mean())
            rows.append({"year": label, "arm": arm, **dict(by_site.iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None)


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels, overall and in winter."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("g", "n", "n_full", "sh", "sh_k1"):
            frame = seeded(label=label, arm=arm, pooled=False).filter(
                pl.col("site_name").is_in(DECISION)
            )
            for season, part in (
                ("all", frame),
                ("Nov-Mar", frame.filter(pl.col("valid_time").dt.month().is_in(WINTER_MONTHS))),
            ):
                actual = part["actual"].to_numpy().astype(np.float64)
                row: dict[str, object] = {"year": label, "arm": arm, "season": season}
                for column in TAIL_LEVELS:
                    level = QUANTILE_LEVELS[QUANTILE_COLUMNS.index(column)]
                    predicted = part[column].to_numpy().astype(np.float64)
                    error = actual - predicted
                    row[f"{column} coverage"] = float((actual <= predicted).mean())
                    row[f"{column} pinball"] = float(
                        np.maximum(level * error, (level - 1.0) * error).mean()
                    )
                rows.append(row)
    return pl.DataFrame(rows)


def reproduction_table() -> pl.DataFrame:
    """Return this experiment's G and N against experiment 109's stored sets (trigger 1)."""
    rows = []
    for label in ("scored", "replicate"):
        for arm, stored in (("g", G_109), ("n", N_109)):
            mine = (
                seeded(label=label, arm=arm, pooled=False)
                .filter(pl.col("site_name").is_in(DECISION))["qs"]
                .to_numpy()
                .mean()
            )
            theirs = (
                load_arm(label=label, arm=stored)
                .filter(pl.col("site_name").is_in(DECISION))["qs"]
                .to_numpy()
                .mean()
            )
            rows.append(
                {
                    "year": label,
                    "arm": arm,
                    "here": mine,
                    "experiment 109": theirs,
                    "difference": mine - theirs,
                }
            )
    return pl.DataFrame(rows)


def fit_files(*, label: str, arm: str) -> list[tuple[int, dict, dict]]:
    """Return (seed, arrays, facts) of every stored library fit of an arm at a cut."""
    out = []
    for path in sorted((common.WORK_DIR / "fits").glob(f"{label}_{arm}_s*.npz")):
        arrays = dict(np.load(path))
        facts = json.loads(path.with_suffix(".json").read_text())
        out.append((int(facts["seed"]), arrays, facts))
    return out


def stability_table() -> pl.DataFrame:
    """Return the matched-cosine stability readouts across seeds and across years."""
    rows = []
    for label in ("tuning", "replicate", "scored"):
        fits = fit_files(label=label, arm="sh")
        if len(fits) < 2:
            continue
        reference = fits[0][1]["shapes"]
        cosines = []
        for _, arrays, _ in fits[1:]:
            cosines.extend(match_shapes(first=reference, second=arrays["shapes"])[1].tolist())
        rows.append(
            {
                "comparison": f"across seeds at {label}",
                "median cosine": float(np.median(cosines)),
                "minimum cosine": float(np.min(cosines)),
                "largest pair |cosine| (headline)": largest_pair_cosine(shapes=reference),
            }
        )
    headlines = {}
    for label in ("tuning", "replicate", "scored"):
        fits = fit_files(label=label, arm="sh")
        if fits:
            best = min(fits, key=lambda fit: fit[2]["objective"])
            headlines[label] = best[1]["shapes"]
    for first, second in (("tuning", "replicate"), ("replicate", "scored"), ("tuning", "scored")):
        if first in headlines and second in headlines:
            cosines = match_shapes(first=headlines[first], second=headlines[second])[1]
            rows.append(
                {
                    "comparison": f"headline fits, {first} against {second}"
                    + (" (disjoint windows)" if (first, second) == ("tuning", "scored") else ""),
                    "median cosine": float(np.median(cosines)),
                    "minimum cosine": float(np.min(cosines)),
                    "largest pair |cosine| (headline)": float("nan"),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def mix_table() -> pl.DataFrame:
    """Return each substation's weight shares and deviation share in the scored-year headline fit.

    The range over the five seeds' matched fits is the interval: the plan's month-block interval
    would need a refit per resample and is not computed (see the write-up).
    """
    fits = fit_files(label="scored", arm="sh")
    best = min(fits, key=lambda fit: fit[2]["objective"])
    reference = best[1]["shapes"]
    labels = [str(label) for label in best[1]["labels"]]
    shares_by_seed = []
    for _, arrays, _ in fits:
        order, _ = match_shapes(first=reference, second=arrays["shapes"])
        weights = arrays["weights"][:, order]
        shares_by_seed.append(weights / weights.sum(axis=1, keepdims=True))
    stacked = np.stack(shares_by_seed)
    library_energy = np.linalg.norm(best[1]["weights"] @ reference, axis=1) ** 2
    deviation_energy = np.linalg.norm(best[1]["deviations"], axis=1) ** 2
    rows = [
        {
            "site": label,
            "shape": k + 1,
            "share": float(stacked[fits.index(best), position, k]),
            "low over seeds": float(stacked[:, position, k].min()),
            "high over seeds": float(stacked[:, position, k].max()),
            "deviation share of profile": float(
                deviation_energy[position] / (library_energy[position] + deviation_energy[position])
            ),
        }
        for position, label in enumerate(labels)
        for k in range(reference.shape[0])
    ]
    return pl.DataFrame(rows)


def _number(*, entry: dict[str, object], key: str) -> float:
    """Return a numeric cell of a contrast row."""
    return float(entry[key])  # ty: ignore[invalid-argument-type]


STEP_COLUMN: Final[int] = 2
"""The profile column of 01:00, where the Economy 7 step starts."""
MIN_SHAPE_USERS: Final[int] = 2
MIN_SHARE: Final[float] = 0.10
"""A shape is a noise shape if fewer than this many substations give it this share of weight."""


def step_table() -> pl.DataFrame:
    """Return the 01:00 step in every substation's profile and in every shape (scored-year fit).

    For each substation, the rise of the fitted weekday profile at 01:00 (column 2 minus column 1,
    in points of capacity) and its contrast over 01:00 to 07:00 against the neighbouring half-hours;
    for each shape, its rise at 01:00 and the substations that use it with a share of 10% or more.
    """
    fits = fit_files(label="scored", arm="sh")
    best = min(fits, key=lambda fit: fit[2]["objective"])
    arrays = best[1]
    labels = [str(label) for label in arrays["labels"]]
    shapes, weights, deviations = arrays["shapes"], arrays["weights"], arrays["deviations"]
    rows = []
    for position, label in enumerate(labels):
        profile = weights[position] @ shapes + deviations[position]
        weekday = profile[:48]
        rows.append(
            {
                "kind": "substation profile (library + deviation)",
                "name": label,
                "rise at 01:00": float(weekday[STEP_COLUMN] - weekday[STEP_COLUMN - 1]),
                "contrast 01:00-07:00": float(
                    weekday[STEP_COLUMN : STEP_COLUMN + 12].mean()
                    - weekday[[0, 1, STEP_COLUMN + 12, STEP_COLUMN + 13]].mean()
                ),
                "users": None,
            }
        )
    shares = weights / weights.sum(axis=1, keepdims=True)
    for k in range(shapes.shape[0]):
        users = int((shares[:, k] >= MIN_SHARE).sum())
        rows.append(
            {
                "kind": "shape",
                "name": f"shape {k + 1}",
                "rise at 01:00": float(shapes[k, STEP_COLUMN] - shapes[k, STEP_COLUMN - 1]),
                "contrast 01:00-07:00": float(
                    shapes[k, STEP_COLUMN : STEP_COLUMN + 12].mean()
                    - shapes[k, [0, 1, STEP_COLUMN + 12, STEP_COLUMN + 13]].mean()
                ),
                "users": users,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"users": pl.Int64}, infer_schema_length=None)


COLD_NIGHT_C: Final[float] = 2.0
MILD_NIGHT_C: Final[float] = 10.0


def step_by_temperature() -> pl.DataFrame:
    """Return the 01:00 step of the actual power and of SH's mean forecast on cold and mild nights.

    The step is the mean over rows at 01:00 to 01:30 local time minus the mean over rows at 00:30
    to 01:00, in points of capacity, separately for rows whose 6-hour-smoothed ERA5 temperature is
    below `COLD_NIGHT_C` (cold) and above `MILD_NIGHT_C` (mild). The profile columns are local
    time, so 01:00 local is 01:00 UTC in winter and 00:00 UTC in summer.
    """
    cache_ = common.load_cache()
    rows = []
    for label in ("scored", "replicate"):
        frame = seeded(label=label, arm="sh", pooled=False).filter(
            pl.col("site_name").is_in(DECISION)
        )
        local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
        end_minutes = local.dt.hour() * 60 + local.dt.minute()
        for site in DECISION:
            features = cache_.sites[site]
            temperature = pl.DataFrame(
                {
                    "valid_time": pl.Series(cache_.end_times).dt.replace_time_zone("UTC"),
                    "temperature": features.smoothed_temperature_c[:, 0],
                }
            )
            part = frame.filter(pl.col("site_name") == site).join(
                temperature, on="valid_time", how="left"
            )
            part = part.with_columns(end_minutes=end_minutes)
            for name, condition in (
                ("cold (below 2 C)", pl.col("temperature") < COLD_NIGHT_C),
                ("mild (above 10 C)", pl.col("temperature") > MILD_NIGHT_C),
            ):
                kept = part.filter(condition)
                for quantity in ("actual", "mean"):
                    after = kept.filter(pl.col("end_minutes") == 90)[quantity].to_numpy()
                    before = kept.filter(pl.col("end_minutes") == 60)[quantity].to_numpy()
                    rows.append(
                        {
                            "year": label,
                            "site": site,
                            "nights": name,
                            "series": quantity,
                            "step at 01:00": float(after.mean() - before.mean())
                            if len(after) and len(before)
                            else float("nan"),
                            "rows": len(after),
                        }
                    )
    return pl.DataFrame(rows, infer_schema_length=None)


def audit_triggers(*, planned: pl.DataFrame, reproduction: pl.DataFrame) -> list[str]:
    """Return which of the plan's audit triggers the numbers fire (trigger 7 is judged by hand)."""
    fired = []
    if float(np.abs(reproduction["difference"].to_numpy()).max()) > 0.01:
        fired.append("1 (G or N not reproduced within 0.01)")
    c1 = planned.filter(pl.col("contrast").str.starts_with("C1")).row(0, named=True)
    c2 = planned.filter(pl.col("contrast").str.starts_with("C2")).row(0, named=True)
    if _number(entry=c2, key="scored QS") < -0.1:
        fired.append("2 (SH beats G by more than 0.1)")
    fired.extend(
        f"3 (quantile score and mean error disagree: {entry['contrast']})"
        for entry in (c1, c2)
        if _number(entry=entry, key="scored QS") * _number(entry=entry, key="scored MAE of mean")
        < 0.0
    )
    width = _number(entry=c1, key="scored high") - _number(entry=c1, key="scored low")
    if width > 0.09 or width < 0.022:
        fired.append(f"6 (C1 interval {width:.3f} wide)")
    for entry in (c1, c2):
        signs = [
            _number(entry=entry, key=f"without {site}") < 0.0
            for site in DECISION
            if f"without {site}" in entry
        ]
        if signs and len(set(signs)) > 1:
            fired.append(f"4 (leaving one site out flips {entry['contrast']})")
    return fired


def main() -> None:
    """Print every table of the report as comma-separated values."""
    print(f"== Lagless XGBoost bar (better of x_2y and x_2yamq on the tuning year): {x_bar()} ==")
    print("== Absolute quantile score by arm, points of capacity (headline seed) ==")
    print(absolute_table().write_csv(float_precision=3), end="")
    print("== Seeds of SH ==")
    print(seed_spread_table().write_csv(float_precision=3), end="")
    print(f"== Planned contrasts ({PLANNED_LEVEL:.2f}% intervals), negative = treatment better ==")
    planned = pl.DataFrame(
        [contrast_row(contrast=c, level=PLANNED_LEVEL) for c in contrasts() if c.planned],
        infer_schema_length=None,
    )
    print(planned.write_csv(float_precision=3), end="")
    print("== Exploratory contrasts (95% intervals) ==")
    exploratory = []
    for contrast in contrasts():
        if contrast.planned:
            continue
        try:
            exploratory.append(contrast_row(contrast=contrast, level=EXPLORATORY_LEVEL))
        except (FileNotFoundError, ValueError) as error:
            print(f"# skipped {contrast.name}: {error}")
    print(pl.DataFrame(exploratory, infer_schema_length=None).write_csv(float_precision=3), end="")
    print("== Winter breakdown of the planned contrasts and SH against the references ==")
    for contrast in contrasts():
        if contrast.planned or contrast.name in ("SH - N-full", "SH - climatology"):
            print(
                winter_table(contrast=contrast, level=PLANNED_LEVEL).write_csv(float_precision=3),
                end="",
            )
    print("== Reproduction of experiment 109's G and N (trigger 1) ==")
    reproduction = reproduction_table()
    print(reproduction.write_csv(float_precision=3), end="")
    print("== Median forecast minus mean forecast, points ==")
    print(median_minus_mean().write_csv(float_precision=3), end="")
    print("== Tail calibration on the five decision sites ==")
    print(tail_table().write_csv(float_precision=4), end="")
    print("== Stability of the shapes ==")
    print(stability_table().write_csv(float_precision=3), end="")
    print("== Shape mix of the scored-year headline fit ==")
    print(mix_table().write_csv(float_precision=3), end="")
    print("== The 01:00 step in the profiles and shapes (scored-year headline fit) ==")
    print(step_table().write_csv(float_precision=3), end="")
    print("== The 01:00 step on cold and mild nights (actual and SH mean forecast) ==")
    print(step_by_temperature().write_csv(float_precision=3), end="")
    print("== Audit triggers that the numbers fire ==")
    print(audit_triggers(planned=planned, reproduction=reproduction))


if __name__ == "__main__":
    main()
