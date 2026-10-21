"""Score GAM-K, the plain GAM, and lagless XGBoost winter by winter from stored predictions.

**Nothing is fitted.** Experiments 109 and 114 stored each arm's day-ahead predictions on past
weather (ERA5 temperature and CAMS irradiance) for the year after the cuts 2023-07-01, 2024-07-01,
and 2025-07-01. This script reads those sets, keeps the five decision Primaries, and scores the
winters (November to March) of 2023-24, 2024-25, and 2025-26. The six planned contrasts, the
exploratory cold-snap and evening-peak contrasts, and the checks are printed into `report.txt`
beside the write-up.

**Statistic.** The mean absolute error of the median forecast is averaged over the rows of each
local (Europe/London) day, all five Primaries pooled, then over the days. A contrast is the paired
difference of the daily means. The 95% interval resamples blocks of consecutive calendar days,
paired across arms, and a block is 7 days (14 days is the sensitivity).

    uv run python ml-experiments/00122-*/e00122_report.py
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.experiment_store import HOLDOUT_START, read_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 122
SLUG: Final[str] = "differentiable-physics-three-winter-backtest-gam-k"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
REPORT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{EXPERIMENT_ID:05d}-{SLUG}"
    / "report.txt"
)
PANEL_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR.joinpath(
    "00103-differentiable-physics-1-data-look", "panel.parquet"
)
"""Experiment 103's panel: the site labels and the ERA5 2 m temperature."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SENSITIVITY_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
WINTERS: Final[tuple[str, ...]] = ("2023-24", "2024-25", "2025-26")
EVENING_START_HOUR: Final[float] = 16.0
EVENING_END_HOUR: Final[float] = 20.0
COLD_SNAP_SHARE: Final[float] = 0.10
MIN_COLD_SNAP_DAYS: Final[int] = 3
MIN_TEMPERATURE_ROWS_PER_SITE_DAY: Final[int] = 40
BAR_POINTS: Final[float] = -1.0
"""A planned contrast meets the bar if its estimate is at most this and its interval is below 0."""
N_RESAMPLES: Final[int] = 2000
BLOCK_DAYS: Final[int] = 7
SENSITIVITY_BLOCK_DAYS: Final[int] = 14
SEED: Final[int] = 122
LOCAL_ZONE: Final[str] = "Europe/London"
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)


@dataclass(frozen=True)
class StoredArm:
    """Where one arm's predictions are stored, by the winter's cut.

    Attributes:
        label: The arm's name on the page.
        experiment_id: The experiment that wrote the sets.
        names: The set's name for each winter.
    """

    label: str
    experiment_id: int
    names: dict[str, str]


ARMS: Final[tuple[StoredArm, ...]] = (
    StoredArm(
        label="GAM",
        experiment_id=114,
        names={"2023-24": "tuning_b0", "2024-25": "replicate_b0", "2025-26": "scored_b0"},
    ),
    StoredArm(
        label="GAM-K",
        experiment_id=114,
        names={"2023-24": "tuning_b0_k", "2024-25": "replicate_b0_k", "2025-26": "scored_b0_k"},
    ),
    StoredArm(
        label="XGBoost",
        experiment_id=109,
        names={"2023-24": "tuning_x_2y", "2024-25": "replicate_x_2y", "2025-26": "scored_x_2y"},
    ),
)
REFERENCE_ARM: Final[str] = "GAM-K"
OTHER_ARMS: Final[tuple[str, ...]] = ("GAM", "XGBoost")


def _winter_label(*, day: date) -> str:
    """Return the winter a local day belongs to, such as `2024-25` for 15 January 2025."""
    start_year = day.year if day.month >= WINTER_MONTHS[0] else day.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def _site_labels() -> dict[int, str]:
    """Return the D-label of each `time_series_id` in the panel."""
    table = pl.read_parquet(PANEL_PATH, columns=["time_series_id", "site"]).unique()
    return dict(zip(table["time_series_id"].to_list(), table["site"].to_list(), strict=True))


def _read_arm(*, arm: StoredArm, winter: str) -> pl.DataFrame:
    """Return one arm's stored predictions for a winter's year, with the columns scoring needs."""
    frame = read_predictions(
        experiment_id=arm.experiment_id, name=arm.names[winter], row_set_id=ROW_SET_ID
    )
    return frame.select(
        *KEYS, "lead_hours", "actual", "switching_mask", *QUANTILE_COLUMNS
    ).with_columns(pl.col("actual", *QUANTILE_COLUMNS).cast(pl.Float64))


def _losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the absolute error of the median and the quantile score of each row, in points."""
    pinball = sum(
        pl.max_horizontal(
            level * (pl.col("actual") - pl.col(column)),
            (level - 1.0) * (pl.col("actual") - pl.col(column)),
        )
        for level, column in zip(QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True)
    )
    return frame.select(
        *KEYS,
        "lead_hours",
        "actual",
        "switching_mask",
        signed_error=pl.col("actual") - pl.col("q50"),
        absolute_error=(pl.col("actual") - pl.col("q50")).abs(),
        quantile_score=2.0 * pinball / len(QUANTILE_LEVELS),
        median_minus_mean_proxy=pl.col("q50") - pl.mean_horizontal(*QUANTILE_COLUMNS),
    )


def _winter_rows(*, winter: str, labels: dict[int, str]) -> pl.DataFrame:
    """Return the shared rows of the three arms in one winter, one row per arm and forecast row.

    The frame has the keys, `site`, `day` (local date), `local_hour`, `arm`, `absolute_error`,
    `signed_error` (observed minus median), `quantile_score`, and `median_minus_mean_proxy`. A row
    is kept only if every arm holds it and no arm marks it as inside a switching event.
    """
    per_arm = {arm.label: _read_arm(arm=arm, winter=winter) for arm in ARMS}
    shared = per_arm[ARMS[0].label].select(*KEYS)
    for frame in list(per_arm.values())[1:]:
        shared = shared.join(frame.select(*KEYS), on=list(KEYS), how="inner")
    masked = pl.concat(
        [
            frame.filter(pl.col("switching_mask").fill_null(value=False)).select(*KEYS)
            for frame in per_arm.values()
        ]
    ).unique()
    shared = shared.join(masked, on=list(KEYS), how="anti")
    parts = [
        _losses(frame=frame.join(shared, on=list(KEYS), how="inner")).with_columns(
            arm=pl.lit(label)
        )
        for label, frame in per_arm.items()
    ]
    local = pl.col("valid_time").dt.convert_time_zone(LOCAL_ZONE)
    rows = pl.concat(parts).with_columns(
        day=local.dt.date(),
        month=local.dt.month(),
        local_hour=local.dt.hour() + local.dt.minute() / 60.0,
        site=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String),
    )
    rows = rows.filter(pl.col("month").is_in(WINTER_MONTHS))
    if rows.select((pl.col("valid_time") >= HOLDOUT_START).any()).item():
        msg = "a valid time falls in the held-back period"
        raise ValueError(msg)
    return rows.filter(
        pl.col("day").map_elements(lambda d: _winter_label(day=d), return_dtype=pl.String) == winter
    )


def _daily_temperature(*, sites: tuple[str, ...]) -> dict[date, float]:
    """Return the daily mean ERA5 2 m temperature over the sites, for days with enough rows."""
    local = pl.col("time").dt.convert_time_zone(LOCAL_ZONE)
    table = (
        pl.read_parquet(PANEL_PATH, columns=["site", "time", "temperature_c"])
        .filter(pl.col("site").is_in(sites) & pl.col("temperature_c").is_not_null())
        .with_columns(day=local.dt.date())
        .group_by("day")
        .agg(temperature=pl.col("temperature_c").mean(), n=pl.len())
        .filter(pl.col("n") >= MIN_TEMPERATURE_ROWS_PER_SITE_DAY * len(sites))
    )
    return dict(zip(table["day"].to_list(), table["temperature"].to_list(), strict=True))


def _cold_snap_days(*, days: list[date], temperature: dict[date, float]) -> list[date]:
    """Return the coldest 10% of a winter's covered days (at least 3) by daily mean temperature."""
    known = [day for day in days if day in temperature]
    count = max(MIN_COLD_SNAP_DAYS, round(COLD_SNAP_SHARE * len(known)))
    return sorted(sorted(known, key=lambda day: temperature[day])[:count])


def _daily(
    *, rows: pl.DataFrame, sites: tuple[str, ...], column: str, evening: bool
) -> pl.DataFrame:
    """Return the per-day mean of a loss column per arm, over the sites (evening peak optional)."""
    selected = rows.filter(pl.col("site").is_in(sites))
    if evening:
        selected = selected.filter(
            (pl.col("local_hour") >= EVENING_START_HOUR) & (pl.col("local_hour") < EVENING_END_HOUR)
        )
    return selected.group_by("arm", "day").agg(value=pl.col(column).mean()).sort("arm", "day")


def _paired_interval(
    *, days: list[date], difference: np.ndarray, block_days: int, seed: int
) -> tuple[float, float, float]:
    """Return the mean paired difference and its 95% interval from resampled blocks of days."""
    origin = date(2020, 1, 1)
    block = np.array([(day - origin).days // block_days for day in days])
    unique_blocks = np.unique(block)
    sums = np.array([difference[block == b].sum() for b in unique_blocks])
    counts = np.array([(block == b).sum() for b in unique_blocks])
    draws = np.random.default_rng(seed).integers(
        0, len(unique_blocks), (N_RESAMPLES, len(unique_blocks))
    )
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(difference.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _contrast(
    *,
    daily: pl.DataFrame,
    other: str,
    days: set[date] | None,
    block_days: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Return GAM-K minus `other`: estimate, 95% interval, and the number of days."""
    reference = daily.filter(pl.col("arm") == REFERENCE_ARM)
    compared = daily.filter(pl.col("arm") == other)
    joined = reference.join(compared, on="day", suffix="_other").sort("day")
    if days is not None:
        joined = joined.filter(pl.col("day").is_in(days))
    if joined.height < MIN_COLD_SNAP_DAYS:
        return float("nan"), float("nan"), float("nan"), joined.height
    difference = (joined["value"] - joined["value_other"]).to_numpy()
    estimate, low, high = _paired_interval(
        days=joined["day"].to_list(), difference=difference, block_days=block_days, seed=seed
    )
    return estimate, low, high, joined.height


def _format(*, contrast: tuple[float, float, float, int]) -> str:
    """Format a contrast as `estimate (low to high)`."""
    estimate, low, high, _ = contrast
    return f"{estimate:+.2f} ({low:+.2f} to {high:+.2f})"


def _meets_bar(*, contrast: tuple[float, float, float, int]) -> bool:
    """Return whether the estimate is at most -1 point and the interval lies wholly below zero."""
    estimate, _, high, _ = contrast
    return estimate <= BAR_POINTS and high < 0.0


def _verdict(*, planned: list[tuple[float, float, float, int]]) -> str:
    """Return the verdict word the decision rule gives the six planned contrasts."""
    if any(low > 0.0 for _, low, _, _ in planned):
        return "refuted"
    met = sum(_meets_bar(contrast=contrast) for contrast in planned)
    if met == len(planned):
        return "supported"
    if met >= 3:
        return "partly supported"
    if all(high < 0.0 for _, _, high, _ in planned):
        return "small gain"
    return "inconclusive"


def _write(*, lines: list[str]) -> None:
    """Write the report beside the write-up and in the private data folder."""
    text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "report.txt").write_text(text)


def _leave_one_site_out_lines(*, rows: pl.DataFrame) -> list[str]:
    """Return the report lines of GAM-K minus each other arm with each site left out in turn."""
    lines = ["", "Leave-one-site-out, GAM-K minus the other arm on all winter days:", ""]
    lines += ["| Site left out | GAM | XGBoost |", "|---|---|---|"]
    for left_out in DECISION_SITES:
        kept = tuple(site for site in DECISION_SITES if site != left_out)
        reduced = _daily(rows=rows, sites=kept, column="absolute_error", evening=False)
        cells = [
            _format(
                contrast=_contrast(
                    daily=reduced, other=other, days=None, block_days=BLOCK_DAYS, seed=SEED
                )
            )
            for other in OTHER_ARMS
        ]
        lines.append(f"| {left_out} | " + " | ".join(cells) + " |")
    return lines


def _proxy_lines(*, rows: pl.DataFrame) -> list[str]:
    """Return the report lines of each arm's median forecast minus the mean of its 11 quantiles."""
    proxy = (
        rows.filter(pl.col("site").is_in(DECISION_SITES))
        .group_by("arm")
        .agg(proxy=pl.col("median_minus_mean_proxy").mean())
        .sort("arm")
    )
    return [
        "",
        (
            "Median forecast minus the mean of the 11 quantiles (a stand-in for median minus mean "
            "forecast; the stored sets hold quantiles only): "
            + ", ".join(f"{row['arm']} {row['proxy']:+.3f}" for row in proxy.iter_rows(named=True))
        ),
        "",
    ]


def main() -> None:
    """Score the three winters and print the report."""
    labels = _site_labels()
    temperature = _daily_temperature(sites=DECISION_SITES)
    lines = [
        "# Experiment 122: GAM-K against GAM and lagless XGBoost over three winters",
        "",
        (
            "Points of effective capacity; negative contrasts favour GAM-K. 95% intervals resample "
            f"{BLOCK_DAYS}-day blocks of days, paired across arms ({N_RESAMPLES} draws, "
            f"seed {SEED})."
        ),
        f"Sites: {', '.join(DECISION_SITES)}. Past weather (ERA5 and CAMS); nothing is refitted.",
        "",
    ]
    planned: list[tuple[float, float, float, int]] = []
    summary: dict[str, object] = {}
    numbers: dict[str, dict[str, dict[str, object]]] = {"mae": {}, "contrast": {}, "bias": {}}
    for winter in WINTERS:
        rows = _winter_rows(winter=winter, labels=labels)
        daily = _daily(rows=rows, sites=DECISION_SITES, column="absolute_error", evening=False)
        evening = _daily(rows=rows, sites=DECISION_SITES, column="absolute_error", evening=True)
        quantile = _daily(rows=rows, sites=DECISION_SITES, column="quantile_score", evening=False)
        days = sorted(daily["day"].unique().to_list())
        cold = _cold_snap_days(days=days, temperature=temperature)
        cold_set = set(cold)
        used_sites = sorted(
            rows.filter(pl.col("site").is_in(DECISION_SITES))["site"].unique().to_list()
        )
        lines += [
            f"## Winter {winter}",
            "",
            (
                f"{len(days)} covered days; sites present: {', '.join(used_sites)}; "
                f"{len(cold)} cold-snap days ({min(temperature[d] for d in cold):.1f} to "
                f"{max(temperature[d] for d in cold):.1f} C daily mean)."
            ),
            "",
            "| Arm | MAE, all days | MAE, cold-snap days | MAE, evening peak | Quantile score |",
            "|---|---|---|---|---|",
        ]
        for arm in ARMS:

            def mean_of(
                table: pl.DataFrame, selected: set[date] | None = None, label: str = arm.label
            ) -> float:
                part = table.filter(pl.col("arm") == label)
                if selected is not None:
                    part = part.filter(pl.col("day").is_in(selected))
                return float(np.mean(part["value"].to_numpy()))

            numbers["mae"].setdefault(winter, {})[arm.label] = {
                "all": mean_of(daily),
                "cold": mean_of(daily, cold_set),
                "evening": mean_of(evening),
            }
            lines.append(
                f"| {arm.label} | {mean_of(daily):.2f} | {mean_of(daily, cold_set):.2f} | "
                f"{mean_of(evening):.2f} | {mean_of(quantile):.3f} |"
            )
        lines += [
            "",
            (
                "| Other arm | All days (planned) | Cold-snap days (exploratory) | Evening peak "
                "(exploratory) | Quantile score (exploratory) | All days, 14-day blocks |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for other in OTHER_ARMS:
            main_contrast = _contrast(
                daily=daily, other=other, days=None, block_days=BLOCK_DAYS, seed=SEED
            )
            planned.append(main_contrast)
            summary[f"{winter}:{other}"] = main_contrast[:3]
            variants = {
                "all": main_contrast,
                "cold": _contrast(
                    daily=daily, other=other, days=cold_set, block_days=BLOCK_DAYS, seed=SEED
                ),
                "evening": _contrast(
                    daily=evening, other=other, days=None, block_days=BLOCK_DAYS, seed=SEED
                ),
                "quantile": _contrast(
                    daily=quantile, other=other, days=None, block_days=BLOCK_DAYS, seed=SEED
                ),
                "blocks14": _contrast(
                    daily=daily,
                    other=other,
                    days=None,
                    block_days=SENSITIVITY_BLOCK_DAYS,
                    seed=SEED,
                ),
            }
            numbers["contrast"].setdefault(winter, {})[other] = {
                key: list(value[:3]) for key, value in variants.items()
            }
            cells = [_format(contrast=value) for value in variants.values()]
            lines.append(f"| {other} | " + " | ".join(cells) + " |")
        lines += _leave_one_site_out_lines(rows=rows)
        bias = (
            rows.filter(pl.col("site").is_in(DECISION_SITES))
            .group_by("arm", "site")
            .agg(bias=pl.col("signed_error").mean())
        )
        lines += [
            "",
            (
                "Mean signed error (observed minus median forecast) by site, all winter days; "
                "positive means the arm under-forecast:"
            ),
            "",
            "| Arm | " + " | ".join(DECISION_SITES) + " |",
            "|---|" + "---|" * len(DECISION_SITES),
        ]
        for arm in ARMS:
            arm_bias = dict(
                bias.filter(pl.col("arm") == arm.label).select("site", "bias").iter_rows()
            )
            numbers["bias"].setdefault(winter, {})[arm.label] = arm_bias
            cells = [f"{arm_bias[site]:+.2f}" for site in DECISION_SITES]
            lines.append(f"| {arm.label} | " + " | ".join(cells) + " |")
        lines += _proxy_lines(rows=rows)
        if winter == WINTERS[-1]:
            extended = _daily(
                rows=rows, sites=SENSITIVITY_SITES, column="absolute_error", evening=False
            )
            lines += [f"Sensitivity, {winter} with D9 added (six Primaries):", ""]
            for other in OTHER_ARMS:
                lines.append(
                    f"- GAM-K minus {other}: "
                    + _format(
                        contrast=_contrast(
                            daily=extended, other=other, days=None, block_days=BLOCK_DAYS, seed=SEED
                        )
                    )
                )
            lines.append("")
    verdict = _verdict(planned=planned)
    met = sum(_meets_bar(contrast=contrast) for contrast in planned)
    lines += [
        "## Decision",
        "",
        (
            f"Contrasts meeting the bar (estimate at most {BAR_POINTS:+.1f} and interval below 0): "
            f"{met} of {len(planned)}."
        ),
        f"Verdict by the decision rule: {verdict}.",
        "",
    ]
    _write(lines=lines)
    (DATA_DIR / "planned_contrasts.json").write_text(json.dumps(summary, indent=2))
    (DATA_DIR / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
