"""Score experiment 11: write the predictions, then print every table the write-up quotes.

The planned contrasts C1 to C6 resample 7-day blocks of local days inside each window, after
averaging the fitting seeds. C5 resamples whole months with `studies.bootstrap`. All errors are the
mean absolute error of the median in points of capacity (power over effective capacity, times 100).

    cd ml-experiments/00011-school-term-and-return-to-work-columns-stage1a-demand
    uv run python e00011_report.py
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00011_fit as fit
import e00011_term_dates as term_dates
import numpy as np
import polars as pl
from baseline_forecasters.climatology import (
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap, write_predictions

EXPERIMENT_ID: Final[int] = 11
RESULTS_DIR: Final[Path] = fit.EXPERIMENT_DIR / "results"
ROW_SET_ID: Final[str] = "era5_val_3folds_2023_07_to_2026_06_v1"
N_PLANNED_CONTRASTS: Final[int] = 6
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_PLANNED_CONTRASTS)
"""Bonferroni-widened coverage, in percent, across the six planned contrasts."""
N_RESAMPLES: Final[int] = 20_000
BLOCK_DAYS: Final[int] = 7
BLOCK_SEED: Final[int] = 11
CLIMATOLOGY: Final[str] = "climatology"
MEDIAN_COLUMN: Final[str] = CLIMATOLOGY_QUANTILE_COLUMNS[25]
"""The climatology's equiprobable quantile at level 25.5 / 51 = 0.5."""
ARM_NAMES: Final[tuple[str, ...]] = tuple(arm.name for arm in fit.ARMS)
TYPES: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
FOLD_YEAR: Final[dict[str, int]] = {"select_2023": 2023, "select_2024": 2024, "confirm_2025": 2025}
WINDOW_NAMES: Final[tuple[str, ...]] = (
    "Return (2 to 20 January)",
    "Easter",
    "Half-term holidays",
    "Summer holiday",
    "Christmas weeks (15 December to 7 January)",
)
PLANNED_WINDOWS: Final[tuple[str, ...]] = WINDOW_NAMES[:3]


def read_fits() -> pl.DataFrame:
    """Read every unit file: rows `(fold, seed, time_series_id, valid_time, target, <arms>)`."""
    pieces = []
    for fold_name in fit.FOLDS:
        for seed_dir in sorted((fit.FITS_DIR / fold_name).glob("seed*")):
            seed = int(seed_dir.name.removeprefix("seed"))
            pieces += [
                pl.read_parquet(path).with_columns(fold=pl.lit(fold_name), seed=pl.lit(seed))
                for path in sorted(seed_dir.glob("*.parquet"))
            ]
    return pl.concat(pieces)


def with_climatology(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add a `climatology` column: the month-by-half-hour median of each fold's training window."""
    features = pl.read_parquet(
        fit.FEATURES_DIR / "features.parquet",
        columns=["time_series_id", "valid_time", "power_mw", "effective_capacity_mw"],
    )
    capacity = features.group_by("time_series_id").agg(c=pl.col("effective_capacity_mw").first())
    pieces = []
    for fold_name, fold in fit.FOLDS.items():
        training = features.filter(
            pl.col("power_mw").is_not_null(),
            pl.col("valid_time") >= fit.FULL_TRAIN_START,
            pl.col("valid_time") < fold.train_end,
        ).select("time_series_id", "valid_time", power=pl.col("power_mw"))
        lookup = _pooled_quantile_lookup(training).select(
            "time_series_id",
            "local_month",
            "local_half_hour_of_day",
            "local_is_weekend",
            median_mw=pl.col(MEDIAN_COLUMN),
        )
        keyed = rows.filter(pl.col("fold") == fold_name).with_columns(
            **_local_calendar_cell_keys(pl.col("valid_time"), time_zone="Europe/London")
        )
        joined = keyed.join(
            lookup,
            on=["time_series_id", "local_month", "local_half_hour_of_day", "local_is_weekend"],
            how="left",
        )
        pieces.append(
            joined.join(capacity, on="time_series_id")
            .with_columns(**{CLIMATOLOGY: pl.col("median_mw") / pl.col("c")})
            .drop("median_mw", "c", "local_month", "local_half_hour_of_day", "local_is_weekend")
        )
    return pl.concat(pieces)


def save_predictions(*, rows: pl.DataFrame) -> None:
    """Write one prediction set per arm, skipping a set that exists."""
    for arm in (*ARM_NAMES, CLIMATOLOGY):
        name = f"e00011-{arm.replace('_', '-')}"
        try:
            write_predictions(
                experiment_id=EXPERIMENT_ID,
                name=name,
                frame=rows.select(
                    "fold",
                    "seed",
                    "time_series_id",
                    "valid_time",
                    "target",
                    prediction=pl.col(arm),
                ),
                kind="stage1",
                row_set_id=ROW_SET_ID,
            )
        except FileExistsError:
            print(f"{name}: exists")


def window_dates() -> dict[str, dict[str, list[list[date]]]]:
    """Return, for each window and fold, the list of occurrences, each a list of local dates."""
    terms = term_dates.read_terms()
    year_terms = {
        (row["academic_year"], row["term"]): (row["first_day"], row["last_day"])
        for row in terms.iter_rows(named=True)
    }
    out: dict[str, dict[str, list[list[date]]]] = {name: {} for name in WINDOW_NAMES}

    def span(first: date, last: date) -> list[date]:
        return [first + timedelta(days=i) for i in range((last - first).days + 1)]

    for fold_name, year in FOLD_YEAR.items():
        this_year = f"{year}-{str(year + 1)[2:]}"
        last_year = f"{year - 1}-{str(year)[2:]}"
        out[WINDOW_NAMES[0]][fold_name] = [span(date(year + 1, 1, 2), date(year + 1, 1, 20))]
        out[WINDOW_NAMES[4]][fold_name] = [span(date(year, 12, 15), date(year + 1, 1, 7))]
        end4, start5 = year_terms[this_year, 4][1], year_terms[this_year, 5][0]
        out[WINDOW_NAMES[1]][fold_name] = [
            span(end4 + timedelta(days=1), start5 + timedelta(days=5))
        ]
        holidays = [
            span(
                year_terms[this_year, a][1] + timedelta(days=1),
                year_terms[this_year, b][0] - timedelta(days=1),
            )
            for a, b in ((1, 2), (3, 4), (5, 6))
        ]
        out[WINDOW_NAMES[2]][fold_name] = holidays
        out[WINDOW_NAMES[3]][fold_name] = [
            span(year_terms[last_year, 6][1] + timedelta(days=1), date(year, 9, 2))
        ]
    return out


def daily_errors(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the mean absolute error per type, fold, seed-averaged local day, and arm column."""
    arms = [*ARM_NAMES, CLIMATOLOGY]
    types = pl.read_parquet(fit.FEATURES_DIR / "series.parquet").select(
        "time_series_id", "substation_type"
    )
    return (
        rows.join(types, on="time_series_id")
        .with_columns(
            local_date=fit.local_date_of_period(valid_time=pl.col("valid_time")),
            **{f"e_{a}": (pl.col(a) - pl.col("target")).abs() * 100.0 for a in arms},
        )
        .group_by("substation_type", "fold", "local_date", "seed")
        .agg([pl.col(f"e_{a}").mean().alias(a) for a in arms])
        .group_by("substation_type", "fold", "local_date")
        .agg(pl.col(arms).mean())
    )


def tagged_days(
    *, daily: pl.DataFrame, windows: dict[str, dict[str, list[list[date]]]]
) -> pl.DataFrame:
    """Return the daily errors with a `window`, an `occurrence`, and a 7-day `block` per row."""
    tags = []
    for window, per_fold in windows.items():
        for fold_name, occurrences in per_fold.items():
            for number, dates in enumerate(occurrences):
                tags += [
                    {
                        "window": window,
                        "fold": fold_name,
                        "local_date": d,
                        "block": f"{window}|{fold_name}|{number}|{i // BLOCK_DAYS}",
                    }
                    for i, d in enumerate(dates)
                ]
    return daily.join(pl.DataFrame(tags), on=["fold", "local_date"], how="inner")


def block_interval(*, difference: pl.DataFrame, level: float) -> tuple[float, float, float]:
    """Return the mean difference and its interval, resampling the `block` column's blocks."""
    by_block = difference.group_by("block").agg(total=pl.col("difference").sum(), days=pl.len())
    totals, days = by_block["total"].to_numpy(), by_block["days"].to_numpy().astype(float)
    generator = np.random.default_rng(BLOCK_SEED)
    draws = generator.integers(0, len(totals), size=(N_RESAMPLES, len(totals)))
    means = totals[draws].sum(axis=1) / days[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return (
        float(totals.sum() / days.sum()),
        float(np.percentile(means, tail)),
        float(np.percentile(means, 100.0 - tail)),
    )


def contrast(
    *,
    tagged: pl.DataFrame,
    substation_type: str,
    treatment: str,
    reference: str,
    windows: tuple[str, ...],
    level: float,
) -> dict[str, object]:
    """Return one window contrast: mean change, interval, per-fold changes, and base error."""
    frame = tagged.filter(
        pl.col("substation_type") == substation_type, pl.col("window").is_in(windows)
    ).with_columns(difference=pl.col(treatment) - pl.col(reference))
    mean, low, high = block_interval(difference=frame, level=level)
    per_fold = frame.group_by("fold").agg(pl.col("difference").mean())
    folds = {r["fold"]: r["difference"] for r in per_fold.iter_rows(named=True)}
    return {
        "type": substation_type,
        "windows": " + ".join(windows),
        "treatment": treatment,
        "reference": reference,
        "days": frame.select(pl.struct("fold", "local_date").n_unique()).item(),
        "reference_error": float(frame[reference].to_numpy().mean()),
        "change": mean,
        "lower": low,
        "upper": high,
        **{f"change_{f}": folds.get(f) for f in fit.FOLDS},
        "same_sign": len({np.sign(v) for v in folds.values()}) == 1,
    }


def whole_year(*, rows: pl.DataFrame, substation_type: str, arms: list[str]) -> pl.DataFrame:
    """Return the whole-year contrasts of `both` against the other arms, month-resampled."""
    types = pl.read_parquet(fit.FEATURES_DIR / "series.parquet").select(
        "time_series_id", "substation_type"
    )
    keep = types.filter(pl.col("substation_type") == substation_type)["time_series_id"]
    chosen = rows.filter(pl.col("time_series_id").is_in(keep.implode())).with_columns(
        power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo")
    )
    frame = losses_for_bootstrap(
        arm_losses={
            arm: chosen.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            ).drop_nulls("abs_error")
            for arm in arms
        }
    )
    out = []
    for arm in arms:
        absolute = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
        row = {"type": substation_type, "arm": arm, "error": absolute["value"]}
        if arm != "base":
            low, high = bootstrap_difference_at_level(
                losses=frame,
                treatment=arm,
                reference="base",
                metric="abs_error",
                level=PLANNED_LEVEL,
                n_resamples=N_RESAMPLES,
            )
            diff = float(
                frame.filter(pl.col("arm") == arm)["abs_error"].to_numpy().mean()
                - frame.filter(pl.col("arm") == "base")["abs_error"].to_numpy().mean()
            )
            row |= {"change_vs_base": diff, "lower": low, "upper": high}
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


def main() -> None:
    """Write the predictions and print the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = with_climatology(rows=read_fits())
    save_predictions(rows=rows)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    print("rows per fold and seed:", rows.group_by("fold", "seed").len().sort("fold", "seed"))
    print("series per fold:", rows.group_by("fold").agg(n=pl.col("time_series_id").n_unique()))
    daily = daily_errors(rows=rows)
    tagged = tagged_days(daily=daily, windows=window_dates())
    specs: list[tuple[str, str, str, tuple[str, ...]]] = [
        (window.split(" (")[0], "both", "base", (window,))
        for window in PLANNED_WINDOWS + WINDOW_NAMES[3:]
    ]
    specs += [
        ("C6 vs placebo A", "both", "placebo_a", PLANNED_WINDOWS),
        ("C6 vs placebo B", "both", "placebo_b", PLANNED_WINDOWS),
        ("placebo A vs base", "placebo_a", "base", PLANNED_WINDOWS),
        ("placebo B vs base", "placebo_b", "base", PLANNED_WINDOWS),
        ("return vs base", "return", "base", PLANNED_WINDOWS),
        ("term vs base", "term", "base", PLANNED_WINDOWS),
        ("both vs one harmonic", "both", "one_harmonic", PLANNED_WINDOWS),
        ("both vs reference", "both", "reference", PLANNED_WINDOWS),
    ]
    out = []
    for substation_type in TYPES:
        for label, treatment, reference, windows in specs:
            result = contrast(
                tagged=tagged,
                substation_type=substation_type,
                treatment=treatment,
                reference=reference,
                windows=windows,
                level=PLANNED_LEVEL,
            )
            out.append({"label": label, **result})
        for window in PLANNED_WINDOWS:
            for arm in ("return", "term", "one_harmonic"):
                result = contrast(
                    tagged=tagged,
                    substation_type=substation_type,
                    treatment=arm,
                    reference="base",
                    windows=(window,),
                    level=PLANNED_LEVEL,
                )
                out.append({"label": f"{arm} vs base, {window.split(' (')[0]}", **result})
    table = pl.DataFrame(out, infer_schema_length=None)
    table.write_parquet(RESULTS_DIR / "window_contrasts.parquet")
    shown = table.select(
        "type",
        "label",
        "windows",
        "days",
        "reference_error",
        "change",
        "lower",
        "upper",
        *[f"change_{f}" for f in fit.FOLDS],
        "same_sign",
    ).with_columns(pl.col(pl.Float64).round(2))
    for substation_type in TYPES:
        print(substation_type)
        print(shown.filter(pl.col("type") == substation_type).drop("type"))
    for substation_type in TYPES:
        year = whole_year(
            rows=rows, substation_type=substation_type, arms=[*ARM_NAMES, CLIMATOLOGY]
        )
        year.write_parquet(RESULTS_DIR / f"whole_year_{substation_type}.parquet")
        print(year.with_columns(pl.col(pl.Float64).round(2)))
    daily.write_parquet(RESULTS_DIR / "daily_errors.parquet")


if __name__ == "__main__":
    main()
