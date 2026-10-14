"""Score the demand experiment's arms and write the report (exploratory, apart from D1 to D3).

Every arm is scored on the same cases: the control-member rows of the validation runs with an
observed power and a valid time before 2026-07-01. The point forecast is the mean of the 51
members' predictions. The error is the absolute error over effective capacity, in points of
capacity. The top decile is the cases where observed power over capacity exceeds the 90th
percentile of the series' cleaned power from 2019-09 to before 2025-06-17.

Two climatology arms need no fit: `climatology` (mean by local month and half-hour of the day) and
`climatology_dow` (the same, separately for weekdays, Saturdays, and Sundays).

Intervals resample whole validation months and the three fitting seeds, the same draw for every
arm (`stage1_stats`). Output under `data/studies/per_study/stage1_pilot/demand/results/`:
`report.md`, `levels.parquet`, `contrasts.parquet`.

    uv run python studies/stage1_pilot/demand_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import demand_fit as fit
import demand_inputs as demand
import numpy as np
import polars as pl
import stage1_inputs as inputs
import stage1_stats as stats

RESULT_DIR: Final[Path] = demand.demand_subfolder(name="results")
VAL_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
BANDS: Final[tuple[str, ...]] = ("all", *(name for name, _, _ in inputs.LEAD_BANDS))
GROUPS: Final[tuple[str, ...]] = ("all", "primary demand", "grid flow", "battery or biofuel")
TAILS: Final[tuple[str, ...]] = ("all cases", "top decile")

MODEL_ARMS: Final[tuple[str, ...]] = tuple(
    fit.column_name(arm=arm, variant=variant) for variant, arms in fit.ARMS.items() for arm in arms
)
ALL_ARMS: Final[tuple[str, ...]] = (*MODEL_ARMS, "climatology", "climatology_dow")

PLANNED: Final[dict[str, tuple[str, str]]] = {
    "D1": ("s1_memberavg__V0", "ens_trained__V0"),
    "D2": ("s1_memberavg__V1", "ens_trained_dedup__V1"),
    "D3": ("s1_memberavg__V2", "ens_trained_dedup__V2"),
}
"""The planned contrasts: treatment, reference."""

EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: ERA5-trained minus ENS-trained, V1": ("s1_memberavg__V1", "ens_trained__V1"),
    "E2: ERA5-trained minus ENS-trained, V2": ("s1_memberavg__V2", "ens_trained__V2"),
    "E3: ENS-trained, one row per valid time, minus all rows, V1": (
        "ens_trained_dedup__V1",
        "ens_trained__V1",
    ),
    "E4: ENS-trained, one row per valid time, minus all rows, V2": (
        "ens_trained_dedup__V2",
        "ens_trained__V2",
    ),
    "E5: short ERA5 history minus long, V0": ("s1_short__V0", "s1_memberavg__V0"),
    "E6: short ERA5 history minus long, V1": ("s1_short__V1", "s1_memberavg__V1"),
    "E7: short ERA5 history minus long, V2": ("s1_short__V2", "s1_memberavg__V2"),
    "E8: recency weights minus equal weights, V0": ("s1_recency__V0", "s1_memberavg__V0"),
    "E9: recency weights minus equal weights, V1": ("s1_recency__V1", "s1_memberavg__V1"),
    "E10: without 2020-03 to 2021-07 minus all years, V0": ("s1_nocovid__V0", "s1_memberavg__V0"),
    "E11: without 2020-03 to 2021-07 minus all years, V1": ("s1_nocovid__V1", "s1_memberavg__V1"),
    "E12: calendar added, ERA5-trained (V1 minus V0)": ("s1_memberavg__V1", "s1_memberavg__V0"),
    "E13: lags added, ERA5-trained (V2 minus V1)": ("s1_memberavg__V2", "s1_memberavg__V1"),
    "E14: calendar added, ENS-trained, one row per valid time (V1 minus V0)": (
        "ens_trained_dedup__V1",
        "ens_trained__V0",
    ),
    "E15: lags added, ENS-trained, one row per valid time (V2 minus V1)": (
        "ens_trained_dedup__V2",
        "ens_trained_dedup__V1",
    ),
    "E16: ERA5-trained V0 minus the climatology": ("s1_memberavg__V0", "climatology"),
    "E17: ERA5-trained V1 minus the climatology by day type": (
        "s1_memberavg__V1",
        "climatology_dow",
    ),
    "E18: ENS-trained V1 (one row per valid time) minus the climatology by day type": (
        "ens_trained_dedup__V1",
        "climatology_dow",
    ),
    "E19: ERA5-trained V2 minus the climatology by day type": (
        "s1_memberavg__V2",
        "climatology_dow",
    ),
}


def _band_expression() -> pl.Expr:
    band = pl.lit("none")
    for name, low, high in reversed(inputs.LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return band


def _climate_key(*, daytype: bool) -> list[pl.Expr]:
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    keys = [
        local.dt.month().alias("month"),
        (local.dt.hour() * 2 + local.dt.minute() // 30).alias("slot"),
    ]
    if daytype:
        keys.append(
            pl.when(local.dt.weekday() <= 5)
            .then(pl.lit("weekday"))
            .when(local.dt.weekday() == 6)
            .then(pl.lit("saturday"))
            .otherwise(pl.lit("sunday"))
            .alias("daytype")
        )
    return keys


def training_power(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power over effective capacity from 2019-09 to before the stage 1a cutoff."""
    return (
        pl.read_parquet(demand.INPUTS_DIR / "observed.parquet")
        .filter(pl.col("valid_time") < inputs.STAGE1_TRAIN_END)
        .join(series.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def scored_cases(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every arm's absolute error (points of capacity) per case and seed, with scope flags.

    Args:
        series: `demand.series_table()`.

    Returns:
        One row per seed and case with `abs_error__<arm>` columns for every arm, `band`, `group`,
        `top`, `month`, and `time_series_id`.
    """
    observed = training_power(series=series)
    threshold = observed.group_by("time_series_id").agg(threshold=pl.col("target").quantile(0.9))
    climates = {}
    for name, daytype in (("climatology", False), ("climatology_dow", True)):
        keys = [e.meta.output_name() for e in _climate_key(daytype=daytype)]
        climates[name] = (
            observed.with_columns(*_climate_key(daytype=daytype))
            .group_by("time_series_id", *keys)
            .agg(**{name: pl.col("target").mean()})
        )
    pieces = []
    for seed in inputs.SEEDS:
        files = sorted((fit.PRED_DIR / f"seed{seed}").glob("init=*.parquet"))
        frame = (
            pl.concat([pl.read_parquet(p) for p in files])
            .filter(pl.col("actual").is_finite() & (pl.col("valid_time") < VAL_END))
            .join(
                series.select("time_series_id", "effective_capacity_mw", "group"),
                on="time_series_id",
            )
            .join(threshold, on="time_series_id")
        )
        for name, daytype in (("climatology", False), ("climatology_dow", True)):
            keys = [e.meta.output_name() for e in _climate_key(daytype=daytype)]
            frame = frame.with_columns(*_climate_key(daytype=daytype)).join(
                climates[name], on=["time_series_id", *keys], how="left", validate="m:1"
            )
            frame = frame.with_columns(
                (pl.col(name) * pl.col("effective_capacity_mw")).alias(name)
            ).drop(*keys)
        frame = frame.with_columns(
            *[
                (
                    100.0 * (pl.col(arm) - pl.col("actual")).abs() / pl.col("effective_capacity_mw")
                ).alias(f"abs_error__{arm}")
                for arm in ALL_ARMS
            ],
            band=_band_expression(),
            top=(pl.col("actual") / pl.col("effective_capacity_mw")) > pl.col("threshold"),
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
            seed=pl.lit(seed),
        )
        if frame.select(
            pl.any_horizontal(pl.col(f"abs_error__{a}").is_null() for a in ALL_ARMS).any()
        ).item():
            msg = "a case has no forecast from some arm"
            raise ValueError(msg)
        pieces.append(
            frame.select(
                "seed",
                "time_series_id",
                "group",
                "band",
                "top",
                "month",
                *[f"abs_error__{a}" for a in ALL_ARMS],
            )
        )
        print(f"seed {seed}: {frame.height:,} cases", flush=True)
    return pl.concat(pieces)


def aggregate(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Sum each arm's error and count cases by seed, series, group, month, band, and tail flag."""
    return cases.group_by("seed", "time_series_id", "group", "month", "band", "top").agg(
        n=pl.len(), **{a: pl.col(f"abs_error__{a}").sum() for a in ALL_ARMS}
    )


def monthly_for(
    *, agg: pl.DataFrame, arm: str, band: str, group: str, tail: str, series: list[int]
) -> stats.Monthly:
    """Return one arm's totals by seed, series, and month within a scope."""
    part = agg
    if band != "all":
        part = part.filter(pl.col("band") == band)
    if group != "all":
        part = part.filter(pl.col("group") == group)
    if tail == "top decile":
        part = part.filter(pl.col("top"))
    grouped = part.group_by("seed", "time_series_id", "month").agg(
        total=pl.col(arm).sum(), count=pl.col("n").sum()
    )
    months = tuple(sorted(agg["month"].unique().to_list()))
    seeds = sorted(agg["seed"].unique().to_list())
    in_scope = (
        series
        if group == "all"
        else [
            int(i)
            for i in agg.filter(pl.col("group") == group)["time_series_id"].unique().to_list()
        ]
    )
    totals = np.zeros((len(seeds), len(in_scope), len(months)))
    counts = np.zeros((len(seeds), len(in_scope), len(months)))
    for row in grouped.iter_rows(named=True):
        counts_index = (
            seeds.index(row["seed"]),
            in_scope.index(int(row["time_series_id"])),
            months.index(row["month"]),
        )
        totals[counts_index] = row["total"]
        counts[counts_index] = row["count"]
    return stats.Monthly(totals=totals, counts=counts[0], months=months, series=tuple(in_scope))


def main() -> None:
    """Score every arm, compute the levels and contrasts, and write the report."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    series = demand.series_table()
    ids = sorted(int(i) for i in series["time_series_id"])
    agg = aggregate(cases=scored_cases(series=series))
    agg.write_parquet(RESULT_DIR / "aggregate.parquet")
    level_rows = []
    for arm in ALL_ARMS:
        for band in BANDS:
            for group in GROUPS:
                for tail in TAILS:
                    table = monthly_for(
                        agg=agg, arm=arm, band=band, group=group, tail=tail, series=ids
                    )
                    interval = stats.level(table=table)
                    level_rows.append(
                        {
                            "arm": arm,
                            "band": band,
                            "group": group,
                            "tail": tail,
                            "value": interval.value,
                            "lower": interval.lower,
                            "upper": interval.upper,
                        }
                    )
    levels = pl.DataFrame(level_rows)
    levels.write_parquet(RESULT_DIR / "levels.parquet")
    contrast_rows = []
    for planned, mapping in ((True, PLANNED), (False, EXPLORATORY)):
        for label, (treatment, reference) in mapping.items():
            for band in BANDS:
                for group in GROUPS:
                    for tail in TAILS:
                        result = stats.contrast(
                            treatment=monthly_for(
                                agg=agg,
                                arm=treatment,
                                band=band,
                                group=group,
                                tail=tail,
                                series=ids,
                            ),
                            reference=monthly_for(
                                agg=agg,
                                arm=reference,
                                band=band,
                                group=group,
                                tail=tail,
                                series=ids,
                            ),
                        )
                        contrast_rows.append(
                            {
                                "contrast": label,
                                "planned": planned,
                                "band": band,
                                "group": group,
                                "tail": tail,
                                "difference": result.difference,
                                "lower": result.lower,
                                "upper": result.upper,
                                "months_better": result.months_better,
                                "material": result.material,
                            }
                        )
    contrasts = pl.DataFrame(contrast_rows)
    contrasts.write_parquet(RESULT_DIR / "contrasts.parquet")
    write_report(levels=levels, contrasts=contrasts, series=series)


def _fmt(*, row: dict) -> str:
    return f"{row['value']:.2f} ({row['lower']:.2f} to {row['upper']:.2f})"


def write_report(*, levels: pl.DataFrame, contrasts: pl.DataFrame, series: pl.DataFrame) -> None:
    """Print every table the write-up quotes."""
    lines = [
        "# Demand experiment (00003) results",
        "",
        (
            "Mean absolute error of the ensemble-mean forecast, points of capacity "
            "(smaller is better)."
        ),
        "",
        "## Series",
        "",
        "| group | series |",
        "| --- | --- |",
    ]
    for group, part in series.group_by("group", maintain_order=True):
        labels = ", ".join(sorted(part["site"].to_list(), key=lambda s: int(s[1:])))
        lines.append(f"| {group[0]} | {labels} |")
    lines += ["", "## Columns of each variant", ""]
    for variant, columns in demand.VARIANTS.items():
        lines.append(
            f"- **{variant}** ({len(columns)} columns): " + ", ".join(f"`{c}`" for c in columns)
        )
    lines += [""]
    for group in GROUPS:
        for tail in TAILS:
            lines += [f"## Mean absolute error: {group}, {tail}", ""]
            lines += ["| arm | " + " | ".join(BANDS) + " |", "| --- |" + " --- |" * len(BANDS)]
            for arm in ALL_ARMS:
                cells = []
                for band in BANDS:
                    row = levels.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("band") == band)
                        & (pl.col("group") == group)
                        & (pl.col("tail") == tail)
                    ).row(0, named=True)
                    cells.append(_fmt(row=row))
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += [
        "## Paired contrasts",
        "",
        (
            "Treatment minus reference, points of capacity (negative: the treatment has the "
            "smaller error). `months` counts the validation months with a negative difference. "
            "D1 to D3 are planned; every other row is exploratory. A planned contrast is material "
            "when the mean difference is at most -0.15, the 95% interval is wholly below zero, "
            "and at least 9 of 12 months are negative."
        ),
        "",
    ]
    for group in GROUPS:
        for tail in TAILS:
            lines += [f"### {group}, {tail}", ""]
            lines += [
                "| contrast | band | difference | 95% interval | months | material |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            part = contrasts.filter((pl.col("group") == group) & (pl.col("tail") == tail))
            for row in part.iter_rows(named=True):
                mark = ("yes" if row["material"] else "no") if row["planned"] else "-"
                lines.append(
                    f"| {row['contrast']} | {row['band']} | {row['difference']:+.3f} | "
                    f"{row['lower']:+.3f} to {row['upper']:+.3f} | "
                    f"{row['months_better']} of 12 | {mark} |"
                )
            lines.append("")
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
