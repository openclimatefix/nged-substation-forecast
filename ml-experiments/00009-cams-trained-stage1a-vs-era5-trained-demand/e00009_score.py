"""Assemble experiment 00009's arms, save their predictions, score them, and print the report.

Every arm is scored on the same cases: the validation runs' rows of the 20 series with an observed
power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted mean
pinball loss of the 9 levels, and the median error is the absolute error of the median, both in
points of capacity (percent of effective capacity). Intervals resample whole validation months and
the three fitting seeds. The two planned contrasts are read at 97.5% (Bonferroni across two) with
10,000 resamples; every other contrast is exploratory and read at 95%.

Arms: `era5-trained`, `cams-trained`, and the control `era5-hourly` (this experiment),
`climatology` (experiment 00005's `r2`). Experiment 00005's `r0_from2023` is read only to check
that the refit reproduces it, and is never scored.

Output: `work/arms/<arm>.parquet`, the prediction sets under
`data/ml-experiments/00009-.../predictions/`, `work/results/*.parquet`, and
`work/results/report.md`.

    uv run python ml-experiments/00009-cams-trained-stage1a-vs-era5-trained-demand/e00009_score.py
"""

from pathlib import Path
from typing import Final

import e00009_common as common
import e00009_fit as fit
import numpy as np
import polars as pl
from studies.solar import zenith

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
N_RUNS: Final[int] = 73
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TREATMENT: Final[str] = "cams-trained"
REFERENCE: Final[str] = "era5-trained"
CONTROL: Final[str] = "era5-hourly"
HEADLINE_GROUP: Final[str] = "Primary"
DOMINANT_LABEL: Final[str] = "P2"
"""The Primary substation that sets experiment 00007's pooled figure."""
PLANNED_LEVEL: Final[float] = 97.5
"""Bonferroni across the two planned contrasts."""
PLANNED_RESAMPLES: Final[int] = 10_000
DEFAULT_RESAMPLES: Final[int] = 2_000
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("3-6 d", 48.0, 144.0),
    ("7-10 d", 144.0, 240.0),
    ("11-15 d", 240.0, 10_000.0),
)
ARM_COLUMNS: Final[tuple[str, ...]] = (
    *KEYS,
    "seed",
    *common.LEVEL_COLUMNS,
    "actual",
    "group",
    "label",
    "daylight",
    "lead_hours",
    "quantile_score",
    "median_error",
)


def _stats() -> pl.DataFrame:
    """Return every ENS run's quantiles of every arm from `e00009_fit.py`.

    Returns:
        The 73 runs' rows with `arm`, `seed`, the 9 quantile columns, and `actual`.

    Raises:
        FileNotFoundError: If the fit has not written all 73 runs.
    """
    paths = sorted(fit.STATS_DIR.glob("init=*.parquet"))
    if len(paths) != N_RUNS:
        msg = f"expected {N_RUNS} runs in {fit.STATS_DIR}, found {len(paths)}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(path) for path in paths])


def _reference(*, name: str, actual: pl.DataFrame) -> pl.DataFrame:
    """Return an experiment 00005 prediction set with this experiment's `actual` attached."""
    frame = experiment_store.read_predictions(
        experiment_id=5, name=name, row_set_id=common.ROW_SET_ID
    )
    return frame.join(actual, on=KEYS, how="inner", validate="m:1")


def reproduction_check(*, ours: pl.DataFrame, theirs: pl.DataFrame) -> pl.DataFrame:
    """Return how far the refitted ERA5-trained arm lies from 00005's `r0_from2023`.

    Args:
        ours: The `era5` arm's rows with the 9 quantile columns.
        theirs: Experiment 00005's `r0_from2023` set.

    Returns:
        One row per quantile level with the mean and maximum absolute difference, over the rows
        both hold.
    """
    joined = ours.join(theirs, on=[*KEYS, "seed"], how="inner", suffix="_theirs")
    return pl.DataFrame(
        {
            "level": list(common.LEVEL_COLUMNS),
            "mean_abs_difference": [
                joined.select((pl.col(c) - pl.col(f"{c}_theirs")).abs().mean()).item()
                for c in common.LEVEL_COLUMNS
            ],
            "max_abs_difference": [
                joined.select((pl.col(c) - pl.col(f"{c}_theirs")).abs().max()).item()
                for c in common.LEVEL_COLUMNS
            ],
            "rows": joined.height,
        }
    )


def _daylight(*, frame: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return each series' valid times with whether the sun is up at the half-hour's midpoint."""
    pieces = []
    for row in series.iter_rows(named=True):
        times = (
            frame.filter(pl.col("time_series_id") == row["time_series_id"])["valid_time"]
            .unique()
            .sort()
        )
        angle = zenith(
            stamps=times.dt.offset_by("-15m"), latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(
            pl.DataFrame(
                {
                    "time_series_id": np.full(times.len(), row["time_series_id"], dtype=np.int32),
                    "valid_time": times,
                    "daylight": np.asarray(angle) < common.MAX_ZENITH_DEGREES,
                }
            )
        )
    return pl.concat(pieces)


def prepare(*, arms: dict[str, pl.DataFrame], series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, group, and daylight flag.

    Args:
        arms: Each arm's rows keyed by series, run, valid time, and seed, with the 9 quantile
            columns and `actual`.
        series: `common.series_table()`.

    Returns:
        The same arms with the columns of `ARM_COLUMNS`.
    """
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < common.HOLDOUT)
            )
            .select(KEYS)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=KEYS, how="inner")
    assert shared is not None
    groups = series.select(
        "time_series_id", group=pl.col("substation_type").cast(pl.String), label=pl.col("label")
    )
    daylight = _daylight(frame=shared, series=series)
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        score = common.pinball_quantile_score(quantiles=quantiles, actual=actual)
        out[name] = (
            kept.join(groups, on="time_series_id", how="left", validate="m:1")
            .join(daylight, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
                quantile_score=pl.Series(score * 100.0),
                median_error=pl.Series(np.abs(quantiles[:, 4] - actual) * 100.0),
            )
            .select(ARM_COLUMNS)
        )
    return out


def save_predictions(*, arms: dict[str, pl.DataFrame]) -> None:
    """Write this experiment's three arms as immutable prediction sets, skipping existing sets."""
    for name in (REFERENCE, TREATMENT, CONTROL):
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=arms[name].select(KEYS[0], KEYS[1], KEYS[2], "seed", *common.LEVEL_COLUMNS),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue


def _scope(
    *,
    frame: pl.DataFrame,
    band: str = "all leads",
    group: str = HEADLINE_GROUP,
    exclude_label: str | None = None,
    only_label: str | None = None,
    daylight: bool | None = None,
) -> pl.DataFrame:
    low, high = next((lo, hi) for name, lo, hi in LEAD_BANDS if name == band)
    frame = frame.filter(
        (pl.col("group") == group) & (pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high)
    )
    if exclude_label is not None:
        frame = frame.filter(pl.col("label") != exclude_label)
    if only_label is not None:
        frame = frame.filter(pl.col("label") == only_label)
    if daylight is not None:
        frame = frame.filter(pl.col("daylight") == daylight)
    return frame


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    level: float = 95.0,
    n_resamples: int = DEFAULT_RESAMPLES,
    band: str = "all leads",
    group: str = HEADLINE_GROUP,
    exclude_label: str | None = None,
    only_label: str | None = None,
    daylight: bool | None = None,
) -> dict[str, float | int]:
    """Return one paired contrast's estimate, interval, and month and series agreement.

    Args:
        arms: `prepare()`'s arms.
        treatment: The arm whose quantile score is compared.
        reference: The arm it is compared against.
        level: The interval's coverage in percent.
        n_resamples: How many resamples to draw.
        band: A lead band of `LEAD_BANDS`.
        group: A substation type.
        exclude_label: A series label to leave out.
        only_label: A series label to keep alone.
        daylight: Keep only daylight (True) or night (False) valid times.

    Returns:
        The mean paired difference in quantile score, the interval bounds, the months and the
        series with a negative difference, and the row count.
    """
    scope = {
        "band": band,
        "group": group,
        "exclude_label": exclude_label,
        "only_label": only_label,
        "daylight": daylight,
    }
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            treatment: _scope(frame=arms[treatment], **scope),
            reference: _scope(frame=arms[reference], **scope),
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="quantile_score",
        level=level,
        n_resamples=n_resamples,
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", treated=pl.col("quantile_score"))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", base=pl.col("quantile_score")
            ),
            on=["site", "time", "seed"],
        )
        .with_columns(difference=pl.col("treated") - pl.col("base"))
    )
    by_month = paired.group_by("month").agg(pl.col("difference").mean())
    by_series = paired.group_by("site").agg(pl.col("difference").mean())
    return {
        "difference": paired.select(pl.col("difference").mean()).item(),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
        "series_negative": int((by_series["difference"] < 0).sum()),
        "series": by_series.height,
        "rows": paired.height,
    }


def planned_contrasts(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the two planned contrasts, C1 and C2, at 97.5% with 10,000 resamples."""
    rows = []
    for label, excluded in (
        ("C1: cams-trained minus era5-trained, Primary", None),
        (f"C2: C1 without {DOMINANT_LABEL}", DOMINANT_LABEL),
    ):
        rows.append(
            {
                "label": label,
                **contrast(
                    arms=arms,
                    treatment=TREATMENT,
                    reference=REFERENCE,
                    level=PLANNED_LEVEL,
                    n_resamples=PLANNED_RESAMPLES,
                    exclude_label=excluded,
                ),
            }
        )
    return pl.DataFrame(rows)


def exploratory_contrasts(*, arms: dict[str, pl.DataFrame], labels: list[str]) -> pl.DataFrame:
    """Return the exploratory contrasts, read at 95%."""
    rows = []

    def add(
        *,
        label: str,
        treatment: str,
        reference: str,
        band: str = "all leads",
        group: str = HEADLINE_GROUP,
        exclude_label: str | None = None,
        only_label: str | None = None,
        daylight: bool | None = None,
    ) -> None:
        rows.append(
            {
                "label": label,
                **contrast(
                    arms=arms,
                    treatment=treatment,
                    reference=reference,
                    band=band,
                    group=group,
                    exclude_label=exclude_label,
                    only_label=only_label,
                    daylight=daylight,
                ),
            }
        )

    for band, _, _ in LEAD_BANDS[1:]:
        add(label=f"C1 by lead band: {band}", treatment=TREATMENT, reference=REFERENCE, band=band)
    add(label="C1, daylight valid times", treatment=TREATMENT, reference=REFERENCE, daylight=True)
    add(label="C1, night valid times", treatment=TREATMENT, reference=REFERENCE, daylight=False)
    for group in ("BSP", "GSP"):
        add(label=f"C1, {group}", treatment=TREATMENT, reference=REFERENCE, group=group)
    for label in labels:
        add(label=f"C1, {label} only", treatment=TREATMENT, reference=REFERENCE, only_label=label)
        add(
            label=f"C1 without {label}",
            treatment=TREATMENT,
            reference=REFERENCE,
            exclude_label=label,
        )
    add(label="X1: era5-hourly minus era5-trained", treatment=CONTROL, reference=REFERENCE)
    add(label="X2: cams-trained minus era5-hourly", treatment=TREATMENT, reference=CONTROL)
    add(label="era5-trained minus climatology", treatment=REFERENCE, reference="climatology")
    add(label="cams-trained minus climatology", treatment=TREATMENT, reference="climatology")
    return pl.DataFrame(rows)


def absolute_table(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return every scored arm's mean quantile score and median error, with 95% intervals."""
    rows = []
    for name in (REFERENCE, TREATMENT, CONTROL, "climatology"):
        for group in GROUPS:
            for band, _, _ in LEAD_BANDS:
                part = experiment_store.losses_for_bootstrap(
                    arm_losses={name: _scope(frame=arms[name], band=band, group=group)}
                )
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=part, arm=name, metric=metric)
                    rows.append(
                        {"arm": name, "group": group, "scope": band, "metric": metric, **interval}
                    )
    return pl.DataFrame(rows)


def observed_weather_diagnostic(*, series: pl.DataFrame) -> pl.DataFrame:
    """Compare the arms at the observed weather in the validation year.

    The `era5` arm reads ERA5 shortwave and the `cams` arm reads CAMS, at every daylight half-hour
    with an observed power. Each prediction is the mean of the three seeds.

    Args:
        series: `common.series_table()`.

    Returns:
        One row per substation type with the daylight mean absolute error of each arm (points of
        capacity), their difference, and its 95% month-resampled interval.
    """
    frames = {
        arm: pl.read_parquet(fit.OBSERVED_DIR / f"{arm}.parquet")
        .filter(pl.col("actual").is_finite())
        .group_by("time_series_id", "valid_time")
        .agg(pl.col("prediction").mean(), pl.col("actual").first())
        for arm in ("era5", "cams")
    }
    joined = frames["era5"].join(
        frames["cams"], on=["time_series_id", "valid_time"], suffix="_cams", validate="1:1"
    )
    daylight = _daylight(frame=joined, series=series)
    groups = series.select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    rows = (
        joined.join(daylight, on=["time_series_id", "valid_time"], validate="1:1")
        .join(groups, on="time_series_id", validate="m:1")
        .filter(pl.col("daylight"))
        .with_columns(
            era5_error=(pl.col("prediction") - pl.col("actual")).abs() * 100.0,
            cams_error=(pl.col("prediction_cams") - pl.col("actual")).abs() * 100.0,
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
        )
    )
    out = []
    for group in GROUPS:
        part = rows.filter(pl.col("group") == group)
        interval = bootstrap.bootstrap_row_difference(
            values=(part["cams_error"] - part["era5_error"]).to_numpy(),
            months=part["month"].to_numpy(),
        )
        out.append(
            {
                "group": group,
                "era5_mae": part.select(pl.col("era5_error").mean()).item(),
                "cams_mae": part.select(pl.col("cams_error").mean()).item(),
                **interval,
            }
        )
    return pl.DataFrame(out)


def shortwave_by_month(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the mean daylight shortwave of CAMS, ERA5, and the ENS members, by period and month.

    The training period is 2023-09 to before 2025-06-17 and compares CAMS with ERA5. The validation
    period is 2025-07 to 2026-06 and adds the mean of the ENS members over the 73 runs, all leads.
    Means are over the 16 Primary series.

    Args:
        series: `common.series_table()`.

    Returns:
        One row per period, source, and calendar month with the mean in W/m2.
    """
    primary = series.filter(pl.col("substation_type") == HEADLINE_GROUP)
    data = fit.load_series_data(series=primary)
    columns = common.feature_columns()
    shortwave_at, zenith_at = columns.index("shortwave"), columns.index("solar_zenith")
    first = int(common.grid_index(times=common.VALIDATION_START))
    stop = int(common.grid_index(times=common.HOLDOUT))
    pieces = []
    for row in primary.iter_rows(named=True):
        arrays = data[row["site"]]
        training = fit.training_rows(data=arrays)
        validation = np.arange(first, stop)
        for period, rows in (("training", training), ("validation", validation)):
            keep = rows[arrays.era5_features[rows, zenith_at] < common.MAX_ZENITH_DEGREES]
            times = arrays.valid_time.gather(keep.tolist())
            for source, matrix in (("ERA5", arrays.era5_features), ("CAMS", arrays.cams_features)):
                values = matrix[keep, shortwave_at].astype(np.float64)
                pieces.append(
                    pl.DataFrame(
                        {
                            "period": period,
                            "source": source,
                            "month": times.dt.month(),
                            "value": values,
                        }
                    ).filter(pl.col("value").is_finite())
                )
        ens = pl.concat(
            [
                pl.read_parquet(path, columns=["time_series_id", "valid_time", "shortwave"]).filter(
                    pl.col("time_series_id") == row["time_series_id"]
                )
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
            ]
        ).filter(pl.col("valid_time") < common.HOLDOUT)
        sun = _daylight(
            frame=ens, series=series.filter(pl.col("time_series_id") == row["time_series_id"])
        )
        ens = ens.join(sun, on=["time_series_id", "valid_time"], validate="m:1").filter(
            pl.col("daylight")
        )
        pieces.append(
            ens.select(
                period=pl.lit("validation"),
                source=pl.lit("ENS members"),
                month=pl.col("valid_time").dt.month(),
                value=pl.col("shortwave").cast(pl.Float64),
            )
        )
    return (
        pl.concat(pieces)
        .group_by("period", "source", "month")
        .agg(pl.col("value").mean())
        .sort("period", "source", "month")
    )


def verdict(*, planned: pl.DataFrame) -> str:
    """Return the pre-registered verdict from the planned contrasts C1 and C2.

    Args:
        planned: `planned_contrasts()`.

    Returns:
        `supported`, `partly supported`, `refuted`, or `inconclusive`, with the numbers it rests on.
    """
    c1, c2 = planned.row(0, named=True), planned.row(1, named=True)
    c1_ok = c1["difference"] <= -0.2 and c1["upper"] < 0
    c2_ok = c2["difference"] <= -0.1 and c2["upper"] < 0
    if c1_ok and c2_ok:
        label = "supported"
    elif c1_ok:
        label = "partly supported"
    elif c1["lower"] >= -0.2:
        label = "refuted"
    else:
        label = "inconclusive"
    return (
        f"{label}: C1 is {c1['difference']:+.3f} ({c1['lower']:+.3f} to {c1['upper']:+.3f} at "
        f"97.5%), C2 is {c2['difference']:+.3f} ({c2['lower']:+.3f} to {c2['upper']:+.3f})"
    )


def _table(*, frame: pl.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.iter_rows(named=True):
        cells = [f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    """Score every arm and write the report."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    stats = _stats()
    actual = stats.select(*KEYS, "actual").unique()
    ours = {
        name: stats.filter(pl.col("arm") == arm).drop("arm")
        for name, arm in ((REFERENCE, "era5"), (TREATMENT, "cams"), (CONTROL, "era5-hourly"))
    }
    check = reproduction_check(
        ours=ours[REFERENCE],
        theirs=experiment_store.read_predictions(
            experiment_id=5, name="r0_from2023", row_set_id=common.ROW_SET_ID
        ),
    )
    arms = prepare(
        arms={**ours, "climatology": _reference(name="r2", actual=actual)}, series=series
    )
    for name, frame in arms.items():
        frame.write_parquet(ARMS_DIR / f"{name}.parquet")
    save_predictions(arms=arms)
    primary_labels = sorted(
        series.filter(pl.col("substation_type") == HEADLINE_GROUP)["label"].to_list(),
        key=lambda label: int(label[1:]),
    )
    planned = planned_contrasts(arms=arms)
    exploratory = exploratory_contrasts(arms=arms, labels=primary_labels)
    absolute = absolute_table(arms=arms)
    observed = observed_weather_diagnostic(series=series)
    shortwave = shortwave_by_month(series=series)
    for name, frame in (
        ("planned", planned),
        ("exploratory", exploratory),
        ("absolute", absolute),
        ("observed_weather", observed),
        ("shortwave_by_month", shortwave),
        ("reproduction", check),
    ):
        frame.write_parquet(RESULTS_DIR / f"{name}.parquet")
    lines = [
        "# Experiment 00009: scores",
        "",
        f"Verdict by the pre-registered rule: {verdict(planned=planned)}",
        "",
        "## Reproduction of 00005's r0_from2023 (points are fractions of capacity)",
        *_table(
            frame=check, columns=["level", "mean_abs_difference", "max_abs_difference", "rows"]
        ),
        "",
        "## Planned contrasts (97.5% intervals; negative favours the CAMS-trained arm)",
        *_table(
            frame=planned,
            columns=[
                "label",
                "difference",
                "lower",
                "upper",
                "months_negative",
                "months",
                "series_negative",
                "series",
            ],
        ),
        "",
        "## Exploratory contrasts, 95% intervals",
        *_table(
            frame=exploratory,
            columns=[
                "label",
                "difference",
                "lower",
                "upper",
                "months_negative",
                "months",
                "series_negative",
                "series",
            ],
        ),
        "",
        "## Absolute scores (points of capacity; 95% month-and-seed intervals)",
        *_table(
            frame=absolute.filter(pl.col("metric") == "quantile_score"),
            columns=["arm", "group", "scope", "value", "lower_95", "upper_95"],
        ),
        "",
        "## Median error, absolute",
        *_table(
            frame=absolute.filter(pl.col("metric") == "median_error"),
            columns=["arm", "group", "scope", "value", "lower_95", "upper_95"],
        ),
        "",
        "## Observed-weather diagnostic: daylight half-hours, validation year, mean absolute error",
        *_table(
            frame=observed,
            columns=[
                "group",
                "era5_mae",
                "cams_mae",
                "difference",
                "lower_95",
                "upper_95",
                "n_rows",
                "n_months",
            ],
        ),
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
