"""Assemble experiment 00014's arms, save their predictions, score them, and print the report.

Every arm is scored on the same cases: the validation runs' rows of the 20 series with an observed
power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted mean
pinball loss of the 9 levels, and the median error is the absolute error of the median, both in
points of capacity (percent of effective capacity). Intervals resample whole validation months and
the three fitting seeds. The two planned contrasts are read at 97.5% (Bonferroni across two) with
10,000 resamples; every other contrast is exploratory and read at 95%.

Arms: `era5-trained` and `cams-trained` (experiment 00009's stage 1a models fed the published ENS
shortwave), `cams-stage0` (the CAMS-trained models fed ENS shortwave mapped onto CAMS),
`era5-stage0` (the ERA5-trained models fed ENS shortwave mapped onto ERA5), and `climatology`
(experiment 00005's `r2`). The raw arms are also compared with experiment 00009's stored
quantiles, to measure the noise of a repeated prediction.

Output: `work/arms/<arm>.parquet`, the prediction sets under
`data/ml-experiments/00014-.../predictions/`, `work/results/*.parquet`, and
`work/results/report.md`.

    uv run python ml-experiments/00014-*/e00014_score.py
"""

from pathlib import Path
from typing import Final

import e00014_common as common
import e00014_fit as fit
import numpy as np
import polars as pl
from studies.solar import zenith

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
N_RUNS: Final[int] = 73
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
ERA5_RAW: Final[str] = "era5-trained"
CAMS_RAW: Final[str] = "cams-trained"
CAMS_STAGE0: Final[str] = "cams-stage0"
ERA5_STAGE0: Final[str] = "era5-stage0"
STAGE_ARMS: Final[dict[str, str]] = {
    "era5-raw": ERA5_RAW,
    "cams-raw": CAMS_RAW,
    "cams-stage0": CAMS_STAGE0,
    "era5-stage0": ERA5_STAGE0,
}
HEADLINE_GROUP: Final[str] = "Primary"
PLANNED_LEVEL: Final[float] = 97.5
"""Bonferroni across the two planned contrasts."""
PLANNED_RESAMPLES: Final[int] = 10_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SUPPORT_THRESHOLD: Final[float] = -0.03
"""C2 must be at most this many points of capacity, with its interval below zero, for support."""
HOLDOUT_MONTH: Final[str] = "2026-07"
"""Valid times from this month on are never scored, so the mismatch table leaves them out."""
C1_IN_00009: Final[float] = -0.035
C1_NOISE: Final[float] = 0.005
BIAS_YEAR_LIMIT: Final[float] = 0.02
BIAS_MONTH_LIMIT: Final[float] = 0.05
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
    """Return every ENS run's quantiles of every arm and input from `e00014_fit.py`.

    Raises:
        FileNotFoundError: If the fit has not written all 73 runs.
    """
    paths = sorted(fit.STATS_DIR.glob("init=*.parquet"))
    if len(paths) != N_RUNS:
        msg = f"expected {N_RUNS} runs in {fit.STATS_DIR}, found {len(paths)}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(path) for path in paths])


def noise_check(*, ours: pl.DataFrame) -> pl.DataFrame:
    """Return how far the repeated raw-input predictions lie from experiment 00009's stored ones.

    Args:
        ours: The `cams-raw` and `era5-raw` arms' rows with the 9 quantile columns and `arm`.

    Returns:
        One row per arm and quantile level with the mean and maximum absolute difference.
    """
    rows = []
    for arm, old in (("era5-raw", "era5"), ("cams-raw", "cams")):
        paths = sorted((common.STAGE1_WORK_DIR / "stats").glob("init=*.parquet"))
        theirs = pl.concat([pl.read_parquet(p) for p in paths]).filter(pl.col("arm") == old)
        joined = ours.filter(pl.col("arm") == arm).join(
            theirs, on=[*KEYS, "seed"], how="inner", suffix="_theirs"
        )
        for level in common.LEVEL_COLUMNS:
            difference = (pl.col(level) - pl.col(f"{level}_theirs")).abs()
            rows.append(
                {
                    "arm": arm,
                    "level": level,
                    "mean_abs_difference": joined.select(difference.mean()).item(),
                    "max_abs_difference": joined.select(difference.max()).item(),
                    "rows": joined.height,
                }
            )
    return pl.DataFrame(rows)


def _reference(*, name: str, actual: pl.DataFrame) -> pl.DataFrame:
    """Return an experiment 00005 prediction set with this experiment's `actual` attached."""
    frame = experiment_store.read_predictions(
        experiment_id=5, name=name, row_set_id=common.ROW_SET_ID
    )
    return frame.join(actual, on=KEYS, how="inner", validate="m:1")


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
    """Write the two stage 0 arms as immutable prediction sets, skipping existing sets."""
    for name in (CAMS_STAGE0, ERA5_STAGE0):
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=arms[name].select(*KEYS, "seed", *common.LEVEL_COLUMNS),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue


def _scope(
    *,
    frame: pl.DataFrame,
    band: str = "all leads",
    group: str | tuple[str, ...] = HEADLINE_GROUP,
    exclude_label: str | None = None,
    only_label: str | None = None,
    daylight: bool | None = None,
) -> pl.DataFrame:
    low, high = next((lo, hi) for name, lo, hi in LEAD_BANDS if name == band)
    frame = frame.filter(
        pl.col("group").is_in([group] if isinstance(group, str) else list(group))
        & (pl.col("lead_hours") >= low)
        & (pl.col("lead_hours") < high)
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
    group: str | tuple[str, ...] = HEADLINE_GROUP,
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
    for label, treatment, reference in (
        ("C1: cams-trained minus era5-trained, Primary", CAMS_RAW, ERA5_RAW),
        ("C2: cams-stage0 minus cams-trained, Primary", CAMS_STAGE0, CAMS_RAW),
    ):
        rows.append(
            {
                "label": label,
                **contrast(
                    arms=arms,
                    treatment=treatment,
                    reference=reference,
                    level=PLANNED_LEVEL,
                    n_resamples=PLANNED_RESAMPLES,
                ),
            }
        )
    return pl.DataFrame(rows)


def exploratory_contrasts(*, arms: dict[str, pl.DataFrame], labels: list[str]) -> pl.DataFrame:
    """Return the exploratory contrasts, read at 95%."""
    rows = []

    def add(
        label: str,
        treatment: str,
        reference: str,
        *,
        band: str = "all leads",
        group: str | tuple[str, ...] = HEADLINE_GROUP,
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
                    only_label=only_label,
                    daylight=daylight,
                ),
            }
        )

    for band, _, _ in LEAD_BANDS[1:]:
        add(f"C2 by lead band: {band}", CAMS_STAGE0, CAMS_RAW, band=band)
    add("C2, daylight valid times", CAMS_STAGE0, CAMS_RAW, daylight=True)
    add("C2, night valid times", CAMS_STAGE0, CAMS_RAW, daylight=False)
    add("C2, BSP and GSP together", CAMS_STAGE0, CAMS_RAW, group=("BSP", "GSP"))
    add("C2, BSP", CAMS_STAGE0, CAMS_RAW, group="BSP")
    add("C2, GSP", CAMS_STAGE0, CAMS_RAW, group="GSP")
    add("C1, BSP and GSP together", CAMS_RAW, ERA5_RAW, group=("BSP", "GSP"))
    add("X1: era5-stage0 minus era5-trained, Primary", ERA5_STAGE0, ERA5_RAW)
    add("X2: cams-stage0 minus era5-stage0, Primary", CAMS_STAGE0, ERA5_STAGE0)
    add("cams-stage0 minus era5-trained, Primary", CAMS_STAGE0, ERA5_RAW)
    add("cams-stage0 minus climatology, Primary", CAMS_STAGE0, "climatology")
    for label in labels:
        add(f"C2, {label} only", CAMS_STAGE0, CAMS_RAW, only_label=label)
    return pl.DataFrame(rows)


def absolute_table(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return every scored arm's mean quantile score and median error, with 95% intervals."""
    rows = []
    for name in (ERA5_RAW, CAMS_RAW, CAMS_STAGE0, ERA5_STAGE0, "climatology"):
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


def mismatch_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the mean daylight shortwave by month and the member spread, before and after stage 0.

    Rows are the member-rows of the 16 Primary series in the 73 validation runs. `raw`, `to_cams`,
    and `to_era5` are ENS as published and mapped; `cams` and `era5` are the irradiance at the same
    valid times. A ratio is to CAMS (for ENS) or to ERA5 (for the ERA5 mapping).
    """
    primary = series.filter(pl.col("substation_type") == HEADLINE_GROUP)["time_series_id"]
    sums = pl.concat([pl.read_parquet(p) for p in sorted(fit.SHORTWAVE_DIR.glob("init=*.parquet"))])
    sums = sums.filter(pl.col("time_series_id").is_in(primary) & (pl.col("month") < HOLDOUT_MONTH))
    if not sums.select(pl.all_horizontal(pl.exclude("month").is_finite())).to_series().all():
        msg = "the shortwave sums hold a NaN or infinite value"
        raise ValueError(msg)
    value_columns = ["raw", "to_cams", "to_era5", "cams", "era5"]

    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            *[(pl.col(c) / pl.col("n")).alias(f"{c}_mean") for c in value_columns],
            raw_to_cams_ratio=pl.col("raw") / pl.col("cams"),
            stage0_to_cams_ratio=pl.col("to_cams") / pl.col("cams"),
            raw_to_era5_ratio=pl.col("raw") / pl.col("era5"),
            stage0_to_era5_ratio=pl.col("to_era5") / pl.col("era5"),
            spread_ratio_cams=pl.col("to_cams_spread") / pl.col("raw_spread"),
            spread_ratio_era5=pl.col("to_era5_spread") / pl.col("raw_spread"),
        ).select(
            "month",
            "n",
            *[f"{c}_mean" for c in value_columns],
            "raw_to_cams_ratio",
            "stage0_to_cams_ratio",
            "raw_to_era5_ratio",
            "stage0_to_era5_ratio",
            "spread_ratio_cams",
            "spread_ratio_era5",
        )

    aggregates = [
        pl.col(c).sum()
        for c in ["n", *value_columns, "raw_spread", "to_cams_spread", "to_era5_spread"]
    ]
    by_month = finish(sums.group_by("month").agg(aggregates).sort("month"))
    overall = finish(sums.select(aggregates).with_columns(month=pl.lit("all")))
    return pl.concat([by_month, overall])


def verdict(*, planned: pl.DataFrame, mismatch: pl.DataFrame) -> str:
    """Return the pre-registered verdict from C2 and the daylight bias check.

    Args:
        planned: `planned_contrasts()`.
        mismatch: `mismatch_table()`.

    Returns:
        `supported`, `refuted`, or `inconclusive`, with the numbers it rests on.
    """
    c1, c2 = planned.row(0, named=True), planned.row(1, named=True)
    months = mismatch.filter(pl.col("month") != "all")
    overall = mismatch.filter(pl.col("month") == "all").row(0, named=True)
    year_ok = abs(overall["stage0_to_cams_ratio"] - 1) <= BIAS_YEAR_LIMIT
    months_ok = bool(((months["stage0_to_cams_ratio"] - 1).abs() <= BIAS_MONTH_LIMIT).all())
    bias_ok = year_ok and months_ok
    if bias_ok and c2["difference"] <= SUPPORT_THRESHOLD and c2["upper"] < 0:
        label = "supported"
    elif bias_ok and c2["lower"] > SUPPORT_THRESHOLD:
        label = "refuted"
    else:
        label = "inconclusive"
    return (
        f"{label}: C2 is {c2['difference']:+.3f} ({c2['lower']:+.3f} to {c2['upper']:+.3f} at "
        f"97.5%), C1 is {c1['difference']:+.3f} ({c1['lower']:+.3f} to {c1['upper']:+.3f}; "
        f"00009 found {C1_IN_00009:+.3f}, noise floor {C1_NOISE:.3f}); "
        f"bias check {'passed' if bias_ok else 'failed'} (year ratio "
        f"{overall['stage0_to_cams_ratio']:.3f}, worst month deviation "
        f"{float(np.abs(months['stage0_to_cams_ratio'].to_numpy() - 1).max()):.3f})"
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
    check = noise_check(ours=stats)
    ours = {
        name: stats.filter(pl.col("arm") == arm).drop("arm") for arm, name in STAGE_ARMS.items()
    }
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
    mismatch = mismatch_table(series=series)
    for name, frame in (
        ("planned", planned),
        ("exploratory", exploratory),
        ("absolute", absolute),
        ("mismatch", mismatch),
        ("noise_check", check),
    ):
        frame.write_parquet(RESULTS_DIR / f"{name}.parquet")
    contrast_columns = [
        "label",
        "difference",
        "lower",
        "upper",
        "months_negative",
        "months",
        "series_negative",
        "series",
    ]
    score_columns = ["arm", "group", "scope", "value", "lower_95", "upper_95"]
    lines = [
        "# Experiment 00014: scores",
        "",
        f"Verdict by the pre-registered rule: {verdict(planned=planned, mismatch=mismatch)}",
        "",
        "## Repeat of experiment 00009's raw-input quantiles (fractions of capacity)",
        *_table(
            frame=check,
            columns=["arm", "level", "mean_abs_difference", "max_abs_difference", "rows"],
        ),
        "",
        "## Planned contrasts (97.5% intervals; negative favours the first arm)",
        *_table(frame=planned, columns=contrast_columns),
        "",
        "## Exploratory contrasts, 95% intervals",
        *_table(frame=exploratory, columns=contrast_columns),
        "",
        "## Absolute scores (points of capacity; 95% month-and-seed intervals)",
        *_table(frame=absolute.filter(pl.col("metric") == "quantile_score"), columns=score_columns),
        "",
        "## Median error, absolute",
        *_table(frame=absolute.filter(pl.col("metric") == "median_error"), columns=score_columns),
        "",
        "## Daylight shortwave mismatch, 16 Primary series, validation year",
        *_table(frame=mismatch, columns=list(mismatch.columns)),
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
