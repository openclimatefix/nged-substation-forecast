"""Print the stage 1 pilot's tables from the saved per-row scores, and save them for the charts.

Reads `scores/rows_seed<s>.parquet` and writes, under `results/`: `report.md` (every table the
README and the charts quote), and `mae.parquet`, `contrasts.parquet`, `crps.parquet`,
`slopes.parquet`, `monthly_mae.parquet`. Series appear only as the anonymous labels A to F and
W1 to W3.

    uv run python studies/stage1_pilot/stage1_report.py
"""

from pathlib import Path
from typing import Final

import polars as pl
import stage1_inputs as inputs
import stage1_stats as stats
from stage1_score_arms import ARMS as NEW_ARMS
from stage1_score_arms import MEMBER_ARMS as NEW_MEMBER_ARMS
from stage1_score_arms import SCORE_DIR

RESULT_DIR: Final[Path] = inputs.RESULTS_DIR

PRIOR_SCORE_DIR: Final[Path] = (
    inputs.OUTPUT_ROOT / "superseded" / "with_calendar_and_production_columns" / "scores"
)
"""The first run's scores: stage 1a and the ENS-trained recipe with hour and day-of-year columns."""

CALENDAR_ARMS: Final[dict[str, str]] = {
    "s1_memberavg": "s1_memberavg_calendar",
    "s1_control": "s1_control_calendar",
    "s1_cams": "s1_cams_calendar",
    "s1_recipe_on_ens": "ens_trained_calendar",
    "era5_ceiling": "era5_ceiling_calendar",
}
"""First-run arms kept for the feature-list comparison, renamed, with the hour sin and cos and
day-of-year sin and cos columns (solar 13 columns, wind 10)."""

MEMBER_ARMS: Final[tuple[str, ...]] = (
    *NEW_MEMBER_ARMS,
    "s1_memberavg_calendar",
    "s1_cams_calendar",
    "ens_trained_calendar",
)
ARMS: Final[tuple[str, ...]] = (*NEW_ARMS, *CALENDAR_ARMS.values())

PER_SERIES_ARMS: Final[tuple[str, ...]] = (
    "c0",
    "climatology",
    "ens_trained",
    "s1_memberavg",
    "ens_trained_plus_s1",
    "era5_ceiling",
)

BANDS: Final[tuple[str, ...]] = ("all", *(name for name, _, _ in inputs.LEAD_BANDS))
TECHNOLOGIES: Final[tuple[str, ...]] = ("solar", "wind")

PLANNED: Final[dict[str, tuple[str, str]]] = {
    "P1 (revised)": ("s1_memberavg", "ens_trained"),
    "P2 (revised)": ("ens_trained_plus_s1", "ens_trained"),
}
"""The planned contrasts as (treatment, reference), in the README's order. Both were revised after
the scientific reviews: the reference changed from the production-column control `c0` to
`ens_trained`, which has the same columns as stage 1a."""

SLOPE_ARMS: Final[tuple[str, ...]] = (
    "c0",
    "ens_trained",
    "ens_trained_control",
    "climatology",
    "s1_memberavg",
    "s1_control",
    "s1_short",
    "ens_trained_plus_s1",
    "s1_cams",
    "era5_ceiling",
)
"""Arms whose slope is reported; the ceiling is the slope with no forecast error in the weather."""

EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: s1_control minus ens_trained_control (ERA5 against ENS training, control member only)": (
        "s1_control",
        "ens_trained_control",
    ),
    "E1b: ens_trained minus ens_trained_control (averaging the members, ENS-trained)": (
        "ens_trained",
        "ens_trained_control",
    ),
    "E2: s1_memberavg minus s1_control (averaging the members)": ("s1_memberavg", "s1_control"),
    "E3: s1_cams minus ens_trained (solar)": ("s1_cams", "ens_trained"),
    "E4: s1_cams minus s1_memberavg (solar)": ("s1_cams", "s1_memberavg"),
    "E5: s1_memberavg minus s1_short (effect of the training length)": (
        "s1_memberavg",
        "s1_short",
    ),
    "E6: s1_short minus ens_trained (effect of the training weather)": (
        "s1_short",
        "ens_trained",
    ),
    "E7: s1_memberavg minus era5_ceiling (forecast-weather penalty)": (
        "s1_memberavg",
        "era5_ceiling",
    ),
    "E8: s1_memberavg minus climatology": ("s1_memberavg", "climatology"),
    "E9: ens_trained minus climatology": ("ens_trained", "climatology"),
    "E10: P1 on forecasts clipped to 0 to capacity": ("s1_memberavg", "ens_trained"),
    "E11: P2 on forecasts clipped to 0 to capacity": ("ens_trained_plus_s1", "ens_trained"),
    "E12: s1_memberavg minus c0 (the original P1, against the production columns)": (
        "s1_memberavg",
        "c0",
    ),
    "E13: ens_trained minus c0 (effect of dropping the production columns)": (
        "ens_trained",
        "c0",
    ),
    "E14: c0 minus climatology": ("c0", "climatology"),
    "F1 (feature lists): s1_memberavg minus s1_memberavg_calendar": (
        "s1_memberavg",
        "s1_memberavg_calendar",
    ),
    "F2 (feature lists): ens_trained minus ens_trained_calendar": (
        "ens_trained",
        "ens_trained_calendar",
    ),
    "F3 (feature lists): s1_control minus s1_control_calendar": (
        "s1_control",
        "s1_control_calendar",
    ),
    "F4 (feature lists): s1_cams minus s1_cams_calendar (solar)": (
        "s1_cams",
        "s1_cams_calendar",
    ),
    "F5 (feature lists): era5_ceiling minus era5_ceiling_calendar": (
        "era5_ceiling",
        "era5_ceiling_calendar",
    ),
    "F6 (first run's P1): s1_memberavg_calendar minus c0": ("s1_memberavg_calendar", "c0"),
}
"""Contrasts that were not planned: all are exploratory."""


def load_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Read every seed's rows and attach technology and lead band."""
    rows = pl.concat([pl.read_parquet(SCORE_DIR / f"rows_seed{s}.parquet") for s in inputs.SEEDS])
    prior = pl.concat(
        [pl.read_parquet(PRIOR_SCORE_DIR / f"rows_seed{s}.parquet") for s in inputs.SEEDS]
    )
    prior = prior.filter(pl.col("arm").is_in(list(CALENDAR_ARMS))).with_columns(
        pl.col("arm").replace(CALENDAR_ARMS)
    )
    keys = ["seed", "time_series_id", "power_fcst_init_time", "valid_time"]
    shared = rows.filter(pl.col("arm") == "era5_ceiling").select(keys)
    if (
        prior.filter(pl.col("arm") == "era5_ceiling_calendar")
        .select(keys)
        .sort(keys)
        .equals(shared.sort(keys))
        is False
    ):
        msg = "the first run scored different cases from this run"
        raise ValueError(msg)
    rows = pl.concat([rows, prior.select(rows.columns)])
    band = pl.lit("none")
    for name, low, high in reversed(inputs.LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return rows.join(
        sites.select("time_series_id", "technology", "site"), on="time_series_id"
    ).with_columns(band=band)


def scope(*, rows: pl.DataFrame, arm: str, technology: str, band: str) -> pl.DataFrame:
    """Return one arm's rows within a technology and lead band (`all` keeps every band)."""
    selected = rows.filter((pl.col("arm") == arm) & (pl.col("technology") == technology))
    return selected if band == "all" else selected.filter(pl.col("band") == band)


def table(*, header: list[str], body: list[list[str]]) -> list[str]:
    """Format a Markdown table."""
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
        "",
    ]


def main() -> None:  # noqa: PLR0912, PLR0915
    """Compute every table and write the report."""
    sites = inputs.site_table()
    rows = load_rows(sites=sites)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Stage 1 pilot results", ""]

    counts = (
        rows.filter(pl.col("seed") == 0)
        .group_by("arm", "technology")
        .agg(rows=pl.len(), series=pl.col("time_series_id").n_unique())
        .sort("arm", "technology")
    )
    report += ["## Rows scored per arm and fitting seed", ""]
    report += table(
        header=["arm", "technology", "scored half-hours", "series"],
        body=[
            [r["arm"], r["technology"], f"{r['rows']:,}", str(r["series"])]
            for r in counts.iter_rows(named=True)
        ],
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    duplicates = rows.group_by("arm", "seed", *keys).len().filter(pl.col("len") > 1).height
    report += [f"Duplicate (arm, seed, case) keys: {duplicates}.", ""]
    if duplicates:
        msg = "duplicate scored cases"
        raise ValueError(msg)

    # Absolute mean error.
    mae_rows = []
    for arm in ARMS:
        for technology in TECHNOLOGIES:
            for band in BANDS:
                part = scope(rows=rows, arm=arm, technology=technology, band=band)
                if part.height == 0:
                    continue
                interval = stats.level(table=stats.monthly(rows=part, value="abs_error"))
                mae_rows.append(
                    {
                        "arm": arm,
                        "technology": technology,
                        "band": band,
                        "value": interval.value,
                        "lower": interval.lower,
                        "upper": interval.upper,
                        "seed_spread": interval.seed_spread,
                        "n_months": interval.n_months,
                        "rows_per_seed": part.height // len(inputs.SEEDS),
                    }
                )
    mae = pl.DataFrame(mae_rows)
    mae.write_parquet(RESULT_DIR / "mae.parquet")
    report += [
        "## Mean absolute error of the ensemble-mean forecast",
        "",
        (
            "Points of effective capacity (smaller is better); 95% intervals resample whole "
            "validation "
            "months and one fitting seed. `era5_ceiling` and `cams_ceiling` apply stage 1a to ERA5 "
            "or CAMS at the valid time, so they do not depend on lead; they are not a bound on "
            "ENS-driven versions, because ERA5 and CAMS are themselves estimates."
        ),
        "",
    ]
    for technology in TECHNOLOGIES:
        report += [f"### {technology}", ""]
        body = []
        for arm in ARMS:
            cells = []
            for band in BANDS:
                hit = mae.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("technology") == technology)
                    & (pl.col("band") == band)
                )
                cells.append(
                    f"{hit['value'][0]:.2f} ({hit['lower'][0]:.2f} to {hit['upper'][0]:.2f})"
                    if hit.height
                    else "n/a"
                )
            body.append([arm, *cells])
        report += table(header=["arm", *BANDS], body=body)

    # Contrasts.
    contrast_rows = []
    for name, (treatment, reference) in {**PLANNED, **EXPLORATORY}.items():
        for technology in TECHNOLOGIES:
            if "(solar)" in name and technology != "solar":
                continue
            for band in BANDS:
                a = scope(rows=rows, arm=treatment, technology=technology, band=band)
                b = scope(rows=rows, arm=reference, technology=technology, band=band)
                if a.height == 0 or b.height == 0:
                    continue
                column = "abs_error_clipped" if "clipped" in name else "abs_error"
                result = stats.contrast(
                    treatment=stats.monthly(rows=a, value=column),
                    reference=stats.monthly(rows=b, value=column),
                )
                contrast_rows.append(
                    {
                        "contrast": name,
                        "planned": name in PLANNED,
                        "technology": technology,
                        "band": band,
                        "difference": result.difference,
                        "lower": result.lower,
                        "upper": result.upper,
                        "months_better": result.months_better,
                        "n_months": result.n_months,
                        "material": result.material and name in PLANNED,
                    }
                )
    contrasts = pl.DataFrame(contrast_rows)
    contrasts.write_parquet(RESULT_DIR / "contrasts.parquet")
    report += [
        "## Paired contrasts in mean absolute error",
        "",
        (
            "Treatment minus reference, in points of capacity (negative: the treatment has the "
            "smaller "
            "error). `months` is the number of validation months in which the difference is "
            "negative. "
            "A planned contrast is material when the mean difference is at most -0.15, the 95% "
            "interval lies wholly below zero, and at least 75% of months (9 of 12) are negative. "
            "Rows marked P1 and P2 are the planned contrasts; every other row is exploratory."
        ),
        "",
    ]
    for technology in TECHNOLOGIES:
        report += [f"### {technology}", ""]
        body = []
        for name in {**PLANNED, **EXPLORATORY}:
            for band in BANDS:
                hit = contrasts.filter(
                    (pl.col("contrast") == name)
                    & (pl.col("technology") == technology)
                    & (pl.col("band") == band)
                )
                if not hit.height:
                    continue
                r = hit.row(0, named=True)
                body.append(
                    [
                        name,
                        band,
                        f"{r['difference']:+.3f}",
                        f"{r['lower']:+.3f} to {r['upper']:+.3f}",
                        f"{r['months_better']} of {r['n_months']}",
                        "yes" if r["material"] else ("no" if r["planned"] else "-"),
                    ]
                )
        report += table(
            header=["contrast", "lead band", "difference", "95% interval", "months", "material"],
            body=body,
        )

    # CRPS.
    crps_rows = []
    for arm in MEMBER_ARMS:
        for technology in TECHNOLOGIES:
            for band in BANDS:
                part = scope(rows=rows, arm=arm, technology=technology, band=band)
                if part.height == 0:
                    continue
                interval = stats.level(table=stats.monthly(rows=part, value="crps"))
                crps_rows.append(
                    {
                        "arm": arm,
                        "technology": technology,
                        "band": band,
                        "value": interval.value,
                        "lower": interval.lower,
                        "upper": interval.upper,
                    }
                )
    crps = pl.DataFrame(crps_rows)
    crps.write_parquet(RESULT_DIR / "crps.parquet")
    report += [
        "## Fair CRPS of the 51 member point predictions (exploratory)",
        "",
        (
            "Points of capacity. The models are squared-error point models, so the 51 member "
            "predictions are not a calibrated distribution; this only shows whether the member "
            "spread is sensible."
        ),
        "",
    ]
    for technology in TECHNOLOGIES:
        report += [f"### {technology}", ""]
        body = []
        for arm in MEMBER_ARMS:
            cells = []
            for band in BANDS:
                hit = crps.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("technology") == technology)
                    & (pl.col("band") == band)
                )
                cells.append(
                    f"{hit['value'][0]:.2f} ({hit['lower'][0]:.2f} to {hit['upper'][0]:.2f})"
                    if hit.height
                    else "n/a"
                )
            body.append([arm, *cells])
        report += table(header=["arm", *BANDS], body=body)

    # Slopes.
    slope_rows = []
    for arm in SLOPE_ARMS:
        for technology in TECHNOLOGIES:
            for band in BANDS:
                part = scope(rows=rows, arm=arm, technology=technology, band=band)
                if part.height == 0:
                    continue
                interval = stats.slope_interval(rows=part)
                slope_rows.append(
                    {
                        "arm": arm,
                        "technology": technology,
                        "band": band,
                        "slope": interval.value,
                        "lower": interval.lower,
                        "upper": interval.upper,
                    }
                )
    slopes = pl.DataFrame(slope_rows)
    slopes.write_parquet(RESULT_DIR / "slopes.parquet")
    report += [
        "## Slope of observed on predicted power (over-reaction diagnostic)",
        "",
        (
            "Ordinary least squares of observed power on the ensemble-mean forecast, both as "
            "fractions "
            "of capacity, pooled over the rows of each cell, then averaged over fitting seeds. A "
            "slope "
            "below 1 means the forecast varies more than the outcomes it predicts, but any noisy "
            "forecast has a slope below 1, so read each slope against `era5_ceiling` (stage 1a "
            "given "
            "the analysis weather), not against 1. The slope pools rows, so a series with more "
            "rows "
            "counts more; the 95% intervals resample months and one seed."
        ),
        "",
    ]
    for technology in TECHNOLOGIES:
        report += [f"### {technology}", ""]
        body = []
        for arm in SLOPE_ARMS:
            cells = []
            for band in BANDS:
                hit = slopes.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("technology") == technology)
                    & (pl.col("band") == band)
                )
                cells.append(
                    f"{hit['slope'][0]:.3f} ({hit['lower'][0]:.3f} to {hit['upper'][0]:.3f})"
                    if hit.height
                    else "n/a"
                )
            body.append([arm, *cells])
        report += table(header=["arm", *BANDS], body=body)

    # Per-series, all leads.
    per_series = (
        rows.filter(pl.col("arm").is_in(PER_SERIES_ARMS))
        .group_by("site", "arm", "seed")
        .agg(pl.col("abs_error").mean())
        .group_by("site", "arm")
        .agg(pl.col("abs_error").mean())
        .sort("site", "arm")
    )
    report += ["## Mean absolute error by series, all leads (points of capacity)", ""]
    report += table(
        header=["series", *PER_SERIES_ARMS],
        body=[
            [
                site,
                *[
                    f"{per_series.filter(site=site, arm=arm)['abs_error'][0]:.2f}"
                    for arm in PER_SERIES_ARMS
                ],
            ]
            for site in sorted(per_series["site"].unique().to_list())
        ],
    )

    # Monthly means for charts.
    monthly = (
        rows.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("arm", "technology", "month", "seed")
        .agg(abs_error=pl.col("abs_error").mean())
        .group_by("arm", "technology", "month")
        .agg(abs_error=pl.col("abs_error").mean())
        .sort("arm", "technology", "month")
    )
    monthly.write_parquet(RESULT_DIR / "monthly_mae.parquet")

    # Feature lists.
    report += ["## Columns each arm was given", ""]
    report += [
        f"- Stage 1a, solar farm ({len(inputs.SOLAR_FEATURES)}): "
        + ", ".join(inputs.SOLAR_FEATURES),
        f"- Stage 1a, wind farm ({len(inputs.WIND_FEATURES)}): " + ", ".join(inputs.WIND_FEATURES),
        (
            "- `c0` (reference only): the 24 production columns of the fixed pipeline (see "
            "`stage1_fit_enstrained.py` and the stored fit's `features.json`)."
        ),
        "- `ens_trained`, `s1_short`, `s1_cams`, `s1_control`, `s1_memberavg`: stage 1a's columns.",
        "- `ens_trained_plus_s1`: the stage 1a columns plus `stage1a_prediction`.",
        (
            "- `*_calendar` arms (first run, with hour and day-of-year sine and cosine added; "
            "solar 13 columns, wind 10): same fits, same cases, `scores` of the superseded run."
        ),
        "- `climatology`: mean power by local month and half-hour of day, 2019-09 to 2025-06.",
        "",
    ]
    (RESULT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
