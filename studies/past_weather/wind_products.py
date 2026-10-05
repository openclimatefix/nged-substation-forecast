"""Score five weather products as descriptions of past wind, on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/wind/>.

**Every product gets one arm with the same four wind columns** — its native hub-height speed, that
height's direction as sine and cosine, and its 10 m speed — plus the hour of day, the day of the
year, and the UKV era, fitted per generator by the tested out-of-fold loop from
`studies.cross_validation`. The hub height is 100 m for ERA5 and UKV and 80 m for the ICON products,
whose served 100 m value is their 120 m speed rescaled. So a contrast between two arms is a contrast
between the products' wind. A second arm per product, shown the served 100 m speed and direction and
the 10 m speed, and a second hyperparameter setting are sensitivity checks. The products are ERA5,
UKV, ICON-D2, ICON-EU, and ICON global, downloaded by `fetch_wind_point.py`.

**The power hour is centred on the label, unlike the solar study's.** Open-Meteo's wind is an
instantaneous value at the label, where its radiation is a mean over the hour ending there, so the
hour labelled T is built from the half-hours ending at T and at T + 30 min. An offset scan in the
plan review found every product scoring best with the hour centred this way.

**An hour holding an exactly-zero half-hour is dropped, whatever any product says.** From April 2026
the feed publishes no exact zeros at two of the generators: their calm half-hours are missing
instead, and `hourly_from_half_hourly` already drops an hour with a missing half-hour. Dropping the
zeros makes the earlier period match. Most dropped hours are calm, so behaviour near cut-in is
under-sampled; dropping no rows at all moves every contrast by 0.03 points or less.

**The folds are cut inside each era of the UKV record, and every arm is told the era**, as in
`weather_products.py`.

Run it with `uv run python studies/past_weather/wind_products.py`, after
`fetch_wind_point.py`. With `--fit-missing` it keeps the losses a full run saved and fits only the
arms they lack. With `--era5-by-year` it fits nothing: it reads the saved losses and writes ERA5's
error against every other product, year by year, under `sources.UPDATE_OUTPUT_DIR`, leaving this
study's own directory untouched.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from studies.arm_runner import add_time_features, run_all
from studies.guards import refuse_to_overwrite
from studies.pv_dataset import wind_sites
from studies.solar_product_frames import with_eras
from studies.sources import STUDY_DATA_DIR, UPDATE_OUTPUT_DIR
from studies.wind_product_frames import (
    OUTPUT_DIR_NAME,
    PRODUCTS,
    ROW_SET_SETTINGS,
    STEP_DATES,
    common_rows,
    hub_height_m,
    jobs,
    joined,
    output_path_for,
)
from weather_products import (
    CONTRAST_HEADER,
    _contrast_line,
    _mae,
    _scope,
    era5_by_year_lines,
    era5_difference_by_year,
    era5_year_change,
    era5_year_change_lines,
    geometry_lines,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


ERA5_BY_YEAR_DIR: Final[Path] = UPDATE_OUTPUT_DIR / "wind"
"""Where `--era5-by-year` writes, apart from the published losses it reads."""


DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_eu_wind", "era5_wind"),
    ("ukv_wind", "era5_wind"),
    ("icon_eu_wind", "ukv_wind"),
    ("icon_d2_wind", "icon_eu_wind"),
)
"""The four contrasts the recommendations rest on, named before the run.

Whether a Great-Britain-wide weather model beats the reanalysis, twice; which of the two
Great-Britain-wide models is better; and whether the regional model adds anything. Every other
contrast in the report is exploratory.
"""

REPORTED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *DECIDING_CONTRASTS,
    ("icon_d2_wind", "ukv_wind"),
)
"""The deciding contrasts, plus ICON-D2 against UKV, reported in every block."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_global_wind", "icon_eu_wind"),
    ("icon_d2_wind", "era5_wind"),
    ("icon_global_wind", "era5_wind"),
)
"""Contrasts reported for context, not relied on."""


STEP_SITE: Final[str] = "W3"
"""The generator at which ICON global's served wind steps, relative to ICON-EU's."""


def _mean_speed_m_s(*, frame: pl.DataFrame, product: str) -> float:
    """Return one product's mean served 100 m wind speed in m/s, from Open-Meteo's km/h.

    Args:
        frame: The common rows.
        product: A key of `PRODUCTS`.

    Returns:
        The mean speed.
    """
    return float(frame.select(pl.col(f"speed_100m_{product}").mean()).item()) / 3.6


def _scoped(*, losses: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Restrict the losses to an era scope from `weather_products`, or to a half of the year.

    Args:
        losses: Per-row losses carrying `month` and `time`.
        scope: `winter` (October to March), `summer` (April to September), or a scope
            `weather_products._scope` accepts.

    Returns:
        The rows belonging to that scope.
    """
    month = pl.col("time").dt.month()
    if scope == "winter":
        return losses.filter((month >= 10) | (month <= 3))
    if scope == "summer":
        return losses.filter(month.is_between(4, 9))
    return _scope(losses=losses, scope=scope)


def _renamed(*, losses: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Rename one family of arms to the `_wind` names the contrasts use.

    Args:
        losses: Per-row losses for one setting.
        suffix: The arm suffix to keep, such as `_100m`.

    Returns:
        Those arms' losses, renamed to end in `_wind`.
    """
    return losses.filter(pl.col("arm").str.ends_with(suffix)).with_columns(
        arm=pl.col("arm").str.replace(f"{suffix}$", "_wind")
    )


def _check_arms(*, losses: pl.DataFrame, wind: pl.DataFrame, sites: list[str]) -> list[str]:
    """Report the post hoc checks: UKV at 80 m, lead-matched contrasts, and ICON global's steps.

    Args:
        losses: The pooled setting's losses, every arm.
        wind: The `_wind` arms.
        sites: The site labels.

    Returns:
        Markdown lines.
    """
    ukv_80m = pl.concat(
        [
            wind.filter(pl.col("arm") != "ukv_wind"),
            losses.filter(pl.col("arm") == "ukv_80m").with_columns(arm=pl.lit("ukv_wind")),
        ],
    )
    lines = [
        "",
        "#### Checks added after the first run (exploratory)",
        "",
        (
            f"MAE: ukv_80m {_mae(losses=losses, arm='ukv_80m'):.3f}, "
            f"era5_step {_mae(losses=losses, arm='era5_step'):.3f}, "
            f"icon_eu_step {_mae(losses=losses, arm='icon_eu_step'):.3f}, "
            f"icon_global_step {_mae(losses=losses, arm='icon_global_step'):.3f}."
        ),
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=ukv_80m, treatment=t, reference=r, label="UKV at 80 m")
        for t, r in (("icon_d2_wind", "ukv_wind"), ("icon_eu_wind", "ukv_wind"))
    ]
    three_hour = wind.with_columns(icon_lead=pl.col("time").dt.hour() % 3)
    lines += [
        _contrast_line(
            losses=three_hour.filter(pl.col("icon_lead") == lead),
            treatment=t,
            reference=r,
            label=f"ICON lead {lead} h",
        )
        for t, r in (("icon_d2_wind", "ukv_wind"), ("icon_eu_wind", "ukv_wind"))
        for lead in (0, 1, 2)
    ]
    lines += [
        _contrast_line(
            losses=_scoped(losses=three_hour, scope=era).filter(pl.col("icon_lead") == lead),
            treatment="icon_d2_wind",
            reference="ukv_wind",
            label=f"ICON lead {lead} h, {era}",
        )
        for era in ("pre", "post")
        for lead in (0, 1, 2)
    ]
    six_hour = wind.with_columns(global_lead=pl.col("time").dt.hour() % 6)
    lines += [
        _contrast_line(
            losses=six_hour.filter(condition),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=label,
        )
        for label, condition in (
            ("ICON global lead 0–2 h, equal to ICON-EU's", pl.col("global_lead") < 3),
            ("ICON global lead 3–5 h", pl.col("global_lead") >= 3),
        )
    ]
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site") == site),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=f"site {site}",
        )
        for site in sites
    ]
    step = losses.filter(pl.col("arm").str.ends_with("_step"))
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_step", reference="icon_eu_step", label=label
        )
        for label, scoped in (
            ("told the step period, all sites", step),
            *(
                (f"told the step period, site {site}", step.filter(pl.col("site") == site))
                for site in sites
            ),
        )
    ]
    others = [site for site in sites if site != STEP_SITE]
    pooled_label = "sites " + " and ".join(others)
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site").is_in(others)),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=pooled_label,
        ),
        _contrast_line(
            losses=step.filter(pl.col("site").is_in(others)),
            treatment="icon_global_step",
            reference="icon_eu_step",
            label=f"told the step period, {pooled_label}",
        ),
    ]
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_step", reference="era5_step", label=label
        )
        for label, scoped in (
            ("told the step period, all sites", step),
            *(
                (f"told the step period, site {site}", step.filter(pl.col("site") == site))
                for site in sites
            ),
        )
    ]
    return lines


def _step_ratio_lines() -> list[str]:
    """Report ICON global's mean wind speed over ICON-EU's at each site, either side of each step.

    Each ratio is of the whole period's means, read from the downloads `fetch_wind_point.py` wrote.

    Returns:
        Markdown lines.
    """
    heights = (10, 80)
    speeds = [f"wind_speed_{height}m" for height in heights]
    joined = (
        pl.read_parquet(output_path_for(product="icon_global"))
        .select("site", "time", *speeds)
        .join(
            pl.read_parquet(output_path_for(product="icon_eu")).select("site", "time", *speeds),
            on=["site", "time"],
            suffix="_eu",
        )
        .with_columns(period=sum(pl.col("time") >= date for date in STEP_DATES))
    )
    lines = [
        "#### ICON global's mean wind speed over ICON-EU's, by step period",
        "",
        "| Site | Height | Before 2 June 2025 | Between | From 2 June 2026 |",
        "|---|---|---|---|---|",
    ]
    for site in sorted(joined["site"].unique().to_list()):
        for height in heights:
            speed = f"wind_speed_{height}m"
            ratios = (
                joined.filter(pl.col("site") == site)
                .group_by("period")
                .agg(ratio=pl.col(speed).mean() / pl.col(f"{speed}_eu").mean())
                .sort("period")["ratio"]
                .to_list()
            )
            cells = " | ".join(f"{ratio:.3f}" for ratio in ratios)
            lines.append(f"| {site} | {height} m | {cells} |")
    return lines


def _row_set_lines(*, losses: pl.DataFrame, wind: pl.DataFrame) -> list[str]:
    """Report the post hoc row-set checks: the solar study's power hour, and keeping zero hours.

    Args:
        losses: Every arm's losses, every setting.
        wind: The main setting's `_wind` arms.

    Returns:
        Markdown lines.
    """
    by_setting = {
        setting: losses.filter(pl.col("setting") == setting) for setting in ROW_SET_SETTINGS
    }
    lines = [
        "#### The power hour and the zero rule (post hoc)",
        "",
        "| Product | Main | Hour ending at the label | Zero hours kept |",
        "|---|---|---|---|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        cells = " | ".join(f"{_mae(losses=scoped, arm=arm):.3f}" for scoped in by_setting.values())
        lines.append(f"| {product} | {_mae(losses=wind, arm=arm):.3f} | {cells} |")
    lines += ["", *CONTRAST_HEADER]
    for setting, scoped in by_setting.items():
        lines += [
            _contrast_line(losses=scoped, treatment=t, reference=r, label=setting)
            for t, r in REPORTED_CONTRASTS
        ]
    lines.append("")
    for setting, scoped in by_setting.items():
        largest = max(
            abs(
                _mae(losses=scoped, arm=t)
                - _mae(losses=scoped, arm=r)
                - (_mae(losses=wind, arm=t) - _mae(losses=wind, arm=r))
            )
            for t, r in REPORTED_CONTRASTS
        )
        lines.append(f"Largest change in a reported contrast's estimate, {setting}: {largest:.3f}.")
    return lines


def _report(*, frame: pl.DataFrame, losses: pl.DataFrame, sites_roster: pl.DataFrame) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        losses: Every arm's losses, at every setting.
        sites_roster: The wind roster, for the distances.

    Returns:
        The report.
    """
    sites = sorted(frame["site"].unique().to_list())
    pooled = losses.filter(pl.col("setting") == "pooled")
    wind = _renamed(losses=pooled, suffix="_wind")
    served_100m = _renamed(losses=pooled, suffix="_100m")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    lines = [
        (
            f"### Five weather products on {frame.height:,} common site-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        "| Product | Hub height shown | All sites | "
        + " | ".join(sites)
        + " | Served 100 m and 10 m | Second setting |",
        "|---" * (len(sites) + 5) + "|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        per_site = [
            f"{_mae(losses=wind.filter(pl.col('site') == site), arm=arm):.3f}" for site in sites
        ]
        lines.append(
            f"| {product} | {hub_height_m(product=product)} m "
            f"| {_mae(losses=wind, arm=arm):.3f} | "
            + " | ".join(per_site)
            + f" | {_mae(losses=served_100m, arm=arm):.3f} "
            f"| {_mae(losses=sensitivity, arm=arm):.3f} |"
        )
    lines += ["", "Mean absolute error as a percentage of each site's P99 output.", ""]
    lines += ["#### Deciding contrasts, named before the run", "", *CONTRAST_HEADER]
    for treatment, reference in REPORTED_CONTRASTS:
        lines.append(
            _contrast_line(losses=wind, treatment=treatment, reference=reference, label="all")
        )
        lines += [
            _contrast_line(
                losses=wind.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in sites
        ]
    lines += [
        "",
        "The last block, ICON-D2 against UKV, is exploratory.",
        "",
        "#### By era and by half of the year (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=_scoped(losses=wind, scope=scope), treatment=t, reference=r, label=scope
        )
        for scope in ("pre", "pre_matched", "post", "winter", "summer")
        for t, r in (*REPORTED_CONTRASTS, ("icon_d2_wind", "era5_wind"))
    ]
    lines += [
        "",
        (
            "The post scope holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        "#### Sensitivity: the served 100 m arm the plan specified, and the second setting",
        "",
        *CONTRAST_HEADER,
    ]
    for label, scoped in (("served 100 m and 10 m", served_100m), ("second setting", sensitivity)):
        lines += [
            _contrast_line(losses=scoped, treatment=t, reference=r, label=label)
            for t, r in REPORTED_CONTRASTS
        ]
    lines += _check_arms(losses=pooled, wind=wind, sites=sites)
    lines += ["", "#### Other contrasts (exploratory)", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=wind, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site") == site),
            treatment="icon_d2_wind",
            reference="era5_wind",
            label=f"site {site}",
        )
        for site in sites
    ]
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_wind", reference="era5_wind", label=label
        )
        for label, scoped in (
            *((scope, _scoped(losses=wind, scope=scope)) for scope in ("winter", "summer")),
            *((f"site {site}", wind.filter(pl.col("site") == site)) for site in sites),
        )
    ]
    lines += [
        "",
        "Mean served 100 m wind speed, all sites: "
        + ", ".join(
            f"{product} {_mean_speed_m_s(frame=frame, product=product):.2f} m/s"
            for product in PRODUCTS
        )
        + ".",
    ]
    lines += ["", *_row_set_lines(losses=losses, wind=wind)]
    lines += ["", *_step_ratio_lines()]
    lines += ["", *geometry_lines(sites=sites_roster, noun="wind farms")]
    return "\n".join(lines) + "\n"


ERA5_BY_YEAR_MONTHS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
"""January to September: the months 2026 shares with every other year in the record.

2026's record stops in September, so comparing full calendar years would compare a partial 2026
against a complete 2025 and let the missing October-December months change the answer. Restricting
every year to the same months keeps the year-to-year comparison paired on season as well as on site.
"""


ERA5_YEAR_CHANGE_YEARS: Final[tuple[int, int]] = (2025, 2026)
"""The two years `write_era5_by_year` tests for a change in ERA5's deficit, on matched months."""


def write_era5_by_year() -> None:
    """Write ERA5's error against every other product, year by year, from the saved losses.

    Reads the main setting's losses the full run saved, and fits nothing. Every year is restricted
    to `ERA5_BY_YEAR_MONTHS`, January to September, so a partial final year does not skew the
    comparison against the complete years before it. Also writes whether the year-by-year
    difference itself changed between `ERA5_YEAR_CHANGE_YEARS`, resampling each year's months
    independently, which two overlapping by-year intervals cannot show.
    """
    losses = pl.read_parquet(STUDY_DATA_DIR / OUTPUT_DIR_NAME / "losses.parquet").filter(
        pl.col("setting") == "pooled"
    )
    paths = [ERA5_BY_YEAR_DIR / "era5_by_year.parquet", ERA5_BY_YEAR_DIR / "era5_by_year.md"]
    change_paths = [
        ERA5_BY_YEAR_DIR / "era5_year_change.parquet",
        ERA5_BY_YEAR_DIR / "era5_year_change.md",
    ]
    refuse_to_overwrite(paths=[*paths, *change_paths])
    other_arms = tuple(f"{product}_wind" for product in PRODUCTS if product != "era5")
    by_year = era5_difference_by_year(
        losses=losses, era5_arm="era5_wind", other_arms=other_arms, months=ERA5_BY_YEAR_MONTHS
    )
    year0, year1 = ERA5_YEAR_CHANGE_YEARS
    changes = era5_year_change(
        losses=losses,
        era5_arm="era5_wind",
        other_arms=other_arms,
        year0=year0,
        year1=year1,
        months=ERA5_BY_YEAR_MONTHS,
    )
    ERA5_BY_YEAR_DIR.mkdir(parents=True, exist_ok=True)
    lines = era5_by_year_lines(by_year=by_year, months_note="January to September")
    change_lines = era5_year_change_lines(changes=changes)
    by_year.write_parquet(paths[0])
    paths[1].write_text("\n".join(lines) + "\n")
    changes.write_parquet(change_paths[0])
    change_paths[1].write_text("\n".join(change_lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n" + "\n".join(change_lines) + "\n")


def main() -> int:
    """Fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-missing",
        action="store_true",
        help="Keep the losses already on disk and fit only the jobs they lack.",
    )
    parser.add_argument(
        "--era5-by-year",
        action="store_true",
        help="Fit nothing; write ERA5's error against every product, year by year.",
    )
    arguments = parser.parse_args()
    if arguments.era5_by_year:
        write_era5_by_year()
        return 0

    sites = wind_sites()
    frame = with_eras(frame=add_time_features(dataset=common_rows(frame=joined(sites=sites))))
    frames = {
        "hour_ending": with_eras(
            frame=add_time_features(dataset=common_rows(frame=joined(sites=sites, centred=False)))
        ),
        "keep_zero_hours": with_eras(
            frame=add_time_features(
                dataset=common_rows(frame=joined(sites=sites), drop_zero_hours=False)
            )
        ),
    }
    by_site = frame.group_by("site", "era").agg(pl.len(), pl.col("month").n_unique()).sort("site")
    _LOG.info("common rows: %d\n%s", frame.height, by_site)

    output_dir = STUDY_DATA_DIR / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "losses.parquet"
    all_jobs = jobs()
    saved = pl.read_parquet(path) if arguments.fit_missing else None
    done = set(saved.select("arm", "setting").unique().iter_rows()) if saved is not None else set()
    missing = [job for job in all_jobs if (job[0], job[1]) not in done]
    _LOG.info("fitting %d jobs of %d", len(missing), len(all_jobs))
    parts = [] if saved is None else [saved]
    for key, dataset in (("main", frame), *frames.items()):
        chosen = [job for job in missing if (job[1] if job[1] in frames else "main") == key]
        if chosen:
            parts.append(run_all(dataset=dataset, jobs=chosen))
    losses = pl.concat(parts)
    losses.write_parquet(path)

    report = _report(frame=frame, losses=losses, sites_roster=sites)
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
