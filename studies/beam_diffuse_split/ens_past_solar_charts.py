"""Draw the leaderboard and planned-contrasts charts for the ENS past-solar section.

One-off throwaway script for the charts of the ECMWF ENS addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, in
`weather_product_charts.py`'s style. **Every number a chart shares with the page is read from
`ens_past_solar.py`'s report**, so a chart cannot disagree with the page.

Generators appear only as `A` to `F`, and no chart carries a calendar date. Every mark is drawn
with `aria=False`.

**Do not run this script until `ens_past_solar.py` has fitted every arm and written its report.**

Run it with `uv run python studies/beam_diffuse_split/ens_past_solar_charts.py`, after
`ens_past_solar.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from build_dataset import _pv_sites
from ens_past_solar import (
    CAMS_3H_ARM,
    CONFOUND_CONTRASTS,
    CONFOUND_HEADING,
    CONTROL_ARM,
    DECIDING_CONTRASTS,
    ERA5_3H_ARM,
    ERA5_3X3_ARM,
    EXPLORATORY_ARMS,
    EXPLORATORY_CONTRASTS,
    MEAN_ARM,
    OUTPUT_DIR,
    _absolute_table_lines,
    _generator_lines,
    _lead_lines,
    _main_panel_lines,
    _servable_lines,
    _support_lines,
    _t3_members,
    build_rows,
)
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    figure,
    interval_panel,
    leaderboard_panel,
    report_contrasts,
    report_errors,
)
from weather_products import METRIC, PERCENTAGE_POINTS, _contrast_line, _mae

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "ens_mean_t3": "ECMWF ENS (T+3 band)",
    "era5_global": "ERA5",
    "cams_global": "CAMS",
}
"""Every arm's public name, as the page writes it."""

EXPLORATORY_NAMES: Final[dict[str, str]] = {
    ERA5_3H_ARM: "ERA5 averaged to 3-hour steps",
    CAMS_3H_ARM: "CAMS averaged to 3-hour steps",
    ERA5_3X3_ARM: "ERA5 averaged over 3 by 3 cells",
}
"""The public names of the exploratory arms, added to `NAMES` for the exploratory chart."""

FAMILIES: Final[dict[str, str]] = {
    "ens_mean_t3": "weather model",
    "era5_global": "reanalysis",
    "cams_global": "satellite",
    ERA5_3H_ARM: "reanalysis",
    ERA5_3X3_ARM: "reanalysis",
    CAMS_3H_ARM: "satellite",
}
"""Every arm's family, which sets its colour in `studies.charts`."""

SECTION_DECIDING: Final[str] = "Planned contrasts"
"""The report heading `ens_past_solar.py` writes above the two planned contrasts."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, April 2024 to September 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends, on either chart."""

FIGURE_LEADERBOARD: Final[int] = 17
FIGURE_CONTRASTS: Final[int] = 18
FIGURE_EXPLORATORY: Final[int] = 19


def _losses(*, setting: str) -> pl.DataFrame:
    """Return every arm's losses at one hyperparameter setting.

    Args:
        setting: `pooled` or `sensitivity`.

    Returns:
        `ens_past_solar.py`'s saved losses at that setting.
    """
    return pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == setting)


def _row_count(*, report: str) -> int:
    """Read the number of common site-hours from the report's heading.

    Args:
        report: The report's text.

    Returns:
        The row count.

    Raises:
        ValueError: If the heading does not hold a count.
    """
    match = re.search(r"on ([\d,]+) common site-hours", report)
    if match is None:
        msg = "report.md has no 'on N common site-hours' heading"
        raise ValueError(msg)
    return int(match[1].replace(",", ""))


def _ens_lead_range(*, report: str) -> str:
    """Read ENS's lead range, such as `5 to 20`, from the report's lead section.

    Args:
        report: The report's text.

    Returns:
        The range, as the report prints it.

    Raises:
        ValueError: If the report has no ENS lead line.
    """
    match = re.search(r"ENS's lead is (\d+ to \d+) hours", report)
    if match is None:
        msg = 'report.md has no "ENS\'s lead is N to M hours" line'
        raise ValueError(msg)
    return match[1]


def _leaderboard(
    *, losses: pl.DataFrame, errors: dict[str, float], report: str, cams_gap: float
) -> alt.VConcatChart:
    """Draw the three headline arms' own mean absolute error, best first, with their 95% intervals.

    Bootstraps each arm's absolute error from `losses.parquet` directly, the same month-and-seed
    resampling `ens_past_solar.py`'s own report interval uses, because `leaderboard_panel` needs a
    `lower_95` and `upper_95` per row and the report's point estimate alone supplies neither. No
    model is refitted.

    Args:
        losses: Every arm's losses, at the `pooled` setting.
        errors: Each arm's pooled mean absolute error, read from the report's first table.
        report: The report's text, for the row count and ENS's leads.
        cams_gap: ENS's planned contrast against CAMS, in points of capacity, for the title.

    Returns:
        The figure.

    Raises:
        ValueError: If a bootstrapped point estimate disagrees with the report's own number.
    """
    order = sorted(NAMES, key=errors.__getitem__)
    records = []
    for arm in order:
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value = interval["value"] * PERCENTAGE_POINTS
        if round(value, 3) != errors[arm]:
            msg = f"{arm}: bootstrapped {value:.3f} but the report says {errors[arm]}"
            raise ValueError(msg)
        records.append(
            {
                "label": NAMES[arm],
                "family": FAMILIES[arm],
                "value": value,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    rows = pl.DataFrame(records)
    domain = (
        min(rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = leaderboard_panel(rows=rows, x_domain=domain, x_title=LEADERBOARD_X_TITLE)
    return figure(
        panels=[panel],
        number=FIGURE_LEADERBOARD,
        figure_planning=None,
        title=f"ENS's own forecast trails CAMS by {cams_gap:.1f} points, and beats ERA5",
        subtitle=[
            f"All three products scored on the same {_row_count(report=report):,} site-hours.",
            (
                f"ENS is a forecast {_ens_lead_range(report=report)} h ahead from a 00 UTC run; "
                "ERA5's radiation is 1 to 12 h ahead; CAMS is a satellite retrieval whose "
                "cloud information has no forecast step."
            ),
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def _planned_contrasts(*, report_path: Path) -> alt.VConcatChart:
    """Draw the two planned contrasts, ENS against ERA5 and against CAMS.

    Args:
        report_path: The `report.md` `ens_past_solar.py` wrote.

    Returns:
        The figure.

    Raises:
        ValueError: If the report does not hold exactly the two planned contrasts, for example
            because `SECTION_DECIDING` no longer matches a renamed report heading.
    """
    contrasts = report_contrasts(report_path=report_path)
    reference_order = {reference: index for index, (_, reference) in enumerate(DECIDING_CONTRASTS)}
    selected = (
        contrasts.filter(
            pl.col("section") == SECTION_DECIDING,
            pl.col("scope") == "all",
            pl.col("treatment") == MEAN_ARM,
            pl.col("reference").is_in(list(reference_order)),
        )
        .with_columns(_order=pl.col("reference").replace_strict(reference_order))
        .sort("_order")
    )
    if selected.height != len(DECIDING_CONTRASTS):
        msg = (
            f"expected {len(DECIDING_CONTRASTS)} planned contrasts under {SECTION_DECIDING!r}, "
            f"found {selected.height}"
        )
        raise ValueError(msg)
    labels = [
        f"{NAMES[MEAN_ARM]} − {NAMES[reference]}" for reference in selected["reference"].to_list()
    ]
    rows = selected.select(
        "treatment", "reference", "difference", "lower_95", "upper_95"
    ).with_columns(
        label=pl.Series(labels), family=pl.lit("weather model"), planned=pl.lit(value=True)
    )
    domain = (
        min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = interval_panel(
        rows=rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="ENS better",
        panel_title="The two planned contrasts",
        figure_planning="planned",
    )
    return figure(
        panels=[panel],
        number=FIGURE_CONTRASTS,
        figure_planning=None,
        title="ENS beats ERA5 but trails CAMS by more than three points",
        subtitle=[
            (
                "Both contrasts were fixed before the first model was fitted, after ERA5 and CAMS "
                "had been scored on the page's main rows."
            ),
            (
                "ENS also differs from ERA5 and CAMS in lead, the 3-hourly steps of its open-data "
                "subset, native resolution, spatial support, and model version."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _exploratory_contrasts(*, report_path: Path) -> alt.VConcatChart:
    """Draw what a 3-hourly step and a wider ERA5 area do to ENS's two gaps.

    Args:
        report_path: The `report.md` `ens_past_solar.py` wrote.

    Returns:
        The figure.

    Raises:
        ValueError: If the report does not hold exactly the `CONFOUND_CONTRASTS`.
    """
    names = {**NAMES, **EXPLORATORY_NAMES}
    contrasts = report_contrasts(report_path=report_path).filter(
        pl.col("section") == CONFOUND_HEADING, pl.col("scope") == "all"
    )
    keys = [f"{treatment} − {reference}" for treatment, reference in CONFOUND_CONTRASTS]
    selected = contrasts.with_columns(
        key=pl.col("treatment") + pl.lit(" − ") + pl.col("reference")
    ).filter(pl.col("key").is_in(keys))
    if selected.height != len(keys):
        msg = f"expected {len(keys)} contrasts under {CONFOUND_HEADING!r}, found {selected.height}"
        raise ValueError(msg)
    selected = selected.with_columns(
        order=pl.col("key").replace_strict({k: i for i, k in enumerate(keys)})
    ).sort("order")
    rows = selected.select(
        "treatment",
        "reference",
        "difference",
        "lower_95",
        "upper_95",
        label=pl.col("treatment").replace_strict(names)
        + pl.lit(" − ")
        + pl.col("reference").replace_strict(names),
        family=pl.col("treatment").replace_strict(FAMILIES),
        planned=pl.lit(value=False),
    )
    domain = (
        min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = interval_panel(
        rows=rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        panel_title="Exploratory contrasts",
        figure_planning="exploratory",
    )
    return figure(
        panels=[panel],
        number=FIGURE_EXPLORATORY,
        figure_planning="exploratory",
        title="Averaging ERA5 and CAMS over 3-hour steps narrows both of ENS's gaps",
        subtitle=[
            (
                "ERA5 and CAMS averaged over the seven 3-hour steps of ENS's open-data subset, "
                "then rebuilt to hourly values by the code that rebuilds ENS's own hours."
            ),
            "ERA5 also averaged over a 3 by 3 block of served 0.25° cells.",
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _check_printed(*, report: str, texts: list[str]) -> None:
    """Stop unless every formatted number appears in the report as printed.

    Args:
        report: The report's text.
        texts: The table-cell fragments to look for.

    Raises:
        ValueError: If any fragment is missing.
    """
    missing = [text for text in texts if text not in report]
    if missing:
        msg = f"{len(missing)} numbers are not in report.md as printed, such as {missing[:3]}"
        raise ValueError(msg)


def _verify_numbers(*, report: str, losses: pl.DataFrame, sensitivity: pl.DataFrame) -> None:
    """Recompute every number this module's charts and prose rest on, and check each is printed.

    `_leaderboard` already checks the two headline arms `NAMES` covers against a fresh bootstrap;
    this recomputes the rest directly from `losses.parquet` and `build_rows`, independently of
    `report_contrasts`'s markdown parsing: the control arm's own error, the two planned contrasts
    per generator, the exploratory contrasts, and each line of the exploratory sections.

    Args:
        report: `report.md`'s text.
        losses: Every arm's losses, at the `pooled` setting.
        sensitivity: Every arm's losses, at the `sensitivity` setting.

    Raises:
        ValueError: If any recomputed number is not in the report as printed.
    """
    site_labels = sorted(losses["site"].unique().to_list())
    control_interval = bootstrap_absolute(losses=losses, arm=CONTROL_ARM, metric=METRIC)
    control_lower, control_upper = (
        control_interval[key] * PERCENTAGE_POINTS for key in ("lower_95", "upper_95")
    )
    texts = [
        (
            f"| {CONTROL_ARM} | {_mae(losses=losses, arm=CONTROL_ARM):.3f} "
            f"| [{control_lower:.3f}, {control_upper:.3f}] |"
        )
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        texts += [
            _contrast_line(
                losses=losses.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in site_labels
        ]
    texts += [
        _contrast_line(losses=losses, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    frame = build_rows()
    texts.append(f"{frame.height:,} common site-hours")
    texts += [
        _contrast_line(losses=losses, treatment=t, reference=r, label="all")
        for t, r in CONFOUND_CONTRASTS
    ]
    texts += [
        _contrast_line(losses=sensitivity, treatment=t, reference=r, label="sensitivity")
        for t, r in CONFOUND_CONTRASTS
    ]
    texts += [
        line
        for lines in (
            _absolute_table_lines(pooled=losses, arms=EXPLORATORY_ARMS),
            _servable_lines(pooled=losses),
            _lead_lines(frame=frame),
            _support_lines(sites=_pv_sites()),
            _generator_lines(frame=frame, members=_t3_members()),
            _main_panel_lines(pooled=losses),
        )
        for line in lines
        if line and not line.startswith("#")
    ]
    _check_printed(report=report, texts=texts)


def main() -> int:
    """Read the report and write the three SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = OUTPUT_DIR / "report.md"
    report = report_path.read_text()
    errors = report_errors(report_path=report_path, column="All sites")
    errors = {arm: errors[arm] for arm in NAMES if arm in errors}
    losses = _losses(setting="pooled")
    _verify_numbers(report=report, losses=losses, sensitivity=_losses(setting="sensitivity"))
    cams_gap = float(
        report_contrasts(report_path=report_path)
        .filter(
            pl.col("section") == SECTION_DECIDING,
            pl.col("scope") == "all",
            pl.col("treatment") == MEAN_ARM,
            pl.col("reference") == "cams_global",
        )["difference"]
        .item()
    )
    charts = {
        "ens_past_solar_leaderboard": _leaderboard(
            losses=losses, errors=errors, report=report, cams_gap=cams_gap
        ),
        "ens_past_solar_planned_contrasts": _planned_contrasts(report_path=report_path),
        "ens_past_solar_exploratory_contrasts": _exploratory_contrasts(report_path=report_path),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
