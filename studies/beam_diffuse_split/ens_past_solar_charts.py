"""Draw the exploratory-contrasts chart for the ENS past-solar section.

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
from figure_numbers import FIGURE_NUMBERS
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    figure,
    interval_panel,
    report_contrasts,
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

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, April 2024 to September 2026."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends, on either chart."""

GENERATOR_LABELS: Final[tuple[str, ...]] = tuple("ABCDEF")
"""The anonymised generator labels, one row per label in a per-generator contrast."""


def _losses(*, setting: str) -> pl.DataFrame:
    """Return every arm's losses at one hyperparameter setting.

    Args:
        setting: `pooled` or `sensitivity`.

    Returns:
        `ens_past_solar.py`'s saved losses at that setting.
    """
    return pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == setting)


SECTION_PER_GENERATOR: Final[str] = "The same two contrasts, per generator (exploratory)"
"""The report heading `ens_past_solar.py` writes above each planned contrast at each generator."""


def per_generator_rows(*, report_path: Path) -> list[pl.DataFrame]:
    """Read each planned contrast at each generator from the report, one frame per contrast.

    Args:
        report_path: The `report.md` `ens_past_solar.py` wrote.

    Returns:
        One frame per planned contrast, in `DECIDING_CONTRASTS` order, with one row per generator
        (`Generator A` to `Generator F`) holding `treatment`, `reference`, `label`, `family`,
        `planned` (false), `difference`, `lower_95` and `upper_95`.

    Raises:
        ValueError: If a contrast lacks exactly one row per generator.
    """
    per_site = report_contrasts(report_path=report_path).filter(
        pl.col("section") == SECTION_PER_GENERATOR
    )
    frames = []
    for treatment, reference in DECIDING_CONTRASTS:
        selected = per_site.filter(
            pl.col("treatment") == treatment, pl.col("reference") == reference
        ).sort("scope")
        if selected.height != len(GENERATOR_LABELS):
            msg = f"{treatment} - {reference}: expected one row per generator"
            raise ValueError(msg)
        frames.append(
            selected.select(
                "treatment",
                "reference",
                "difference",
                "lower_95",
                "upper_95",
                label=pl.col("scope").str.replace("site ", "Generator "),
                family=pl.lit(FAMILIES[treatment]),
                planned=pl.lit(value=False),
            )
        )
    return frames


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
        number=FIGURE_NUMBERS["ens_exploratory"],
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
    """Check the report against the saved losses, then write the exploratory-contrasts SVG."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = OUTPUT_DIR / "report.md"
    report = report_path.read_text()
    losses = _losses(setting="pooled")
    _verify_numbers(report=report, losses=losses, sensitivity=_losses(setting="sensitivity"))
    charts = {
        "ens_past_solar_exploratory_contrasts": _exploratory_contrasts(report_path=report_path),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
