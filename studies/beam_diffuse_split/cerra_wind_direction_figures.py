"""Draw the three figures of the write-up on whether CERRA's wind direction adds to its wind speed.

One-off throwaway script for the figures of
<https://github.com/openclimatefix/nged-substation-forecast/issues/994>. **Every number a figure
shares with the page is read from `intervals.parquet` and checked against the `report.md` beside
it**, so a figure cannot disagree with the page.

Figure 1 draws the three planned contrasts with their 98.33% intervals. Figure 2 draws the
exploratory contrasts about wind veer and the negative controls, with the positive controls under
them. Figure 3 splits the three planned contrasts by wind farm.

Wind farms appear only as `W1` to `W3`, and no figure carries a calendar date. Every mark is drawn
with `aria=False`.

**The script refuses to overwrite a figure that exists.** Move an old figure out of the way
first. `--dry-run` reads and checks everything, builds the three figures in memory, and writes
nothing.

Run it with
`uv run python studies/beam_diffuse_split/cerra_wind_direction_figures.py`, then optimise each
SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from cerra_wind_levels import PRIMARY_SETTING, SENSITIVITY_SETTING
from sources import STUDIES_DATA_DIR
from studies.charts import figure, interval_panel
from studies.guards import refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR: Final[Path] = STUDIES_DATA_DIR / "cerra_wind_direction"
"""Where `cerra_wind_direction.py` wrote the intervals and the report."""

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

FIGURE_FILES: Final[dict[int, str]] = {
    1: "cerra_wind_direction_planned.svg",
    2: "cerra_wind_direction_veer.svg",
    3: "cerra_wind_direction_farms.svg",
}
"""The file name of each figure, by the figure's number on the page."""

FAMILY: Final = "reanalysis"
"""Every row reads CERRA, so every row takes the reanalysis colour."""

FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
FARM_COLOURS: Final[tuple[str, ...]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE)
"""One main data colour per wind farm, so the three farms stay distinct in Figure 3."""

NOISE_FLOOR_PP: Final[float] = 0.05
"""Half the width of the shaded band in Figure 2: the size of change that four to eight unusable
columns produced in the negative controls (0.040 to 0.060 points)."""

DOMAIN_STEP: Final[float] = 0.05
"""Axis limits are rounded outward to a multiple of this many points."""

DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
HOLLOW: Final[str] = "Hollow triangle: the second hyperparameter setting."
CAPACITY: Final[str] = "Differences are in points of each wind farm's capacity."
SCOPE: Final[str] = "Three wind farms, September 2019 to June 2026, on CERRA's 3-hourly analysis."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"

ROW_STEP_PX: Final[int] = 44
FARM_ROW_STEP_PX: Final[int] = 78
"""The height of one row in Figure 3, which holds three marks per row."""

ReportKey = tuple[str, str, str]
"""A contrast's setting, treatment arm, and reference arm."""

PLANNED: Final[tuple[ReportKey, ...]] = (
    (PRIMARY_SETTING, "speed_100m_dir", "speed_100m"),
    (PRIMARY_SETTING, "speed_10m_dir", "speed_10m"),
    (PRIMARY_SETTING, "speed_100m_dir", "speed_10m_dir"),
)
"""Figure 1's rows, which are also Figure 3's."""

PLANNED_LABELS: Final[dict[tuple[str, str], str]] = {
    ("speed_100m_dir", "speed_100m"): "100 m direction added to the 100 m speed",
    ("speed_10m_dir", "speed_10m"): "10 m direction added to the 10 m speed",
    ("speed_100m_dir", "speed_10m_dir"): "100 m against 10 m, each with its direction",
}
"""The public name of each planned contrast, by (treatment, reference)."""

VEER_LABELS: Final[dict[tuple[str, str], str]] = {
    ("veer_dir_10_100", "veer_dir_100"): "10 m direction added to the 100 m direction",
    ("veer_angle_10_100", "veer_dir_100"): "Explicit veer added to the 100 m direction",
    ("veer_dir_all5", "veer_dir_100"): "Directions at all 5 heights against the 100 m direction",
    ("veer_dir_all5", "veer_dir_10_100"): "Directions at 5 heights against 10 m and 100 m",
    ("veer_dir_10_100", "veer_dir_100_noise"): "10 m direction against a shuffled 10 m direction",
}
"""Figure 2's exploratory rows, top to bottom."""

NEGATIVE_LABELS: Final[dict[tuple[str, str], str]] = {
    ("speed_100m_dir_noise", "speed_100m"): (
        "Negative control: shuffled 100 m direction added to the 100 m speed"
    ),
    ("veer_dir_100_noise", "veer_dir_100"): (
        "Negative control: shuffled 10 m direction added to the 100 m direction"
    ),
}
"""Figure 2's negative controls, below the exploratory rows."""

POSITIVE_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("injected_veer_40", "veer_angle_10_100", "veer_dir_100", "40% cut: explicit veer"),
    ("injected_veer_40", "veer_dir_10_100", "veer_dir_100", "40% cut: 10 m direction"),
    ("injected_veer_40", "veer_dir_all5", "veer_dir_100", "40% cut: all 5 directions"),
    ("injected_veer_10", "veer_angle_10_100", "veer_dir_100", "10% cut: explicit veer"),
    ("injected_veer_10", "veer_dir_10_100", "veer_dir_100", "10% cut: 10 m direction"),
    ("injected_veer_10", "veer_dir_all5", "veer_dir_100", "10% cut: all 5 directions"),
)
"""Figure 2's positive controls: (setting, treatment, reference, label), top to bottom. Each row
adds the named direction columns to the 100 m direction on a target with a veer effect injected."""

POSITIVE_PREFIX: Final[str] = "Injected veer, "


def _intervals(*, intervals: pl.DataFrame, key: ReportKey, scope: str = "all") -> dict:
    """Return the one saved row of a contrast.

    Args:
        intervals: `intervals.parquet`.
        key: The contrast's setting, treatment arm, and reference arm.
        scope: `all`, or a wind farm's label.

    Returns:
        The row, as a dictionary.

    Raises:
        ValueError: If the contrast does not have exactly one row.
    """
    setting, treatment, reference = key
    found = intervals.filter(
        pl.col("setting") == setting,
        pl.col("scope") == scope,
        pl.col("treatment") == treatment,
        pl.col("reference") == reference,
    )
    if found.height != 1:
        msg = f"expected one row for {key} at scope {scope}, found {found.height}"
        raise ValueError(msg)
    return found.row(0, named=True)


def _check_printed(*, report: str, texts: list[str]) -> None:
    """Stop unless every formatted number appears in the report as printed.

    Args:
        report: The report's text.
        texts: The table-row fragments to look for.

    Raises:
        ValueError: If any fragment is missing.
    """
    missing = [text for text in texts if text not in report]
    if missing:
        msg = f"{len(missing)} numbers are not in report.md as printed, such as {missing[:3]}"
        raise ValueError(msg)


def _fragment(*, row: dict, with_adjusted: bool = False) -> str:
    """Format a contrast row as the report prints it, from the scope to the unadjusted interval.

    Args:
        row: A row of `intervals.parquet`.
        with_adjusted: Whether to add the 98.33% interval's cell after the significance cell.

    Returns:
        The report's text for the row.
    """
    text = (
        f"| {row['setting']}, {row['scope']} | {row['treatment']} − {row['reference']} "
        f"| {row['treatment_mae_pp']:.3f} | {row['reference_mae_pp']:.3f} "
        f"| {row['difference_pp']:+.3f} "
        f"| [{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}] |"
    )
    if with_adjusted:
        text += (
            f" {'**yes**' if row['significant'] else 'no'} "
            f"| [{row['lower_bonferroni_pp']:+.3f}, {row['upper_bonferroni_pp']:+.3f}] |"
        )
    return text


def _domain(*, lows: list[float], highs: list[float]) -> tuple[float, float]:
    """Round the data's range, with zero, outward to a multiple of `DOMAIN_STEP` plus one step.

    Args:
        lows: Every row's lowest plotted value.
        highs: Every row's highest plotted value.

    Returns:
        The axis limits.
    """
    low = math.floor(min(0.0, *lows) / DOMAIN_STEP) * DOMAIN_STEP - DOMAIN_STEP
    high = math.ceil(max(0.0, *highs) / DOMAIN_STEP) * DOMAIN_STEP + DOMAIN_STEP
    return round(low, 2), round(high, 2)


def _panel_rows(*, rows: list[dict]) -> pl.DataFrame:
    """Build an `interval_panel` frame from the rows' prepared fields.

    Args:
        rows: One dictionary per mark, with `label`, `difference`, `lower_95`, `upper_95`, and
            optionally `second_difference` and `condition`.

    Returns:
        The frame, with the reanalysis family on every row and no row planned.
    """
    frame = pl.DataFrame(rows)
    if "second_difference" not in frame.columns:
        frame = frame.with_columns(second_difference=pl.lit(None, dtype=pl.Float64))
    return frame.with_columns(family=pl.lit(FAMILY), planned=pl.lit(value=False))


def _contrast_marks(
    *, intervals: pl.DataFrame, report: str, keys: list[ReportKey], labels: list[str]
) -> list[dict]:
    """Read each contrast's primary-setting row and, where present, its second-setting row.

    Args:
        intervals: `intervals.parquet`.
        report: `report.md`, which every row is checked against.
        keys: The contrasts, each as the primary setting's key.
        labels: The row label of each contrast.

    Returns:
        One dictionary per contrast.
    """
    marks = []
    for key, label in zip(keys, labels, strict=True):
        row = _intervals(intervals=intervals, key=key)
        _check_printed(report=report, texts=[_fragment(row=row)])
        second = intervals.filter(
            pl.col("setting") == SENSITIVITY_SETTING,
            pl.col("scope") == "all",
            pl.col("treatment") == key[1],
            pl.col("reference") == key[2],
        )
        second_difference = None
        if second.height == 1:
            second_row = second.row(0, named=True)
            _check_printed(report=report, texts=[_fragment(row=second_row)])
            second_difference = second_row["difference_pp"]
        marks.append(
            {
                "label": label,
                "difference": row["difference_pp"],
                "lower_95": row["lower_95_pp"],
                "upper_95": row["upper_95_pp"],
                "second_difference": second_difference,
            }
        )
    return marks


def _noise_band(*, x_domain: tuple[float, float]) -> alt.Chart:
    """Draw the shaded band of plus and minus `NOISE_FLOOR_PP` points, full height.

    Args:
        x_domain: The panel's x range, which the band's scale must repeat.

    Returns:
        The band layer.
    """
    return (
        alt.Chart(pl.DataFrame({"low": [-NOISE_FLOOR_PP], "high": [NOISE_FLOOR_PP]}))
        .mark_rect(color=ocf.DATA_BLUE_LIGHT, opacity=0.3, aria=False, tooltip=None)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", scale=alt.Scale(domain=list(x_domain), nice=False, zero=False)),
            x2="high:Q",
        )
    )


def _with_band(*, panel: alt.LayerChart, x_domain: tuple[float, float]) -> alt.LayerChart:
    """Put the noise-floor band behind a panel's layers.

    Args:
        panel: An `interval_panel` result that has no keys, so is a single layered chart.
        x_domain: The panel's x range.

    Returns:
        The panel with the band as its first layer.
    """
    return alt.LayerChart(
        layer=[_noise_band(x_domain=x_domain), *panel.layer],
        width=panel.width,
        height=panel.height,
        title=panel.title,
    )


def planned_figure(*, intervals: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw Figure 1: the three planned contrasts with their 98.33% intervals.

    Args:
        intervals: `intervals.parquet`.
        report: `report.md`.

    Returns:
        Figure 1.
    """
    marks = _contrast_marks(
        intervals=intervals,
        report=report,
        keys=list(PLANNED),
        labels=[PLANNED_LABELS[(key[1], key[2])] for key in PLANNED],
    )
    for mark, key in zip(marks, PLANNED, strict=True):
        row = _intervals(intervals=intervals, key=key)
        _check_printed(report=report, texts=[_fragment(row=row, with_adjusted=True)])
        mark["lower_95"] = row["lower_bonferroni_pp"]
        mark["upper_95"] = row["upper_bonferroni_pp"]
    frame = _panel_rows(rows=marks).with_columns(planned=pl.lit(value=True))
    x_domain = _domain(
        lows=[*frame["lower_95"].to_list(), *frame["second_difference"].to_list()],
        highs=[*frame["upper_95"].to_list(), *frame["second_difference"].to_list()],
    )
    panel = interval_panel(
        rows=frame,
        x_domain=x_domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first set better",
        figure_planning="planned",
        value_labels=True,
        row_step_px=ROW_STEP_PX,
        row_bands=True,
    )
    return figure(
        panels=[panel],
        number=1,
        figure_planning="planned",
        title=(
            "CERRA's wind direction cut the error at 100 m and at 10 m, and the two did equally "
            "well"
        ),
        subtitle=[
            "Each row: the first set of columns minus the second set, on the same rows.",
            (
                "Dot: estimate. Line: 98.33% interval from resampling whole months, "
                "adjusted for the three planned contrasts. "
                f"{HOLLOW} {CAPACITY}"
            ),
            "Negative differences mean the first set has the lower error.",
            SCOPE,
        ],
    )


def veer_figure(*, intervals: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw Figure 2: the exploratory veer contrasts and negative controls, positive controls below.

    Args:
        intervals: `intervals.parquet`.
        report: `report.md`.

    Returns:
        Figure 2.
    """
    exploratory = _contrast_marks(
        intervals=intervals,
        report=report,
        keys=[(PRIMARY_SETTING, treatment, reference) for treatment, reference in VEER_LABELS],
        labels=list(VEER_LABELS.values()),
    )
    negative = _contrast_marks(
        intervals=intervals,
        report=report,
        keys=[(PRIMARY_SETTING, treatment, reference) for treatment, reference in NEGATIVE_LABELS],
        labels=list(NEGATIVE_LABELS.values()),
    )
    positive = []
    for setting, treatment, reference, label in POSITIVE_ROWS:
        row = _intervals(intervals=intervals, key=(setting, treatment, reference))
        _check_printed(report=report, texts=[_fragment(row=row)])
        positive.append(
            {
                "label": f"{POSITIVE_PREFIX}{label}",
                "difference": row["difference_pp"],
                "lower_95": row["lower_95_pp"],
                "upper_95": row["upper_95_pp"],
            }
        )
    top = _panel_rows(rows=[*exploratory, *negative])
    bottom = _panel_rows(rows=positive)
    x_domain = _domain(
        lows=[
            *top["lower_95"].to_list(),
            *top["second_difference"].drop_nulls().to_list(),
            *bottom["lower_95"].to_list(),
        ],
        highs=[
            *top["upper_95"].to_list(),
            *top["second_difference"].drop_nulls().to_list(),
            *bottom["upper_95"].to_list(),
        ],
    )
    panels = []
    for index, (frame, title) in enumerate(
        (
            (top, "Exploratory contrasts and negative controls, primary setting"),
            (bottom, "Positive controls: a veer effect injected into the real target"),
        )
    ):
        panel = interval_panel(
            rows=frame,
            x_domain=x_domain,
            x_title=X_TITLE if index == 1 else "",
            zero_label="no difference",
            better_label="first set better",
            panel_title=title,
            reference_labels=index == 0,
            figure_planning="exploratory",
            value_labels=True,
            row_step_px=ROW_STEP_PX,
            row_bands=True,
        )
        panels.append(_with_band(panel=panel, x_domain=x_domain))  # ty: ignore[invalid-argument-type]
    return figure(
        panels=panels,
        number=2,
        figure_planning="exploratory",
        title=(
            "Direction at several heights changed the error by less than the pipeline's noise floor"
        ),
        subtitle=[
            "Each row: the first set of columns minus the second set, on the same rows.",
            f"{DOTS} {HOLLOW} {CAPACITY}",
            (
                "Shaded band: plus and minus 0.05 points, the change the negative controls "
                "produced from columns that carry no information."
            ),
            (
                "The injected veer effect cuts power by 40% (0.69 points over all rows) or by 10% "
                "(0.17 points) where the wind turns clockwise with height by 10 degrees or more."
            ),
            SCOPE,
        ],
    )


def farms_figure(*, intervals: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw Figure 3: the three planned contrasts split by wind farm.

    Args:
        intervals: `intervals.parquet`.
        report: `report.md`.

    Returns:
        Figure 3.
    """
    rows = []
    for key in PLANNED:
        label = PLANNED_LABELS[(key[1], key[2])]
        for farm in FARMS:
            row = _intervals(intervals=intervals, key=key, scope=farm)
            _check_printed(report=report, texts=[_fragment(row=row)])
            rows.append(
                {
                    "label": label,
                    "condition": farm,
                    "difference": row["difference_pp"],
                    "lower_95": row["lower_95_pp"],
                    "upper_95": row["upper_95_pp"],
                }
            )
    frame = _panel_rows(rows=rows)
    x_domain = _domain(lows=frame["lower_95"].to_list(), highs=frame["upper_95"].to_list())
    panel = interval_panel(
        rows=frame,
        x_domain=x_domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first set better",
        conditions=FARMS,
        condition_colours=FARM_COLOURS,
        condition_title="Wind farm",
        figure_planning="exploratory",
        row_step_px=FARM_ROW_STEP_PX,
        row_bands=False,
    )
    return figure(
        panels=[panel],
        number=3,
        figure_planning="exploratory",
        planning_note=("The three contrasts are planned. Their split by wind farm is exploratory."),
        title="Direction cut the error at every wind farm, and 100 m against 10 m changed sign",
        subtitle=[
            "Each row: the first set of columns minus the second set, on the same rows.",
            f"{DOTS} {CAPACITY}",
            "The intervals resample whole months, so they do not cover differences between farms.",
            SCOPE,
        ],
    )


def main() -> int:
    """Check the saved intervals against the report, then write the three SVGs.

    Returns:
        0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and check everything and build the figures in memory, but write nothing.",
    )
    arguments = parser.parse_args()
    paths = {number: ASSETS_DIR / name for number, name in FIGURE_FILES.items()}
    if not arguments.dry_run:
        refuse_to_overwrite(paths=paths.values())
    intervals = pl.read_parquet(OUTPUT_DIR / "intervals.parquet")
    report = (OUTPUT_DIR / "report.md").read_text()
    figures = {
        1: planned_figure(intervals=intervals, report=report),
        2: veer_figure(intervals=intervals, report=report),
        3: farms_figure(intervals=intervals, report=report),
    }
    for number, chart in figures.items():
        if arguments.dry_run:
            _LOG.info("dry run: figure %d built and checked, would write %s", number, paths[number])
            continue
        chart.save(paths[number])
        _LOG.info("wrote %s", paths[number])
    return 0


if __name__ == "__main__":
    sys.exit(main())
