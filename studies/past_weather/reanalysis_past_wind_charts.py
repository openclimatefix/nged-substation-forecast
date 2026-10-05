"""Draw the CERRA and NORA3 past-wind page's leaderboard (Figure 1) and contrasts (Figure 2).

Reads the two write-once folders `reanalysis_past_wind.py` wrote, `wind_cerra` and `wind_nora3`,
under `past_weather_v2/`. Every drawn number is recomputed from `losses.parquet` with the same
month-and-seed resampling the fit script used, and **the script stops before drawing unless each
recomputed number rounds to the number the folder's `report.md` prints.** Nothing is refitted and
nothing is written under `data/` except each `per_site.md`.

Each product is one block, scored on its own rows, so the two blocks are not comparable with each
other: CERRA's rows are 1 hour in 3, and the two blocks' ERA5 rows differ. The script also writes
a new `per_site.md` beside each report, holding the contrasts the page quotes per farm, each with
its 95% interval from resampling months within that farm, and the three drawn contrasts the fit
report does not print (UKV, ICON-EU, and ICON global minus ERA5). `_check_contrasts` skips those
three, because the report prints only the contrasts it names, so `per_site.md` is where a committed
script prints them. It never touches `losses.parquet`, `losses.fingerprint`, or `report.md`.
Generators appear only as `W1` to `W3`, and no chart carries a calendar date.

Run it with `uv run python studies/past_weather/reanalysis_past_wind_charts.py`. Optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from studies.bootstrap import BootstrapInterval, bootstrap_difference
from studies.charts import (
    PERCENTAGE_POINTS,
    BlockArm,
    PlannedContrast,
    RowSetBlock,
    assert_matches_printed,
    block_contrast_rows,
    block_leaderboard_rows,
    planned_contrast_rows,
    report_contrasts,
    report_errors,
    stacked_contrasts,
    stacked_leaderboard,
    wrapped,
)
from studies.sources import UPDATE_OUTPUT_DIR

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the page's images live."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss column the fit script averages."""

PER_SITE_FILE: Final[str] = "per_site.md"
"""The new file written beside each fit report."""

PRIMARY: Final[str] = "pooled"
SECOND: Final[str] = "sensitivity"
ERROR_COLUMN: Final[str] = "All sites"
"""The header of the report's error column."""

CAPTION_CHARACTERS: Final[int] = 100


class Product(NamedTuple):
    """One block: a reanalysis scored against the main study's five products."""

    key: str
    label: str
    new_arm: str
    new_label: str
    extra: tuple[BlockArm, ...]
    folder: Path


def _arm(*, arm: str, label: str, family: str, planned: bool = False) -> BlockArm:
    """Return a block arm; the family string is one of `studies.charts`' product families."""
    return BlockArm(arm=arm, label=label, family=family, planned=planned)  # ty: ignore[invalid-argument-type]


def _arms(*, product: Product) -> tuple[BlockArm, ...]:
    """Return the block's arms: ERA5 (the reference), the four weather models, then the new one."""
    return (
        _arm(arm="era5_wind", label="ERA5", family="reanalysis"),
        _arm(arm="ukv_wind", label="UKV", family="weather model"),
        _arm(arm="icon_d2_wind", label="ICON-D2", family="weather model"),
        _arm(arm="icon_eu_wind", label="ICON-EU", family="weather model"),
        _arm(arm="icon_global_wind", label="ICON global", family="weather model"),
        _arm(arm=product.new_arm, label=product.new_label, family="reanalysis", planned=True),
        *product.extra,
    )


PRODUCTS: Final[tuple[Product, ...]] = (
    Product(
        key="cerra",
        label="CERRA",
        new_arm="cerra_wind",
        new_label="CERRA 100 m",
        extra=(),
        folder=UPDATE_OUTPUT_DIR / "wind_cerra",
    ),
    Product(
        key="nora3",
        label="NORA3",
        new_arm="nora3_wind",
        new_label="NORA3 100 m",
        extra=(_arm(arm="nora3_50m_wind", label="NORA3 50 m", family="reanalysis"),),
        folder=UPDATE_OUTPUT_DIR / "wind_nora3",
    ),
)
"""The two blocks, CERRA first."""

REFERENCE_ARM: Final[str] = "era5_wind"
D2_ARM: Final[str] = "icon_d2_wind"
UNPRINTED_ARMS: Final[tuple[tuple[str, str], ...]] = (
    ("ukv_wind", "UKV"),
    ("icon_eu_wind", "ICON-EU"),
    ("icon_global_wind", "ICON global"),
)
"""The arms whose contrast with ERA5 Figure 2 draws and the fit report does not print."""

BLOCKS_NOT_COMPARABLE: Final[str] = (
    "The two blocks are scored on different rows, so compare arms only within a block."
)
DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling calendar months and a fitting seed."
)
CAPACITY: Final[str] = (
    "Errors are a percentage of each farm's 99th-percentile output, not its nameplate capacity."
)
PLANNING_NOTE: Final[str] = (
    "Planned: written into the study plan before the block's first fit, which is the new "
    "product's contrast against ERA5 and against ICON-D2. Every other row is exploratory."
)
REFERENCE_NOTE: Final[str] = "The lighter, hollow row is ERA5, scored on that block's own rows."
CONTRAST_REFERENCE_NOTE: Final[str] = (
    "Every contrast in a top panel is against ERA5, scored on that block's own rows."
)
CHANCE_NOTE: Final[str] = "No correction is made for the number of exploratory contrasts."


def _read_rows(*, report: str) -> int:
    """Read the block's number of site-hours from its report."""
    match = re.search(r"Rows scored here: ([\d,]+) \(", report)
    if match is None:
        msg = "the report has no 'Rows scored here' line"
        raise ValueError(msg)
    return int(match.group(1).replace(",", ""))


def _check_contrasts(
    *, rows: pl.DataFrame, reference: str, report_path: Path, setting: str
) -> None:
    """Stop unless each drawn difference matches the report's all-sites row at that setting."""
    scope = "all" if setting == PRIMARY else "sensitivity"
    printed = report_contrasts(report_path=report_path).filter(
        pl.col("scope").str.starts_with(scope)
    )
    for row in rows.iter_rows(named=True):
        hit = printed.filter(
            pl.col("treatment") == row["arm"],
            pl.col("reference") == row.get("reference_arm", reference),
        )
        if hit.is_empty():
            continue  # the report prints only the contrasts it names
        assert_matches_printed(
            name=f"{row['arm']} at {setting}",
            recomputed=row["difference"],
            printed=hit["difference"][0],
        )


def per_site_interval(
    *, losses: pl.DataFrame, site: str, treatment: str, reference: str
) -> BootstrapInterval:
    """Return one site's treatment-minus-reference difference and its 95% interval.

    The interval resamples whole calendar months and a fitting seed within the one site, at the
    primary setting.

    Args:
        losses: A `losses.parquet`.
        site: The anonymised site label, such as `W1`.
        treatment: The arm whose error is the minuend.
        reference: The arm whose error is subtracted.

    Returns:
        The bootstrap result, with `difference`, `lower_95` and `upper_95` in points of capacity.
    """
    at_site = losses.filter(pl.col("setting") == PRIMARY, pl.col("site") == site)
    interval = bootstrap_difference(
        losses=at_site, treatment=treatment, reference=reference, metric=METRIC
    )
    return {
        **interval,
        "difference": interval["difference"] * PERCENTAGE_POINTS,
        "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
        "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
    }


def _signed(*, value: float) -> str:
    """Format a value to three decimals with an explicit sign, as the fit reports do."""
    return f"{value:+.3f}"


def per_site_report(*, product: Product, losses: pl.DataFrame) -> str:
    """Return the text of one block's `per_site.md`.

    Args:
        product: The block.
        losses: The block's `losses.parquet`.

    Returns:
        Markdown holding each quoted per-farm contrast with its interval, then the drawn contrasts
        against ERA5 that the fit report does not print.
    """
    sites = sorted(losses["site"].unique().to_list())
    contrasts = [
        (product.new_arm, REFERENCE_ARM, product.new_label, "ERA5"),
        *(
            [(product.new_arm, "icon_global_wind", product.new_label, "ICON global")]
            if product.key == "nora3"
            else []
        ),
    ]
    lines = [
        f"## {product.label}: contrasts by farm",
        "",
        (
            "Points of capacity at the primary setting. Each interval is a 95% bound from "
            "resampling whole calendar months and a fitting seed within the one farm, so it does "
            "not cover differences between farms."
        ),
        "",
        "| Contrast | Farm | Difference | 95% interval |",
        "|---|---|---|---|",
    ]
    for treatment, reference, treatment_label, reference_label in contrasts:
        for site in sites:
            hit = per_site_interval(
                losses=losses, site=site, treatment=treatment, reference=reference
            )
            lines.append(
                f"| {treatment_label} minus {reference_label} | {site} | "
                f"{_signed(value=hit['difference'])} | "
                f"[{_signed(value=hit['lower_95'])}, {_signed(value=hit['upper_95'])}] |"
            )
    lines += [
        "",
        f"## {product.label}: drawn contrasts the fit report does not print",
        "",
        "All farms, at the primary setting, in points of capacity.",
        "",
        "| Contrast | Difference | 95% interval | Rows |",
        "|---|---|---|---|",
    ]
    at_primary = losses.filter(pl.col("setting") == PRIMARY)
    for arm, label in UNPRINTED_ARMS:
        hit = bootstrap_difference(
            losses=at_primary, treatment=arm, reference=REFERENCE_ARM, metric=METRIC
        )
        lines.append(
            f"| {label} minus ERA5 | {_signed(value=hit['difference'] * PERCENTAGE_POINTS)} | "
            f"[{_signed(value=hit['lower_95'] * PERCENTAGE_POINTS)}, "
            f"{_signed(value=hit['upper_95'] * PERCENTAGE_POINTS)}] | {hit['n_rows']:,} |"
        )
    return "\n".join(lines) + "\n"


def _build(*, product: Product) -> tuple[RowSetBlock, RowSetBlock]:
    """Build one product's leaderboard block and contrast block, checked against its report."""
    report_path = product.folder / "report.md"
    report = report_path.read_text()
    losses = pl.read_parquet(product.folder / "losses.parquet")
    site_hours = _read_rows(report=report)
    arms = _arms(product=product)
    printed = report_errors(report_path=report_path, column=ERROR_COLUMN)
    by_arm = {arm.arm: arm for arm in arms}
    renamed = {arm.arm: printed[arm.arm] for arm in arms}
    leaderboard = block_leaderboard_rows(
        losses=losses,
        arms=arms,
        setting=PRIMARY,
        site_hours=site_hours,
        metric=METRIC,
        printed=renamed,
    )
    leaderboard = leaderboard.with_columns(reference=pl.col("arm") == REFERENCE_ARM)
    contrast_arms = tuple(arm for arm in arms if arm.arm != REFERENCE_ARM)
    primary = block_contrast_rows(
        losses=losses,
        arms=contrast_arms,
        reference_arm=REFERENCE_ARM,
        setting=PRIMARY,
        site_hours=site_hours,
        metric=METRIC,
    )
    second = block_contrast_rows(
        losses=losses,
        arms=contrast_arms,
        reference_arm=REFERENCE_ARM,
        setting=SECOND,
        site_hours=site_hours,
        metric=METRIC,
    )
    _check_contrasts(
        rows=primary, reference=REFERENCE_ARM, report_path=report_path, setting=PRIMARY
    )
    _check_contrasts(rows=second, reference=REFERENCE_ARM, report_path=report_path, setting=SECOND)
    contrasts = primary.with_columns(second_difference=second["difference"]).sort("difference")
    pairs = (
        PlannedContrast(by_arm[product.new_arm], by_arm[REFERENCE_ARM]),
        PlannedContrast(by_arm[product.new_arm], by_arm[D2_ARM]),
    )
    planned_primary = planned_contrast_rows(
        losses=losses, contrasts=pairs, setting=PRIMARY, site_hours=site_hours, metric=METRIC
    )
    planned_second = planned_contrast_rows(
        losses=losses, contrasts=pairs, setting=SECOND, site_hours=site_hours, metric=METRIC
    )
    _check_contrasts(
        rows=planned_primary, reference=REFERENCE_ARM, report_path=report_path, setting=PRIMARY
    )
    _check_contrasts(
        rows=planned_second, reference=REFERENCE_ARM, report_path=report_path, setting=SECOND
    )
    planned = planned_primary.with_columns(second_difference=planned_second["difference"])
    (product.folder / PER_SITE_FILE).write_text(per_site_report(product=product, losses=losses))
    dates = "Aug 2024 to Jun 2026" if product.key == "cerra" else "Aug 2024 to Aug 2026"
    return (
        RowSetBlock(
            label=product.label,
            dates=dates,
            site_hours=site_hours,
            rows=leaderboard,
            hours_unit="farm-hours",
        ),
        RowSetBlock(
            label=product.label,
            dates=dates,
            site_hours=site_hours,
            rows=contrasts,
            planned_rows=planned,
            hours_unit="farm-hours",
        ),
    )


def _narrow(*, lines: list[str]) -> list[str]:
    """Wrap each caption line, which keeps it inside the figure's edge."""
    return [piece for line in lines for piece in wrapped(text=line, width=CAPTION_CHARACTERS)]


def main() -> int:
    """Check each block against its report, then write the two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    built = [_build(product=product) for product in PRODUCTS]
    leaderboard_blocks = [pair[0] for pair in built]
    contrast_blocks = [pair[1] for pair in built]
    leaderboard = stacked_leaderboard(
        blocks=leaderboard_blocks,
        reference_note=REFERENCE_NOTE,
        number=1,
        title="At three wind farms, ICON-D2's wind has the lowest estimated error in both blocks",
        subtitle=_narrow(
            lines=[
                "Each arm's mean absolute error, sorted best first within its block. " + CAPACITY,
                BLOCKS_NOT_COMPARABLE,
                (
                    "Overlapping intervals do not show that two arms are equal: Figure 2's paired "
                    "contrasts cancel the swing the arms share from month to month."
                ),
                DOTS,
            ]
        ),
    )
    contrasts = stacked_contrasts(
        blocks=contrast_blocks,
        number=2,
        title=(
            "At three wind farms, CERRA's wind has a higher error than ERA5's and NORA3's "
            "does not clearly differ"
        ),
        subtitle=_narrow(
            lines=[
                (
                    "Top panel of each block: each arm's error minus ERA5's, sorted. Lower panel: "
                    "the block's two planned contrasts, the first arm minus the second."
                ),
                BLOCKS_NOT_COMPARABLE,
                CHANCE_NOTE,
                DOTS,
            ]
        ),
        planning_note=PLANNING_NOTE,
        reference_note=CONTRAST_REFERENCE_NOTE,
    )
    for name, chart in {
        "reanalysis_wind_leaderboard": leaderboard,
        "reanalysis_wind_contrasts": contrasts,
    }.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
