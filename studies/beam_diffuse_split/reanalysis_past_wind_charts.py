"""Draw the CERRA and NORA3 past-wind page's leaderboard (Figure 1) and contrasts (Figure 2).

Reads the two write-once folders `reanalysis_past_wind.py` wrote, `wind_cerra` and `wind_nora3`,
under `past_weather_v2/`. Every drawn number is recomputed from `losses.parquet` with the same
month-and-seed resampling the fit script used, and **the script stops before drawing unless each
recomputed number rounds to the number the folder's `report.md` prints.** Nothing is refitted and
nothing is written under `data/`.

Each product is one block, scored on its own rows, so the two blocks are not comparable with each
other: CERRA's rows are 1 hour in 3, and the two blocks' ERA5 rows differ. The script also prints,
to its log, each product's contrast with ERA5 at each farm, which the page quotes. Generators
appear only as `W1` to `W3`, and no chart carries a calendar date.

Run it with `uv run python studies/beam_diffuse_split/reanalysis_past_wind_charts.py`. Optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from sources import UPDATE_OUTPUT_DIR
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

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the page's images live."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss column the fit script averages."""

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

BLOCKS_NOT_COMPARABLE: Final[str] = (
    "Compare arms only within a block: the blocks differ in rows, hours of the day, period, and "
    "fitted XGBoost models, and even their ERA5 rows differ, so no CERRA row is comparable with "
    "any NORA3 row."
)
DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole calendar months, each with all three "
    "farms' rows, and a fitting seed. The interval does not cover variation between the three "
    "farms."
)
CAPACITY: Final[str] = (
    "Errors are a fraction of each generator's 99th-percentile output, not its nameplate capacity."
)
SCOPE: Final[str] = "Three wind farms in Lincolnshire. CERRA and NORA3 are read at 100 m."
HEIGHTS: Final[str] = (
    "Wind heights: ERA5, CERRA, and NORA3 at 100 m, the ICON products at 80 m, UKV at 100 m."
)
PLANNING_NOTE: Final[str] = (
    "Planned: written into the study plan before the block's first fit, which is the new "
    "product's contrast against ERA5 and against ICON-D2. Every other row is exploratory."
)
REFERENCE_NOTE: Final[str] = "The lighter, hollow row is ERA5, scored on that block's own rows."
CONTRAST_REFERENCE_NOTE: Final[str] = (
    "Every contrast in a top panel is against ERA5, scored on that block's own rows."
)
CHANCE_NOTE: Final[str] = (
    "No correction is made for the number of exploratory contrasts, and the contrasts are "
    "correlated."
)


def _read_rows(*, report: str) -> int:
    """Read the block's number of site-hours from its report."""
    match = re.search(r"Rows scored here: ([\d,]+) \(", report)
    if match is None:
        msg = "the report has no 'Rows scored here' line"
        raise ValueError(msg)
    return int(match.group(1).replace(",", ""))


def _read_shares(*, report: str) -> tuple[float, float]:
    """Read the avoidable and one-year-only uncovered-month shares from a report."""
    match = re.search(
        r"\(avoidable\): ([\d.]+)%; the share in a calendar month seen in one year only, which "
        r"no fold design can cover: ([\d.]+)%",
        report,
    )
    if match is None:
        msg = "the report has no uncovered-month line"
        raise ValueError(msg)
    return float(match.group(1)), float(match.group(2))


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


def per_site_differences(
    *, losses: pl.DataFrame, treatment: str, reference: str
) -> dict[str, float]:
    """Return each site's mean of treatment minus reference error, in points of capacity.

    Args:
        losses: A `losses.parquet`.
        treatment: The arm whose error is the minuend.
        reference: The arm whose error is subtracted.

    Returns:
        Site label to difference, at the primary setting, averaged over seeds.
    """
    pivot = (
        losses.filter(pl.col("setting") == PRIMARY, pl.col("arm").is_in([treatment, reference]))
        .pivot(on="arm", index=["site", "time", "seed"], values=METRIC)
        .with_columns(difference=pl.col(treatment) - pl.col(reference))
        .group_by("site")
        .agg(pl.col("difference").mean() * PERCENTAGE_POINTS)
        .sort("site")
    )
    return dict(pivot.iter_rows())


def _build(*, product: Product) -> tuple[RowSetBlock, RowSetBlock, tuple[float, float]]:
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
    contrasts = primary.with_columns(second_difference=second["difference"])
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
    sites = per_site_differences(losses=losses, treatment=product.new_arm, reference=REFERENCE_ARM)
    _LOG.info(
        "%s minus ERA5 by site, points of capacity: %s",
        product.label,
        {site: round(value, 2) for site, value in sites.items()},
    )
    if product.key == "nora3":
        against_global = per_site_differences(
            losses=losses, treatment=product.new_arm, reference="icon_global_wind"
        )
        _LOG.info(
            "NORA3 minus ICON global by site, points of capacity: %s",
            {site: round(value, 2) for site, value in against_global.items()},
        )
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
        _read_shares(report=report),
    )


def _narrow(*, lines: list[str]) -> list[str]:
    """Wrap each caption line, which keeps it inside the figure's edge."""
    return [piece for line in lines for piece in wrapped(text=line, width=CAPTION_CHARACTERS)]


def _share_lines(*, shares: dict[str, tuple[float, float]]) -> list[str]:
    """State each block's uncovered-month shares, as the Methods page defines them."""
    return [
        f"{label}: {avoidable:.1f}% of scored rows are in a calendar month, seen in two or more "
        f"years, with no training row in their fold, and {one_year:.1f}% are in a calendar month "
        "seen in one year only, which no fold design can cover."
        for label, (avoidable, one_year) in shares.items()
    ]


def main() -> int:
    """Check each block against its report, then write the two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    built = [_build(product=product) for product in PRODUCTS]
    leaderboard_blocks = [pair[0] for pair in built]
    contrast_blocks = [pair[1] for pair in built]
    shares = {product.label: pair[2] for product, pair in zip(PRODUCTS, built, strict=True)}
    leaderboard = stacked_leaderboard(
        blocks=leaderboard_blocks,
        reference_note=REFERENCE_NOTE,
        number=1,
        title="At three wind farms, ICON-D2's wind has the lowest error in both blocks",
        subtitle=_narrow(
            lines=[
                "Each arm's own mean absolute error, sorted best first within its block.",
                BLOCKS_NOT_COMPARABLE,
                (
                    "Overlapping intervals do not show that two arms are equal: every arm's error "
                    "swings together from month to month, a swing that Figure 2's paired "
                    "contrasts cancel."
                ),
                *_share_lines(shares=shares),
                HEIGHTS,
                DOTS,
                CAPACITY,
                SCOPE,
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
                    "Top panel of each block: each arm's mean absolute error minus ERA5's. Lower "
                    "panel: that block's two planned contrasts, the first arm's error minus the "
                    "second's."
                ),
                "Each arm's own error is in Figure 1.",
                BLOCKS_NOT_COMPARABLE,
                CHANCE_NOTE,
                *_share_lines(shares=shares),
                HEIGHTS,
                DOTS,
                CAPACITY,
                SCOPE,
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
