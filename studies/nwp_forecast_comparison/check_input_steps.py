"""V1c: look for steps in the planned inputs' served values, month to month.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>.

**Method.** For each planned Previous Runs input other than the reference (UKV and IFS 0.25°), this
takes the monthly mean of its day-1 global horizontal irradiance and 100 m wind speed, per
anonymised site, and divides it by ICON-EU's day-1 monthly mean at the same site and month —
ICON-EU is the reference because it has the longest continuous Previous Runs history among the
planned inputs and covers the whole trial area. A step in one product but not in the reference
shows up as a jump in the ratio that persists. GEFS and ENS are not compared here: GEFS's extract
is not downloaded yet, and ENS is read through `ens_forecast_horizons.py`'s different schema, which
`build_forecast_inputs.py` joins onto Previous Runs rather than this screen.

**This is a screen, not a statistical test.** A jump flagged here is a candidate for a known upgrade
date (recorded in the study's README) or a genuine defect; either way a flagged jump is a reason to
look at the month, not a verdict on it.

No metered generator's name, identifier or coordinate appears anywhere in this script or its
output: every value already carries only the anonymised `site` label.

Run it with `uv run python studies/nwp_forecast_comparison/check_input_steps.py --output-dir <dir>`.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Final

import polars as pl
from contracts.settings import PROJECT_ROOT

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

REFERENCE_PRODUCT: Final[str] = "ICON-EU"
"""Every other planned input's ratio is read against this product's own day-1 monthly mean."""

COMPARED_PRODUCTS: Final[tuple[str, ...]] = ("UKV", "ECMWF-IFS-025")
"""The Previous Runs planned inputs compared against `REFERENCE_PRODUCT`. GEFS is not compared: its
extract is not downloaded yet, and V2 checks its own steps separately."""

FIELDS: Final[dict[str, str]] = {
    "ghi": "shortwave_radiation_previous_day1",
    "wind_speed_100m": "wind_speed_100m_previous_day1",
}

STEP_RATIO_THRESHOLD: Final[float] = 1.15
"""A month-to-month change in a site's ratio to the reference beyond this factor is flagged."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    See `verify_previous_runs_leads._repo_data_dir` for the reasoning; duplicated here because
    study scripts in different directories cannot import one another's private helpers.

    Returns:
        The directory holding `studies/`, `NGED/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def _monthly_mean(*, product_dir: str, field_column: str) -> pl.DataFrame:
    """Return one product's monthly mean of one field, per site.

    Args:
        product_dir: The product's directory name under `data/studies/weather/`.
        field_column: The Previous Runs column to average.

    Returns:
        Rows of `site`, `month` (a `%Y-%m` string) and `value`.
    """
    path = (
        _repo_data_dir()
        / "studies"
        / "weather"
        / product_dir
        / "previous_runs"
        / "combined.parquet"
    )
    return (
        pl.scan_parquet(path)
        .select("site", "time", value=pl.col(field_column))
        .drop_nulls()
        .with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("site", "month")
        .agg(value=pl.col("value").mean())
        .collect()
    )


def _step_lines(*, ratios: pl.DataFrame, product: str, field: str) -> list[str]:
    """Flag month-to-month jumps in a product's ratio to the reference, per site.

    Args:
        ratios: Rows of `site`, `month`, `ratio` for one product and field.
        product: The compared product's name, for the table.
        field: The field's name, for the table.

    Returns:
        One markdown table row per flagged (site, month) jump, in site-month order.
    """
    lines = []
    for site in sorted(ratios["site"].unique().to_list()):
        site_rows = ratios.filter(pl.col("site") == site).sort("month")
        months = site_rows["month"].to_list()
        values = site_rows["ratio"].to_list()
        for previous, current, month in zip(values[:-1], values[1:], months[1:], strict=True):
            if previous <= 0 or current <= 0:
                continue
            change = max(current / previous, previous / current)
            if change >= STEP_RATIO_THRESHOLD:
                lines.append(
                    f"| {product} | {field} | {site} | {month} | {previous:.3f} → {current:.3f} |"
                )
    return lines


def main() -> int:
    """Compute and flag monthly steps in each planned input's ratio to ICON-EU, per site."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Product | Field | Site | Month | Ratio to ICON-EU (prev → this) |",
        "|---|---|---|---|---|",
    ]
    for field_name, column in FIELDS.items():
        reference = _monthly_mean(product_dir=REFERENCE_PRODUCT, field_column=column)
        for product in COMPARED_PRODUCTS:
            product_frame = _monthly_mean(product_dir=product, field_column=column)
            joined = product_frame.join(
                reference.rename({"value": "reference_value"}), on=["site", "month"], how="inner"
            ).with_columns(ratio=pl.col("value") / pl.col("reference_value"))
            lines += _step_lines(ratios=joined, product=product, field=field_name)

    if len(lines) == 2:
        lines.append(f"\nNo month-to-month change at or above {STEP_RATIO_THRESHOLD}x found.")
    (args.output_dir / "v1c_steps.md").write_text("\n".join(lines) + "\n")
    _LOG.info("V1c: %d candidate steps flagged.", max(len(lines) - 2, 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
