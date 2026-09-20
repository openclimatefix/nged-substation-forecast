"""Compare two irradiance sources on the hours they both cover.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Which source predicts power better is a different question from whether its split helps, and the
arm-to-arm tables cannot answer it**, because each source's run is scored on its own rows: the CAMS
reliability flag removes hours ERA5 keeps, so the two runs' pooled errors are not measured on the
same weather. This script restricts both runs to the hours and sites they share and reports the
global-irradiance-only arm on those rows alone, which is the comparison the write-up can make.

Run it with `uv run --no-project --with polars python
scripts/experiments/beam_diffuse_split/compare_sources.py --alignment shifted`.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import polars as pl

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")

PERCENTAGE_POINTS: Final[float] = 100.0

COMPARED_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "xgboost": ("A_global_only", "B_erbs", "C_era5_split"),
    "physics": ("P_A_global_only", "P_B_erbs", "P_C_source_split"),
}
"""The arms reported for each instrument, in the order the table prints them."""


def _losses_for(*, instrument: str, source: str, alignment: str) -> pl.DataFrame:
    """Read one run's per-row losses, restricted to the primary setting."""
    stem = "results" if instrument == "xgboost" else "physics"
    path = (
        REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_{stem}_{source}_{alignment}"
        / "per_row_losses.parquet"
    )
    return pl.read_parquet(path).filter(pl.col("setting") == "primary")


def main() -> int:
    """Print the shared-hours comparison for both instruments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", choices=("as-labelled", "shifted"), default="shifted")
    parser.add_argument("--first-source", default="cams")
    parser.add_argument("--second-source", default="open-meteo")
    arguments = parser.parse_args()

    lines: list[str] = []
    for instrument, arms in COMPARED_ARMS.items():
        first = _losses_for(
            instrument=instrument, source=arguments.first_source, alignment=arguments.alignment
        )
        second = _losses_for(
            instrument=instrument, source=arguments.second_source, alignment=arguments.alignment
        )
        shared = (
            first.select("site", "time")
            .unique()
            .join(second.select("site", "time").unique(), on=["site", "time"], how="inner")
        )
        lines += [
            f"### {instrument}, {arguments.alignment} stamps",
            "",
            f"{shared.height:,} hours shared by both sources.",
            "",
            (
                "| Arm | "
                f"{arguments.first_source} MAE (% of capacity) | "
                f"{arguments.second_source} MAE (% of capacity) |"
            ),
            "|---|---|---|",
        ]
        for arm in arms:
            cells = []
            for losses in (first, second):
                scoped = losses.filter(pl.col("arm") == arm).join(
                    shared, on=["site", "time"], how="semi"
                )
                cells.append(
                    float(scoped["absolute_error_fraction_of_capacity"].mean()) * PERCENTAGE_POINTS
                )
            lines.append(f"| {arm} | {cells[0]:.3f} | {cells[1]:.3f} |")
        lines.append("")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
