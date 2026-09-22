"""Split the headline contrast by how high the sun was.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**A beam flux measured on a horizontal plane has to be divided by the cosine of the solar zenith
angle before it can be projected onto a tilted panel, and that division amplifies whatever error the
flux carries.** Near sunrise and sunset the divisor is small, so a published beam field that is only
slightly wrong on the horizontal can be badly wrong once transposed. If the arm given the product's
own split loses only at low sun, the finding is about the transposition amplifying an error; if it
loses at every elevation, the finding is about the field carrying less information.

The band edges are fixed here and the split is applied after the fact to losses that were produced
without knowledge of it, so no arm's model was fitted differently because of the band.

Run it with `uv run --no-project --with polars python
scripts/experiments/beam_diffuse_split/elevation_breakdown.py --source cams --alignment shifted`.
"""

import argparse
import sys
from typing import Final

import polars as pl
from sources import REPO_DATA_DIR, SOURCE_CHOICES

PERCENTAGE_POINTS: Final[float] = 100.0

ELEVATION_BAND_EDGES_DEGREES: Final[tuple[float, ...]] = (0.0, 10.0, 20.0, 30.0, 90.0)
"""The solar elevation bands the contrast is reported in, in degrees."""

CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "xgboost": ("C_era5_split", "B_erbs"),
    "physics": ("P_C_source_split", "P_B_erbs"),
}
"""Each instrument's headline contrast, as (treatment, reference)."""


def _banded_losses(*, instrument: str, source: str, alignment: str) -> pl.DataFrame:
    """Read one run's primary losses and attach the solar elevation band of each row."""
    stem = "results" if instrument == "xgboost" else "physics"
    losses = pl.read_parquet(
        REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_{stem}_{source}_{alignment}"
        / "per_row_losses.parquet"
    ).filter(pl.col("setting") == "primary")
    elevation = pl.read_parquet(
        REPO_DATA_DIR / "ERA5" / f"beam_diffuse_dataset_{source}_{alignment}.parquet"
    ).select("site", "time", "solar_elevation_deg")
    return losses.join(elevation, on=["site", "time"], how="inner").with_columns(
        band=pl.col("solar_elevation_deg").cut(
            breaks=list(ELEVATION_BAND_EDGES_DEGREES[1:-1]), labels=None
        )
    )


def main() -> int:
    """Print each instrument's headline contrast inside every elevation band."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    parser.add_argument(
        "--alignment", choices=("as-labelled", "shifted", "piecewise"), default="piecewise"
    )
    arguments = parser.parse_args()

    lines: list[str] = []
    for instrument, (treatment, reference) in CONTRASTS.items():
        banded = _banded_losses(
            instrument=instrument, source=arguments.source, alignment=arguments.alignment
        )
        paired = (
            banded.filter(pl.col("arm") == reference)
            .select("site", "time", "seed", "band", reference=pl.col("absolute_error_capped_mw"))
            .join(
                banded.filter(pl.col("arm") == treatment).select(
                    "site",
                    "time",
                    "seed",
                    "effective_capacity_mw",
                    treatment=pl.col("absolute_error_capped_mw"),
                ),
                on=["site", "time", "seed"],
                how="inner",
            )
            .group_by("band")
            .agg(
                difference=(
                    (pl.col("treatment") - pl.col("reference")) / pl.col("effective_capacity_mw")
                ).mean(),
                reference_mae=(pl.col("reference") / pl.col("effective_capacity_mw")).mean(),
                n_rows=pl.len(),
            )
            .sort("band")
        )
        lines += [
            f"### {instrument}, {arguments.source}, {arguments.alignment} stamps",
            "",
            (
                "| Solar elevation (degrees) | ΔMAE (pp of P99 output) |"
                " Reference MAE (% of P99 output) | Relative | Hours |"
            ),
            "|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {row['band']} | {row['difference'] * PERCENTAGE_POINTS:+.4f} | "
            f"{row['reference_mae'] * PERCENTAGE_POINTS:.3f} | "
            f"{row['difference'] / row['reference_mae'] * PERCENTAGE_POINTS:+.2f}% | "
            f"{row['n_rows']:,} |"
            for row in paired.iter_rows(named=True)
        )
        lines.append("")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
