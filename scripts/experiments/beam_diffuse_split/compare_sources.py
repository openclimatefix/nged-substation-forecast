"""Compare two irradiance sources on the hours they both cover.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Which source predicts power better is a different question from whether its split helps, and the
per-source tables cannot answer it**, because each source's run is scored on its own rows: the CAMS
reliability flag removes hours ERA5 keeps, so the two runs' pooled errors are not measured on the
same weather. This script restricts both runs to the hours and sites they share and reports, on
those rows alone, each arm's mean absolute error and each contrast's bootstrap interval.

**Restricting the scoring does not restrict the training.** Each source's models were fitted on that
source's own rows, so a difference that survives here is still a difference between two pipelines
rather than between two grids alone. That confound cannot be removed without refitting one source on
the other's rows, which is not what any of the arms are for.

Run it with `uv run --no-project --with polars --with numpy --with xgboost python
scripts/experiments/beam_diffuse_split/compare_sources.py --alignment shifted`.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import polars as pl
from run_experiment import _bootstrap_difference

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")

PERCENTAGE_POINTS: Final[float] = 100.0

COMPARED_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "xgboost": ("A_global_only", "B_erbs", "C_era5_split"),
    "physics": ("P_A_global_only", "P_B_erbs", "P_C_source_split"),
}
"""The arms reported for each instrument, in the order the table prints them."""

COMPARED_CONTRASTS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "xgboost": (("C_era5_split", "B_erbs"), ("C_era5_split", "A_global_only")),
    "physics": (("P_C_source_split", "P_B_erbs"), ("P_C_source_split", "P_A_global_only")),
}
"""The contrasts recomputed on the shared rows, headline first."""


def _scalar(value: object) -> float:
    """Narrow a Polars aggregate to a plain float.

    `Series.mean()` is typed as a union covering every dtype a Series could hold, so a checker
    cannot know this column is a float.
    """
    return float(value)  # ty: ignore[invalid-argument-type]


def _losses_for(*, instrument: str, source: str, alignment: str) -> pl.DataFrame:
    """Read one run's per-row losses, restricted to the primary setting.

    Args:
        instrument: `xgboost` or `physics`.
        source: The irradiance source the run used.
        alignment: The stamp alignment the run used.

    Returns:
        The primary setting's per-row losses.

    Raises:
        FileNotFoundError: If that run has not been produced.
    """
    stem = "results" if instrument == "xgboost" else "physics"
    path = (
        REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_{stem}_{source}_{alignment}"
        / "per_row_losses.parquet"
    )
    if not path.exists():
        msg = f"{path} missing; run the {instrument} instrument on {source} first"
        raise FileNotFoundError(msg)
    return pl.read_parquet(path).filter(pl.col("setting") == "primary")


def main() -> int:
    """Print the shared-hours comparison for both instruments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", choices=("as-labelled", "shifted"), default="shifted")
    parser.add_argument("--first-source", default="cams")
    parser.add_argument("--second-source", default="open-meteo")
    arguments = parser.parse_args()
    sources = (arguments.first_source, arguments.second_source)

    lines: list[str] = []
    for instrument, arms in COMPARED_ARMS.items():
        runs = {
            source: _losses_for(instrument=instrument, source=source, alignment=arguments.alignment)
            for source in sources
        }
        shared = (
            runs[sources[0]]
            .select("site", "time")
            .unique()
            .join(
                runs[sources[1]].select("site", "time").unique(), on=["site", "time"], how="inner"
            )
        )
        restricted = {
            source: losses.join(shared, on=["site", "time"], how="semi")
            for source, losses in runs.items()
        }

        lines += [
            f"### {instrument}, {arguments.alignment} stamps",
            "",
            f"{shared.height:,} hours shared by both sources.",
            "",
            f"| Arm | {sources[0]} MAE (% of P99 output) | {sources[1]} MAE (% of P99 output) |",
            "|---|---|---|",
        ]
        for arm in arms:
            cells = [
                _scalar(
                    restricted[source]
                    .filter(pl.col("arm") == arm)["absolute_error_fraction_of_capacity"]
                    .mean()
                )
                * PERCENTAGE_POINTS
                for source in sources
            ]
            lines.append(f"| {arm} | {cells[0]:.3f} | {cells[1]:.3f} |")

        lines += [
            "",
            (
                f"| Contrast | {sources[0]} ΔMAE (pp) | 95% interval "
                f"| {sources[1]} ΔMAE (pp) | 95% interval |"
            ),
            "|---|---|---|---|---|",
        ]
        for treatment, reference in COMPARED_CONTRASTS[instrument]:
            cells = []
            for source in sources:
                interval = _bootstrap_difference(
                    losses=restricted[source],
                    treatment=treatment,
                    reference=reference,
                    metric="absolute_error_fraction_of_capacity",
                )
                cells.append(
                    f"{interval['difference'] * PERCENTAGE_POINTS:+.4f} | "
                    f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.4f}, "
                    f"{interval['upper_95'] * PERCENTAGE_POINTS:+.4f}]"
                )
            lines.append(f"| {treatment} − {reference} | {cells[0]} | {cells[1]} |")
        lines.append("")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
