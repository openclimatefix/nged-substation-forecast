"""Fit day 5 of AIFS Single, the AIFS ENS mean, and WeatherNext 3 into one write-once folder.

One-off throwaway script that completes the leaderboards' day-5 cells for the products whose day-5
inputs are not in an earlier folder. It calls the two existing fits, `fit_aifs.run_lean` (AIFS
Single and the AIFS ENS mean, each beside ENS's mean on its own rows) and `fit_aifs.run_wn3`
(WeatherNext 3 beside ENS's mean, with its shuffled-weather negative control, the sensitivity
refits, and for wind the `ens_meanvec_day5` reference), both at day 5 only, with the settings,
seeds, folds, feature columns, and target capping of the days already fitted. It adds no fitting
code of its own.

The folder `nwp_forecast_comparison_day5_aifs_wn3` must hold `<domain>_aifs_inputs.parquet`, built
with `build_forecast_inputs.py --aifs --aifs-days 5`, and `<domain>_wn3_inputs.parquet`, built with
`build_wn3_inputs.py --build --days 5`. The two fits write their losses, predictions, and stamps
there under different file names, and their two reports as `report_aifs.md` and `report_wn3.md`.
This script joins those into `report.md`, and writes a `README.md` if none exists. It refuses any
other output folder name, so it cannot write into a folder that holds an earlier fit.

`--check` fits one arm at one wind site twice on the GPU for each of the two fits, and prints the
time and the estimated total without fitting anything else.

Every output carries only the anonymised `site` label.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import fit_aifs
from build_forecast_inputs import DAY5_OUTPUT_DIR_NAME
from studies.guards import refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DAY5: Final[tuple[int, ...]] = (5,)
"""The one lead day this script fits."""

OUTPUT_DIR_NAME: Final[str] = DAY5_OUTPUT_DIR_NAME
"""Under `data/studies/`, the only folder this script writes to."""

AIFS_REPORT_NAME: Final[str] = "report_aifs.md"
"""The AIFS fit's report, which `fit_aifs.run_lean` writes."""

WN3_REPORT_NAME: Final[str] = "report_wn3.md"
"""The WeatherNext 3 fit's report, which `fit_aifs.run_wn3` writes."""

REPORT_NAME: Final[str] = "report.md"
"""The two reports joined, written once after both fits."""

README_NAME: Final[str] = "README.md"
"""The folder's README, written if absent."""

README_TEXT: Final[str] = """# AIFS and WeatherNext 3 at day 5 (write-once)

Day-5 cells for the leaderboards of `docs/studies/forecasts/matched-lead.md`. Never overwrite a file
in this folder.

- `<domain>_aifs_inputs.parquet` and `<domain>_wn3_inputs.parquet` hold the day-5 input columns, on
  the published inputs' `(site, time)` keys.
- `<domain>_single_day5_*` and `<domain>_ens_day5_*` hold the AIFS Single and AIFS ENS mean fits,
  each beside ENS's mean on its own rows, at the primary setting. `<domain>_wn3_day5_*` holds the
  WeatherNext 3 fit beside ENS's mean, with the shuffled-weather negative control and the
  sensitivity refits, and for wind the `ens_meanvec_day5` reference.
- Each stage has `_losses.parquet` (one row per arm, setting, site, time, and seed),
  `_predictions.parquet`, and a `.json` stamp that names the device, the input files' SHA-256, and
  the settings.
- `report.md` joins `report_aifs.md` and `report_wn3.md`.
"""


def check_output_dir(*, output_dir: Path, published_dir: Path) -> None:
    """Raise unless `output_dir` is the one folder this script may write to.

    Args:
        output_dir: Where the fits would write.
        published_dir: The folder holding the published inputs.

    Raises:
        ValueError: If `output_dir` is the published folder or has a name other than
            `OUTPUT_DIR_NAME`.
    """
    if output_dir.resolve() == published_dir.resolve() or output_dir.name != OUTPUT_DIR_NAME:
        msg = f"this script writes only to a folder named {OUTPUT_DIR_NAME}, not {output_dir}"
        raise ValueError(msg)


def demoted(*, text: str) -> str:
    """Return Markdown with every heading one level lower."""
    return "\n".join(f"#{line}" if line.startswith("#") else line for line in text.splitlines())


def write_joined_report(*, output_dir: Path) -> Path:
    """Join the two fits' reports into `report.md`, once.

    Args:
        output_dir: The folder holding `report_aifs.md` and `report_wn3.md`.

    Returns:
        The written report's path.

    Raises:
        FileExistsError: If `report.md` exists.
    """
    path = output_dir / REPORT_NAME
    refuse_to_overwrite(paths=[path])
    parts = [
        "# AIFS Single, the AIFS ENS mean, and WeatherNext 3 at day 5, beside ENS's mean: report",
        "",
        *(
            demoted(text=(output_dir / name).read_text())
            for name in (AIFS_REPORT_NAME, WN3_REPORT_NAME)
        ),
        "",
    ]
    path.write_text("\n\n".join(parts))
    return path


def main() -> int:
    """Fit day 5 of the AIFS and WeatherNext 3 arms, or time one fit of each."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=fit_aifs.workers_argument,
        default=1,
        help=f"(arm, site) fits run at once, at most {fit_aifs.MAX_WORKERS}.",
    )
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    parser.add_argument(
        "--lookahead-cleared",
        action="store_true",
        help="Confirm that the run log of `build_wn3_inputs.py --read-store` and the page's "
        "lookahead section have been read.",
    )
    args = parser.parse_args()
    check_output_dir(output_dir=args.output_dir, published_dir=args.published_dir)
    fit_aifs.require_lookahead_cleared(cleared=args.lookahead_cleared)
    fit_aifs.check_gpu_visible()
    leads_day10_dir = (
        args.published_dir.resolve().parent / fit_aifs.EXTRA_FOLDERS[fit_aifs.LEAN_DAY10_FOLDER]
    )
    if args.check:
        agree_lean = fit_aifs.check_lean(
            published_dir=args.published_dir,
            lean_dir=args.output_dir,
            leads_day10_dir=leads_day10_dir,
            days=DAY5,
        )
        agree_wn3 = fit_aifs.check_wn3(
            published_dir=args.published_dir, wn3_dir=args.output_dir, days=DAY5
        )
        sys.stdout.write(f"two GPU runs agree: {agree_lean and agree_wn3}\n")
        return 0 if agree_lean and agree_wn3 else 1
    refuse_to_overwrite(paths=[args.output_dir / REPORT_NAME])
    readme = args.output_dir / README_NAME
    if not readme.exists():
        readme.write_text(README_TEXT)
    # A rerun after a crash in the second fit skips the first fit, whose report already exists;
    # each fit reuses its own saved losses after checking their stamp.
    if not (args.output_dir / AIFS_REPORT_NAME).exists():
        fit_aifs.run_lean(
            published_dir=args.published_dir,
            output_dir=args.output_dir,
            leads_day10_dir=leads_day10_dir,
            workers=args.workers,
            days=DAY5,
            report_name=AIFS_REPORT_NAME,
        )
    if not (args.output_dir / WN3_REPORT_NAME).exists():
        fit_aifs.run_wn3(
            published_dir=args.published_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            days=DAY5,
            report_name=WN3_REPORT_NAME,
        )
    write_joined_report(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
