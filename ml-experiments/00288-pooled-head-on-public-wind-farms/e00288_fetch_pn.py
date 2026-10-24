"""Fetch Elexon physical notifications (PN) for the experiment 00288 BMUs.

Runs `studies/market_downloads/fetch_bmu_notifications.py` for the BMUs in `work/bmu_list.csv`
over 2019-09-01 to 2026-06-30. The output is under `market_wind00288/elexon_pn/`.

The shared fetcher stops with `IncompleteChunkError` when a BMU has no PN for a whole month, and
some of these BMUs were not yet notifying in their first months. The run for this experiment
switched that guard off in the interpreter before calling the fetcher's `main()`, so that a
half-hour with no PN is left without a PN, and the clean-output script reports how many half-hours
of each farm that is. A script under `ml-experiments/` may not change `sys.path` or load another
script by path, so this file runs the shared fetcher unchanged. Re-running it from scratch stops
at the first BMU with an empty month until the shared fetcher gains an option to allow one.

    uv run --no-sync python ml-experiments/00288-*/e00288_fetch_pn.py

Contains BMRS data © Elexon Limited copyright and database right 2026.
"""

import subprocess
import sys
from typing import Final

from contracts.settings import PROJECT_ROOT
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
ROOT: Final = REPO_DATA_DIR / "studies/downloads/market_wind00288"
FETCHER: Final = PROJECT_ROOT / "studies/market_downloads/fetch_bmu_notifications.py"


def main() -> None:
    """Run the shared fetcher for the experiment's BMU list."""
    subprocess.run(
        [
            sys.executable,
            str(FETCHER),
            "--sources",
            "elexon_pn",
            "--bmu-list",
            str(WORK_DIR / "bmu_list.csv"),
            "--start",
            "2019-09-01",
            "--end",
            "2026-06-30",
            "--output-root",
            str(ROOT),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
