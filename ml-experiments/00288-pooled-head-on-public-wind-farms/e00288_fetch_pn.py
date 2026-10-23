"""Fetch Elexon physical notifications (PN) for the experiment 00288 BMUs.

Runs `studies/market_downloads/fetch_bmu_notifications.py` for the BMUs in `work/bmu_list.csv`
over 2019-09-01 to 2026-06-30, with that script's whole-empty-month guard switched off. The guard
stops a batch when a BMU has no PN for a whole month, and some of these BMUs were not yet notifying
in their first months. A half-hour with no PN is left without a PN, and the clean-output script
reports how many half-hours of each farm that is. The output is under
`market_wind00288/elexon_pn/`.

    uv run --no-sync python ml-experiments/00288-*/e00288_fetch_pn.py

Contains BMRS data © Elexon Limited copyright and database right 2026.
"""

import importlib
import sys
from typing import Final

from contracts.settings import PROJECT_ROOT
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
ROOT: Final = REPO_DATA_DIR / "studies/downloads/market_wind00288"
FETCHER_DIR: Final = PROJECT_ROOT / "studies/market_downloads"


def main() -> None:
    """Run the fetcher with its whole-empty-month guard replaced by a no-op."""
    sys.path.insert(0, str(FETCHER_DIR))
    fetcher = importlib.import_module("fetch_bmu_notifications")
    fetcher.check_no_empty_month = lambda **_: None
    sys.argv = [
        "fetch_bmu_notifications.py",
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
    ]
    fetcher.main()


if __name__ == "__main__":
    main()
