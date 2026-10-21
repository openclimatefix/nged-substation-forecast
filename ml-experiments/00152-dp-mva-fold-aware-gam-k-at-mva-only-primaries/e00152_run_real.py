"""Run arms N, A, and F on the real readings of the 8 MVA-only Primaries (stage 2).

Usage: `e00152_run_real.py [--workers 4] [--sites D12 ...]`
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import e00152_common as common
import e00152_engine as engine

DONORS = ("D2", "D3", "D4", "D19", "D21", "D22")
ARMS = {"N": "naive", "A": "meter", "F_fold": "meter"}


def one_site(site: str) -> tuple[str, dict[str, int]]:
    """Run one site and return its row counts per period."""
    cache = common.load_cache()
    return site, engine.run_site(
        site=site, variant="real", cache=cache, kinds=("naive", "meter"), arms=ARMS, donors=DONORS
    )


def main() -> None:
    """Run the sites in parallel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="*", default=list(common.MVA_ONLY_PRIMARIES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for site, written in pool.map(one_site, args.sites):
            print(site, written, flush=True)


if __name__ == "__main__":
    main()
