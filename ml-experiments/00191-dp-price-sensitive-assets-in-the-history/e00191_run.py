"""Run experiment 191's arms at one or more sites and write the prediction frames.

Run it under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` with
`uv run --no-sync python`, for example with the arguments `--sites D8 D14 [--ungated]`.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import e00191_common as common
import e00191_engine as engine


def one_site(job: tuple[str, bool, str]) -> tuple[str, dict[str, int]]:
    """Run one site and return its row counts per period."""
    site, ungated, features = job
    cache = common.load_cache()
    return site, engine.run_site(
        site=site, cache=cache, ungated=ungated, features=features, kinds=tuple(engine.KINDS)
    )


def main() -> None:
    """Run the sites in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sites", nargs="*", default=list(common.ALL_SITES))
    parser.add_argument("--ungated", action="store_true")
    parser.add_argument("--features", default="halfhourly", choices=sorted(common.FEATURE_SETS))
    args = parser.parse_args()
    jobs = [(site, args.ungated, args.features) for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for site, written in pool.map(one_site, jobs):
            print(site, written, flush=True)


if __name__ == "__main__":
    main()
