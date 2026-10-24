"""Run experiment 294's arms at one or more sites and write the prediction frames.

Run it under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` with
`uv run --no-sync python`, for example with `--config whole --sites D8 D5 D10 D11 --workers 2`.
`--config post` is experiment 191's reference rows (D8 from 2025-04-01, a 24-month window).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import e00294_common as common
import e00294_engine as engine


def one_site(job: tuple[str, str]) -> tuple[str, dict[str, int]]:
    """Run one site and return its row counts per period."""
    site, config = job
    cache = common.load_cache()
    return site, engine.run_site(site=site, cache=cache, config=config)


def main() -> None:
    """Run the sites in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sites", nargs="*", default=list(common.ALL_SITES))
    parser.add_argument("--config", required=True, choices=sorted(engine.CONFIGS))
    args = parser.parse_args()
    jobs = [(site, args.config) for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for site, written in pool.map(one_site, jobs):
            print(site, written, flush=True)


if __name__ == "__main__":
    main()
