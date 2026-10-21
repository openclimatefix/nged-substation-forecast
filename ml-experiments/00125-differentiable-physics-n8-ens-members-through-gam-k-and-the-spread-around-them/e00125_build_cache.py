"""Build experiment 125's cache of corrected member means (see `e00125_cache`).

uv run python ml-experiments/00125-*/e00125_build_cache.py --periods tuning replicate \
        [--workers 4]
"""

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import e00125_cache as cache_module
import e00125_common as common


def _job(job: tuple[str, str]) -> str:
    import torch

    torch.set_num_threads(1)
    site, period = job
    started = time.monotonic()
    runs = cache_module.build_site(site=site, period=period)
    path = cache_module.cache_path(site=site, period=period)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(runs))
    temporary.rename(path)
    return f"{site} {period}: {len(runs)} runs in {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build the cache for the chosen periods and sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=["tuning"])
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(site, period) for period in args.periods for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
