"""Stage 1: the known-answer test on the five MW decision Primaries.

For each Primary the signed series with a metered farm hidden (levels 0, 1.0, and 2.0 times the
site's midday median absolute reading) is folded to apparent power with zero reactive power or a
power factor of 0.97. Arms N, A, and F run on the folded series and the signed ceiling S on the
signed series, all for the scored year only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import e00152_common as common
import e00152_engine as engine
import e00152_inject as inject

FOLDED_ARMS = {"N": "naive", "A": "meter", "F_fold": "meter"}
LEVELS = (0.0, 1.0, 2.0)


def job(spec: tuple[str, float, str]) -> tuple[tuple[str, float, str], dict[str, int]]:
    """Run one (site, level, reactive mode) combination; mode `signed` runs the ceiling arm."""
    site, level, mode = spec
    cache = common.load_cache()
    signed, folded = inject.signed_and_folded(
        cache=cache, site=site, level=level, power_factor=mode == "pf97"
    )
    variant = f"known_{site}_L{level}_{mode}"
    if mode == "signed":
        written = engine.run_site(
            site=site,
            variant=variant,
            cache=cache,
            kinds=("naive",),
            arms={"S": "naive"},
            y=signed,
            periods=("scored",),
        )
    else:
        written = engine.run_site(
            site=site,
            variant=variant,
            cache=cache,
            kinds=("naive", "meter"),
            arms=FOLDED_ARMS,
            y=folded,
            periods=("scored",),
        )
    return spec, written


def main() -> None:
    """Run all combinations in parallel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="*", default=list(common.DECISION_SITES))
    args = parser.parse_args()
    specs = [(site, 0.0, "q0") for site in args.sites]
    specs += [
        (site, level, mode) for site, level, mode in product(args.sites, LEVELS[1:], ("q0", "pf97"))
    ]
    specs += [(site, level, "signed") for site, level in product(args.sites, LEVELS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for spec, written in pool.map(job, specs):
            print(spec, written, flush=True)


if __name__ == "__main__":
    main()
