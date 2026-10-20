"""Run experiment 113's synthetic grid, one worker process per substation.

    uv run python ml-experiments/00113-*/e00113_grid.py reals
    uv run python ml-experiments/00113-*/e00113_grid.py main [--draws 20] [--workers 5]
    uv run python ml-experiments/00113-*/e00113_grid.py control [--workers 5]
    uv run python ml-experiments/00113-*/e00113_grid.py extra [--draws 20] [--workers 5]

`reals` fits the full model to each decision substation's real target and stores the residuals
that the real noise reuses. `main` runs `--draws` draws per substation. `control` runs the
noise-free and iid sd 1.0 draw. A finished draw's file is skipped on a re-run.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import e00113_cell as cell_module
import e00113_common as common

MAX_WORKERS: int = 6


def _run(arguments: tuple[str, range, str]) -> str:
    site, draws, stage = arguments
    cells = {
        "main": cell_module.draw_cells,
        "control": cell_module.control_cells,
        "extra": cell_module.extra_cells,
    }[stage]()
    cell_module.run_site(site=site, draws=draws, cells=cells, tag=stage)
    return site


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("reals", "main", "control", "extra"))
    parser.add_argument("--draws", type=int, default=common.N_DRAWS_PER_SITE)
    parser.add_argument("--workers", type=int, default=len(common.DECISION_SITES))
    parser.add_argument("--sites", nargs="*", default=list(common.DECISION_SITES))
    args = parser.parse_args()
    if args.workers > MAX_WORKERS:
        msg = f"the slot allows at most {MAX_WORKERS} cores"
        raise SystemExit(msg)
    if args.stage == "reals":
        common.configure_threads(threads=args.workers)
        cell_module.build_real_fits(cache=common.load_cache())
        return
    draws = (
        range(args.draws)
        if args.stage in ("main", "extra")
        else range(cell_module.CONTROL_DRAW, cell_module.CONTROL_DRAW + 1)
    )
    jobs = [(site, draws, args.stage) for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for site in pool.map(_run, jobs):
            print("finished", site, flush=True)


if __name__ == "__main__":
    main()
