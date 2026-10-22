# ruff: noqa: E501
"""Experiment 191 control: the engine's a0 fit equals `studies.dp_gam.fit_gam`, and the price term works.

uv run --no-sync python ml-experiments/00191-*/e00191_check.py
"""

import time
from datetime import UTC, datetime

import e00191_common as common
import e00191_engine as engine
import numpy as np
from studies.dp_gam import TrainingWindow, fit_gam


def main() -> None:
    """Fit D14 at one cut both ways and print the differences."""
    torch_threads = 4
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(torch_threads)
    cache = common.load_cache()
    cut = datetime(2025, 7, 1, tzinfo=UTC)
    for site, ungated in (("D14", False), ("D8", False)):
        work = engine.masked_cache(cache=cache, site=site)
        grid = engine.build_price_grid(cache=work, site=site, ungated=ungated)
        started = time.monotonic()
        mine = engine._cut_fits(cache=work, site=site, cut=cut, grid=grid)
        seconds = time.monotonic() - started
        reference = fit_gam(
            cache=work,
            site=site,
            specs=[engine.gam_spec()],
            cut=cut,
            window=TrainingWindow(months=common.WINDOW_MONTHS),
            season_years=common.SEASON_YEARS,
        )
        diff = float(np.abs(mine.a0.theta - reference.theta).max())
        print(
            f"{site}: fit seconds {seconds:.0f}; rows {mine.n_rows}; masked {mine.n_masked}; "
            f"max |theta_a0 - fit_gam theta| = {diff:.2e}; "
            f"price coefficients {np.round(mine.joint.theta_price, 4)}; "
            f"rows with a gate of 1 and no price: {grid.n_missing}; "
            f"theta_t differs from a0 by {float(np.abs(mine.t.theta - mine.a0.theta).max()):.3g}"
        )


if __name__ == "__main__":
    main()
