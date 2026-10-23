# ruff: noqa: E501
"""Time one T-all fit (the convex model's refit) and report it, for the cost column of experiment 278.

uv run --no-sync python ml-experiments/00278-*/e00278_fit_time.py
"""

import time
from datetime import UTC, datetime

import e00278_g_common as common
import e00278_g_tall as tall
import torch  # ty: ignore[unresolved-import]
from studies.dp_pinned_capacity import read_vintages


def main() -> None:
    """Fit D5 at the 2025-10-01 cut three times and print the seconds."""
    torch.set_num_threads(1)
    cache = common.load_cache()
    vintages = read_vintages()
    cut = datetime(2025, 10, 1, tzinfo=UTC)
    for _ in range(3):
        started = time.monotonic()
        tall.fit_tall(
            cache=cache, vintages=vintages, site="D5", cut=cut, penalty_weight=common.T_ALL_LAMBDA
        )
        print(f"{time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
