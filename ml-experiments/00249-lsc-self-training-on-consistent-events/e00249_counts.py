# ruff: noqa: E501
"""Print the size of the injected training sets and of the real candidate table.

uv run --no-sync python ml-experiments/00249-*/e00249_counts.py > ml-experiments/00249-*/printed_counts.txt
"""

import polars as pl
from e00249_data import KINDS, block_days
from e00249_fit import fold_frames, real_candidates
from e00249_models import usable


def main() -> None:
    """Print the counts the page quotes."""
    table, _ = real_candidates()
    print(f"real candidates: {table.height}; block days: {block_days().height}")
    for fold in range(3):
        train, _, _ = fold_frames(fold=fold)
        feat = usable(frame=train, sources=("x", "p"))[0]
        x_only = usable(frame=train, sources=("x",))[0]
        print(
            f"fold {fold}: {KINDS['train'][0]} training worlds; usable candidates {feat.height} "
            f"({int((feat['label'] == 1).sum())} injected positives); RO-NC candidates only {x_only.height} ({int((x_only['label'] == 1).sum())} positives)"
        )
    assert pl is not None


if __name__ == "__main__":
    main()
