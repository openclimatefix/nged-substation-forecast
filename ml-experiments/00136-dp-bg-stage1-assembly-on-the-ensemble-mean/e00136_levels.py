"""Per-quantile pinball loss and mean width of A0 against experiment 111's GAM-K (C1 diagnostic).

Tests the explanation for C1 (the ensemble-mean arm scores worse than the member mixture): the
single residual table around one mean forecast gives narrower lower tails than the mixture of 51
members. Prints, per level, each arm's mean pinball loss and the 5%-to-95% width at 12 to 36 hours.

    uv run python ml-experiments/00136-*/e00136_levels.py
"""

import e00136_report as rep
import numpy as np
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS


def main() -> None:
    """Print the per-level diagnostic on the scored year's common rows."""
    loaded = {a: rep.load(arm=a, period="scored") for a in ("a0", "k111")}
    frames = rep.common_rows(frames={a: f for a, f in loaded.items() if f is not None})
    for name, loaded in frames.items():
        frame = loaded.filter(
            pl.col("site").is_in(list(rep.common.DECISION_SITES)) & (pl.col("lead_hours") > 0)
        )
        actual = frame["actual"].to_numpy().astype(np.float64)
        print(name)
        for level, column in zip(rep.LEVELS, QUANTILE_COLUMNS, strict=True):
            error = actual - frame[column].to_numpy().astype(np.float64)
            pinball = np.maximum(level * error, (level - 1) * error).mean()
            print(f"  {column}: pinball {pinball:.4f}, coverage {(error <= 0).mean():.4f}")
        width = (frame["q95"] - frame["q05"]).mean()
        print(f"  mean q05-q95 width {width:.3f}")


if __name__ == "__main__":
    main()
