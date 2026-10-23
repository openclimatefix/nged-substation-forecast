"""Check that the control arm c0 reproduces experiment 180's control frame for frame.

uv run --no-sync python ml-experiments/00280-*/e00280_check_control.py
"""

import e00280_common as common
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SOURCE = (
    ML_EXPERIMENTS_DATA_DIR / "00180-dp-sunny-daylight-fix-d5-solar-shape" / "work" / "arms" / "c"
)


def main() -> None:
    """Raise if any stored c0 frame differs from experiment 180's control frame."""
    frames = sorted((common.WORK_DIR / "arms" / "c0" / "dp_preds_s2").glob("d_*.parquet"))
    for path in frames:
        reference = pl.read_parquet(SOURCE / "dp_preds_s2" / path.name)
        if not pl.read_parquet(path).equals(reference):
            msg = f"{path.name} differs from experiment 180's control"
            raise ValueError(msg)
    print(f"{len(frames)} frames identical to experiment 180's control")


if __name__ == "__main__":
    main()
