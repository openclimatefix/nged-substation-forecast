"""Join the stage reports of experiment 00347 into `stage123b_report.txt` (private data folder).

export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_report.py
"""

from typing import Final

import e00347_common as common

PARTS: Final[tuple[str, ...]] = (
    "stage1_report.txt",
    "stage2_report.txt",
    "stage2_known_answer_report.txt",
    "stage3_report.txt",
    "stage3b_report.txt",
)


def main() -> None:
    """Write the joined report."""
    text = "\n\n".join((common.DATA_DIR / name).read_text().rstrip() for name in PARTS) + "\n"
    (common.DATA_DIR / "stage123b_report.txt").write_text(text)
    print(f"wrote {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
