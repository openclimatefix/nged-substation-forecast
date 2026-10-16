"""Print the numbers of experiment 00049 that the stage scripts save but do not print.

Prints the blend's weight table, the number of validation runs, and the number of refit files per
arm.

    uv run python ml-experiments/00049-vp1-screening-reference-under-p-roll-demand/e00049_report.py
"""

import e00049_fit as fitting
import e00049_regimes as regimes
import e00049_stage0 as stage0
import polars as pl


def main() -> None:
    """Print the weight tables, the run count, and the refit file counts."""
    print("validation runs:", len(regimes.run_origins()))
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        for name in ("blend_five", "blend_three"):
            print(name)
            print(pl.read_parquet(stage0.WORK_DIR / f"weights_{name}.parquet"))
    for folder in sorted(p for p in fitting.ANCHOR_PREDS_DIR.iterdir() if p.is_dir()):
        print("refit files", folder.name, len(list(folder.glob("*.parquet"))))


if __name__ == "__main__":
    main()
