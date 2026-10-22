# ruff: noqa: E501, D103
"""Print the markdown tables of the write-up from the stored tables of `e00217_run.py`."""

from __future__ import annotations

import e00217_score as score
import polars as pl

WORK_DIR = score.WORK_DIR


def main() -> None:
    rates = pl.read_parquet(WORK_DIR / "rates.parquet")
    stats = pl.read_parquet(WORK_DIR / "stats.parquet").sort("misfit_share", descending=True)
    print(
        "| Primary | Years of residual | Noise, 24 h (MW) | MAD 2 h | MAD 4 h | MAD 6 h | MAD 24 h | 95th pct 24 h | Kurtosis 24 h | Misfit days | Holiday misfit | RO-NC all | winter | rest | Pair all | winter | rest |"
    )
    print("|" + "---|" * 17)
    for r in stats.iter_rows(named=True):

        def rate(det: str, g: str, site: str = r["site"]) -> float:
            return rates.filter(
                (pl.col("detector") == det) & (pl.col("site") == site) & (pl.col("group") == g)
            )["rate"].item()

        print(
            f"| {r['site']} | {r['years']:.1f} | {r['cur_48']:.3f} | {r['mad_all_4']:.3f} | {r['mad_all_8']:.3f} | {r['mad_all_12']:.3f} | {r['mad_all_48']:.3f} | {r['q95_48']:.3f} | {r['kurt_48']:.1f} | {r['misfit_share']:.2f} | {r['holiday_misfit_share']:.2f} | "
            f"{rate('RO-NC/current', 'all'):.1f} | {rate('RO-NC/current', 'winter'):.1f} | {rate('RO-NC/current', 'other'):.1f} | "
            f"{rate('pair/current', 'all'):.1f} | {rate('pair/current', 'winter'):.1f} | {rate('pair/current', 'other'):.1f} |"
        )


if __name__ == "__main__":
    main()
