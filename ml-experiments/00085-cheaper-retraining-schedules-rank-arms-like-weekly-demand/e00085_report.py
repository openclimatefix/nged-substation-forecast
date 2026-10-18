"""Print every number the experiment 00085 page quotes: scores, tails, contrasts, and cost.

Reads `work/results/` (from `e00085_stats.py`) and prints the absolute scores with tail
calibration, the contrasts, the agreement table, and the cost of a year of refits under each
schedule. A configuration's mean fit seconds varied with the load of other jobs on the machine, so
the cost is given as the range over the configurations that share a schedule.

    uv run python ml-experiments/00085-*/e00085_report.py
"""

from typing import Final

import e00085_common as common
import polars as pl

N_SERIES: Final[int] = 16
SCHEDULES: Final[dict[str, tuple[int, tuple[str, ...]]]] = {
    "Full weekly (52 refits)": (52, ("m0",)),
    "Full monthly (12)": (12, ("m0",)),
    "Weekly, whole history, 150 rounds (52)": (52, ("fw", "fw0")),
    "Monthly, whole history, 150 rounds (12)": (12, ("fm",)),
    "P-roll weekly (52)": (52, ("pw", "pw0")),
    "P-roll monthly (12)": (12, ("pm_rep", "pm0", "pm0_rep")),
    "P-roll quarterly (4)": (4, ("pq", "pq0")),
}
"""Each schedule: refits in a year and the configurations whose fit time prices one refit. Full
weekly is priced with the full monthly configuration `m0`, which has the same recipe."""


def cost_table() -> pl.DataFrame:
    """Return each schedule's compute minutes for a year and 16 series, low and high."""
    fits = pl.read_parquet(common.RESULTS_DIR / "cost.parquet")
    rows = []
    for name, (refits, configs) in SCHEDULES.items():
        seconds = fits.filter(pl.col("config").is_in(configs))["mean_fit_s"]
        rows.append(
            {
                "schedule": name,
                "minutes_low": refits * N_SERIES * min(seconds.to_list()) / 60.0,
                "minutes_high": refits * N_SERIES * max(seconds.to_list()) / 60.0,
            }
        )
    table = pl.DataFrame(rows)
    full = table.filter(pl.col("schedule").str.starts_with("Full weekly"))["minutes_high"][0]
    return table.with_columns(share_of_full_weekly=pl.col("minutes_high") / full)


def main() -> None:
    """Print the tables."""
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=3, fmt_str_lengths=60):
        print(pl.read_parquet(common.RESULTS_DIR / "absolute.parquet"))
        print(pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet"))
        print(pl.read_parquet(common.RESULTS_DIR / "agreement.parquet"))
        print(pl.read_parquet(common.RESULTS_DIR / "cost.parquet"))
        table = cost_table()
        table.write_parquet(common.RESULTS_DIR / "cost_schedules.parquet")
        print(table)


if __name__ == "__main__":
    main()
