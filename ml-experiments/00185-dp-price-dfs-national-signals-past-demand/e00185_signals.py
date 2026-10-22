"""Build the daily national signal table of experiment 185 from the cached public files.

One row per local (Europe/London) calendar day, 2024-07-01 to 2026-06-30. Every aggregate is over
the day's settlement periods. The evening peak is settlement periods 33 to 38 (16:00 to 19:00 local
on a 48-period day). The DFS flag is true when the NESO archive lists a live event on the delivery
date; the archive runs from 2024-11-29 to 2026-03-27, so `dfs_covered` is false after that date.
"""

from datetime import date
from pathlib import Path
from typing import Final

import e00185_common as common
import polars as pl

FIRST: Final[date] = date(2024, 7, 1)
LAST: Final[date] = date(2026, 6, 30)
DFS_FIRST_COVERED: Final[date] = date(2024, 11, 29)
DFS_LAST_COVERED: Final[date] = date(2026, 3, 27)
EVENING: Final[tuple[int, int]] = (33, 38)
FLOWS: Final[tuple[str, ...]] = (
    "IFA_FLOW", "IFA2_FLOW", "BRITNED_FLOW", "MOYLE_FLOW", "EAST_WEST_FLOW", "NEMO_FLOW",
    "NSL_FLOW", "ELECLINK_FLOW", "VIKING_FLOW", "GREENLINK_FLOW",
)  # fmt: skip


def _read_parquets(*, pattern: str) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(f) for f in sorted(Path(pattern).parent.glob(Path(pattern).name))]
    )


def price_days() -> pl.DataFrame:
    """Return daily mean, maximum, and evening-peak system price (pounds per megawatt-hour)."""
    prices = pl.concat(
        [
            _read_parquets(pattern=str(common.PUBLIC_DIR / "system_prices" / "*.parquet")),
            _read_parquets(pattern=str(common.E177_PUBLIC / "system_prices" / "*.parquet")),
        ]
    ).unique(subset=["settlementDate", "settlementPeriod"])
    prices = prices.with_columns(
        date=pl.col("settlementDate").str.to_date(),
        price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"),
    )
    return prices.group_by("date").agg(
        price_mean=pl.col("price").mean(),
        price_max=pl.col("price").max(),
        price_eve=pl.col("price").filter(pl.col("settlementPeriod").is_between(*EVENING)).mean(),
        n_price_periods=pl.len(),
    )


def demand_days() -> pl.DataFrame:
    """Return daily national demand, embedded wind and solar, and net interconnector flow (MW)."""
    pieces = []
    for year in (2024, 2025, 2026):
        frame = pl.read_csv(common.PUBLIC_DIR / f"demanddata_{year}.csv", infer_schema_length=10000)
        flows = [c for c in FLOWS if c in frame.columns]
        pieces.append(
            frame.select(
                date=pl.coalesce(
                    pl.col("SETTLEMENT_DATE").str.to_date("%d-%b-%Y", strict=False),
                    pl.col("SETTLEMENT_DATE").str.to_date("%Y-%m-%d", strict=False),
                ),
                period=pl.col("SETTLEMENT_PERIOD"),
                nd=pl.col("ND"),
                wind_emb=pl.col("EMBEDDED_WIND_GENERATION"),
                solar_emb=pl.col("EMBEDDED_SOLAR_GENERATION"),
                ic_net=pl.sum_horizontal(*flows),
            )
        )
    frame = pl.concat(pieces, how="diagonal")
    evening = pl.col("period").is_between(*EVENING)
    return frame.group_by("date").agg(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in ("nd", "wind_emb", "solar_emb", "ic_net")],
        *[
            pl.col(c).filter(evening).mean().alias(f"{c}_eve")
            for c in ("nd", "wind_emb", "solar_emb", "ic_net")
        ],
    )


def wind_days() -> pl.DataFrame:
    """Return daily mean and evening-peak transmission-connected wind output (Elexon FUELHH)."""
    wind = _read_parquets(pattern=str(common.PUBLIC_DIR / "wind_fuelhh" / "*.parquet")).unique(
        subset=["settlementDate", "settlementPeriod"]
    )
    wind = wind.with_columns(date=pl.col("settlementDate").str.to_date())
    return wind.group_by("date").agg(
        wind_tx_mean=pl.col("generation").mean(),
        wind_tx_eve=pl.col("generation")
        .filter(pl.col("settlementPeriod").is_between(*EVENING))
        .mean(),
    )


def dfs_days() -> pl.DataFrame:
    """Return the days with a live DFS event, and the event's megawatt-hours requested."""
    dfs = pl.read_csv(common.E177_PUBLIC / "dfs_service_requirement.csv").with_columns(
        date=pl.col("Delivery Date").str.to_date("%d/%m/%Y")
    )
    return (
        dfs.group_by("date")
        .agg(dfs_mw_max=pl.col("Service Requirement MW").max())
        .with_columns(dfs=pl.lit(True))
    )


def build() -> pl.DataFrame:
    """Return the daily national signal table."""
    days = pl.DataFrame({"date": pl.date_range(FIRST, LAST, "1d", eager=True)})
    return (
        days.join(price_days(), on="date", how="left")
        .join(demand_days(), on="date", how="left")
        .join(wind_days(), on="date", how="left")
        .join(dfs_days(), on="date", how="left")
        .with_columns(
            dfs=pl.col("dfs").fill_null(False),
            dfs_mw_max=pl.col("dfs_mw_max").fill_null(0),
            dfs_covered=pl.col("date").is_between(DFS_FIRST_COVERED, DFS_LAST_COVERED),
        )
        .rename({"date": "local_date"})
    )


def main() -> None:
    """Write the table and print its checks."""
    table = build()
    table.write_parquet(common.WORK_DIR / "national_signals.parquet")
    print(table.shape, table["local_date"].min(), table["local_date"].max())
    print(table.null_count().row(0, named=True))
    print(table.filter(pl.col("n_price_periods") != 48).select("local_date", "n_price_periods"))
    print(table.select(pl.exclude("local_date", "dfs_covered")).describe())


if __name__ == "__main__":
    main()
