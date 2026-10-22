# ruff: noqa: E501
"""Experiment 191, step 2: the half-hourly system price, gas price and price features.

Reads the public Elexon system prices and the National Gas daily system average price already
stored by experiments 177, 185 and 186, builds the four price indicators the GAM uses, and checks
the table for gaps and implausible values before anything fits to it.

    uv run --no-sync python ml-experiments/00191-*/e00191_prices.py
"""

import e00191_common as common
import polars as pl


def build() -> pl.DataFrame:
    """Return one row per half-hour: start (UTC), price, gas, spread and the four indicators."""
    files = [
        *sorted((common.PUBLIC_186 / "system_prices").glob("*.parquet")),
        *sorted((common.PUBLIC_185 / "system_prices").glob("*.parquet")),
        *sorted((common.PUBLIC_177 / "system_prices").glob("*.parquet")),
    ]
    system = (
        pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        .with_columns(
            start=pl.col("startTime").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"),
            date=pl.col("settlementDate").str.to_date(),
            price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"),
        )
        .select("start", "date", "price")
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )
    gas = (
        pl.read_csv(common.PUBLIC_186 / "gas_sap.csv")
        .with_columns(
            date=pl.col("Applicable For").str.to_date("%d/%m/%Y"),
            pence=pl.col("Value").cast(pl.Float64, strict=False),
        )
        .select("date", "pence")
        .unique(subset="date", keep="first", maintain_order=True)
        .filter(
            pl.col("pence").is_between(
                common.MIN_PLAUSIBLE_GAS_P_PER_KWH, common.MAX_PLAUSIBLE_GAS_P_PER_KWH
            )
        )
        .with_columns(gas_gbp_per_mwh_thermal=pl.col("pence") * 10.0)
    )
    table = system.join(gas.select("date", "gas_gbp_per_mwh_thermal"), on="date", how="left")
    table = table.with_columns(
        spread=pl.col("price")
        - pl.col("gas_gbp_per_mwh_thermal") / common.EFFICIENCY
        - common.VARIABLE_COST_GBP_PER_MWH
    )
    for level in common.PRICE_PERCENTILES:
        threshold = table["price"].quantile(level)
        table = table.with_columns(
            (pl.col("price") > threshold).cast(pl.Float64).alias(f"price_p{round(level * 100)}")
        )
    table = table.with_columns(
        spread_pos=pl.when(pl.col("spread").is_null())
        .then(None)
        .otherwise((pl.col("spread") > 0.0).cast(pl.Float64))
    )
    # Post hoc daily features: the settlement day's mean price and mean spread, thresholded the same way.
    daily = table.group_by("date").agg(
        day_price=pl.col("price").mean(), day_spread=pl.col("spread").mean()
    )
    for level in common.PRICE_PERCENTILES:
        threshold = daily["day_price"].quantile(level)
        daily = daily.with_columns(
            (pl.col("day_price") > threshold).cast(pl.Float64).alias(f"day_p{round(level * 100)}")
        )
    daily = daily.with_columns(
        day_spread_pos=pl.when(pl.col("day_spread").is_null())
        .then(None)
        .otherwise((pl.col("day_spread") > 0.0).cast(pl.Float64))
    )
    return table.join(
        daily.select("date", "day_p50", "day_p75", "day_p90", "day_spread_pos"),
        on="date",
        how="left",
    )


def main() -> None:
    """Build, check and store the table."""
    table = build()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.PRICES_PATH)
    first, last = table["start"].min(), table["start"].max()
    full = pl.datetime_range(first, last, interval="30m", time_zone="UTC", eager=True)  # ty: ignore[no-matching-overload]
    missing = full.len() - table.height
    print(f"rows {table.height}, {first} to {last}; half-hours missing from the span: {missing}")
    print(f"duplicate starts: {int(table['start'].is_duplicated().sum())}")
    print(table.select("price", "gas_gbp_per_mwh_thermal", "spread").describe())
    print(
        "share of half-hours with a null spread (no plausible gas price):",
        round(table["spread"].null_count() / table.height, 4),
    )
    for level in common.PRICE_PERCENTILES:
        print(f"price percentile {level}: {table['price'].quantile(level):.1f} pounds per MWh")
    print("share with spread > 0 by year:")
    print(
        table.group_by(pl.col("start").dt.year().alias("year"))
        .agg(share=pl.col("spread_pos").mean(), price=pl.col("price").mean())
        .sort("year")
    )
    print("hour-of-day profile of price (UTC hour, mean):")
    print(
        table.group_by(pl.col("start").dt.hour().alias("hour"))
        .agg(price=pl.col("price").mean())
        .sort("hour")
        .transpose()
        .to_pandas()
        .round(0)
        .to_string()
        if False
        else table.group_by(pl.col("start").dt.hour().alias("hour"))
        .agg(price=pl.col("price").mean())
        .sort("hour")
    )


if __name__ == "__main__":
    main()
