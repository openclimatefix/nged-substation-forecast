from typing import cast
import polars as pl
from contracts.settings import Settings

SUBSTATIONS = [110375, 110644, 110772, 110803, 110804]
settings = Settings()
actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
df = pl.read_delta(str(actuals_path)).lazy()
df = df.filter(pl.col("substation_number").is_in(SUBSTATIONS))

health_df = cast(
    pl.DataFrame,
    df.with_columns(MW_or_MVA=pl.coalesce(["MW", "MVA"]), date=pl.col("timestamp").dt.date())
    .group_by(["substation_number", "date"])
    .agg(
        max_abs=pl.col("MW_or_MVA").abs().max(),
        std=pl.col("MW_or_MVA").std(),
        count=pl.col("MW_or_MVA").count(),
    )
    .filter(
        (pl.col("count") > 50) & ((pl.col("max_abs") < 0.5) | (pl.col("std").fill_null(0.0) < 0.01))
    )
    .select("substation_number")
    .unique()
    .collect(),
)

bad_substations = health_df.get_column("substation_number").to_list()
print(f"Bad substations: {bad_substations}")
