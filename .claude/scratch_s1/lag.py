import polars as pl
b="/home/jack/dev/nged-substation-forecast/data/studies/downloads/market/"
n=pl.read_parquet(b+"neso_n2ex_day_ahead/neso_n2ex_day_ahead.parquet")
a=pl.read_parquet(b+"elexon_mid_apx/elexon_mid_apx.parquet").sort("time")
print(n.head(2), a.head(2), a.schema)
ah=a.group_by_dynamic("time",every="1h").agg(pl.col("price_gbp_per_mwh").mean().alias("apx"))
for name,months in [("winter",[11,12,1,2]),("summer",[5,6,7,8,9])]:
    for lag in range(-3,4):
        j=n.with_columns(t2=pl.col("time")+pl.duration(hours=lag)).join(ah,left_on="t2",right_on="time").filter(pl.col("time").dt.month().is_in(months))
        print(name,lag,round(j.select(pl.corr("price_gbp_per_mwh","apx")).item(),3))
