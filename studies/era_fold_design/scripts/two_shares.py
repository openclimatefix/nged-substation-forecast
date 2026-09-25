"""Two coverage shares per saved losses file (read-only): (a) months in >=2 years with no training row, (b) months in one year only."""
from pathlib import Path
import polars as pl
ROOT = Path("/home/jack/dev/nged-substation-forecast/data/studies/beam_diffuse_split")
FILES = ["beam_diffuse_wind_products/losses.parquet","past_weather_v2/ens_hres_past_wind/losses.parquet","past_weather_v2/ens_hres_past_wind/losses_long_rows.parquet","past_weather_v2/wind_icon_dream/losses.parquet","past_weather_v2/station_wind_arms/losses.parquet"]
for name in FILES:
    p = ROOT / name
    f = pl.scan_parquet(p).select("site","time","fold").unique().collect()
    f = f.unique(subset=["site","time"], keep="first")
    r = f.with_columns(cm=pl.col("time").dt.month(), yr=pl.col("time").dt.year())
    tot = r.group_by("site","cm").agg(n_tot=pl.len(), n_years=pl.col("yr").n_unique())
    cells = r.group_by("site","fold","cm").agg(n=pl.len()).join(tot, on=["site","cm"])
    hole = cells.filter(pl.col("n_tot")==pl.col("n"))
    a = int(hole.filter(pl.col("n_years")>1)["n"].sum()); b = int(hole.filter(pl.col("n_years")==1)["n"].sum())
    print(f"{name:75s} rows={f.height:6d} (a) {a:6d} {a/f.height:6.2%}  (b) {b:6d} {b/f.height:6.2%}  b_months={sorted(set(hole.filter(pl.col('n_years')==1)['cm'].to_list()))}")
