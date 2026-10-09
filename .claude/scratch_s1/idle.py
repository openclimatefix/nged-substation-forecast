import polars as pl, glob
rows=[]
for f in sorted(glob.glob("/home/jack/dev/nged-substation-forecast/data/studies/per_study/embedded_battery_forecast/inputs/ID-1h__*.parquet")):
    b=f.split("__")[1][:-8]
    d=pl.read_parquet(f,columns=["time","output_mw"]).drop_nulls().with_columns(m=pl.col("time").dt.strftime("%Y-%m"))
    z=d.group_by("m").agg(zs=(pl.col("output_mw")==0).mean(),n=pl.len()).sort("m")
    rows.append((b,z))
    hi=z.filter(pl.col("zs")>=0.9)
    if hi.height: print(b,[(r["m"],round(r["zs"],3)) for r in z.iter_rows(named=True)])
