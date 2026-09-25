"""Coverage of the folds saved with every published study's per-row losses (review 2). Read-only."""
from pathlib import Path
import polars as pl
ROOT = Path("/home/jack/dev/nged-substation-forecast/data/studies")
for p in sorted(ROOT.rglob("*.parquet")):
    if "superseded" in p.parts or p.stat().st_size > 2_000_000_000:
        continue
    try:
        names = pl.scan_parquet(p).collect_schema().names()
    except Exception:
        continue
    if not {"site", "time", "fold"} <= set(names):
        continue
    f = pl.scan_parquet(p).select("site", "time", "fold").unique().collect()
    if f.select(pl.struct("site", "time").is_duplicated().any()).item():
        note = "(site,time) in >1 fold"
        f = f.unique(subset=["site", "time"], keep="first")
    else:
        note = ""
    r = f.with_columns(cm=pl.col("time").dt.month(), yr=pl.col("time").dt.year())
    tot = r.group_by("site", "cm").agg(n_tot=pl.len(), n_years=pl.col("yr").n_unique())
    cells = r.group_by("site", "fold", "cm").agg(n=pl.len()).join(tot, on=["site", "cm"])
    u = cells.filter(pl.col("n_tot") == pl.col("n"), pl.col("n_years") > 1)
    print(f"{str(p.relative_to(ROOT)):90s} rows={f.height:7d} uncovered={int(u['n'].sum()):6d} ({u['n'].sum()/f.height:6.2%}) months={sorted(set(u['cm'].to_list()))} {note}", flush=True)
