"""List the exploratory contrasts near the 5% line (an interval bound within 20% of the width from zero)."""
import json
import polars as pl
from common import OUT
c = pl.read_parquet(OUT / "partC_contrasts.parquet").filter(
    pl.col("setting") == "pooled", ~pl.col("design").str.contains("-"), ~pl.col("planned"))
c = c.with_columns(near=pl.min_horizontal(pl.col("lo").abs(), pl.col("hi").abs()) <= 0.2 * (pl.col("hi") - pl.col("lo")))
sel = c.filter(pl.col("near")).select("domain", "treatment", "reference").unique().sort("domain", "treatment", "reference")
print(sel.height, "near-line exploratory contrasts (any design)")
print(sel)
(OUT / "sens_extra_contrasts.json").write_text(json.dumps(sel.rows()))
