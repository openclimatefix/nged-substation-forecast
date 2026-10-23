"""Choose the best any-alarm variant (A to C) and the best neighbour-confirmed one (D to F).

Selection uses only the tuning onsets (odd months) of the injected worlds: the variant with the
highest recall over shifts of 3 days or more and at least 10% of load; metered worlds for D to F.

    uv run --no-sync python e00298_select.py
"""

from __future__ import annotations

import json

import e00298_common as common
import polars as pl


def main() -> None:
    """Print the tuning-onset recall of each variant and write the two choices."""
    d = pl.read_parquet(common.EXPERIMENT_DIR / "injected_detection.parquet").filter(
        pl.col("run") == "true",
        pl.col("onset").dt.month() % 2 == 1,
        pl.col("duration") >= 3,
        pl.col("size") >= 0.10,
    )
    plain = d.filter(~pl.col("metered"))
    met = d.filter(pl.col("metered"))
    table = (
        plain.group_by("variant")
        .agg(recall_unmetered=pl.col("found").mean())
        .join(met.group_by("variant").agg(recall_metered=pl.col("found").mean()), on="variant")
        .sort("recall_unmetered", descending=True)
    )
    print(table)
    chosen = json.loads((common.EXPERIMENT_DIR / "chosen_k.json").read_text())
    any_alarm = table.filter(pl.col("variant").is_in(["A2", "A3", "A7", "B", "C"]))
    best_any = any_alarm["variant"][0]
    conf = table.filter(
        pl.col("variant").is_in([f for f in chosen if f[0] in "DEF" and "@" not in f])
    ).sort("recall_metered", descending=True)
    best_conf = conf["variant"][0]
    out = {"any": best_any, "confirmed": best_conf}
    (common.EXPERIMENT_DIR / "selected.json").write_text(json.dumps(out))
    print(out)


if __name__ == "__main__":
    main()
