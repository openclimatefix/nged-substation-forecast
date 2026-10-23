"""Step 0: how many of the 35 real load shifts have a metered partner substation?

Writes the neighbour list (the interconnection list of experiment 37, restricted to the 15
Primaries with a stored residual series) to `neighbours.json` before any alarm is scored.
A shift has a partner when one of its Primaries has an interconnected, metered neighbour.
It has a counterpart when both Primaries of its logged pair are metered.

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth \
        uv run --no-sync python ml-experiments/00315-lsc-neighbour-residual-filter/e00315_ceiling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import e00309_data as dat
import e00309_harness as h
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00315-lsc-neighbour-residual-filter"
E302 = ML_EXPERIMENTS_DATA_DIR / "00302-lsc-learned-score-without-nged-labels"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metered = set(h.SERIES)
    neighbours = {s: [n for n in dat.ADJACENCY[s] if n in metered] for s in h.SERIES}
    (OUT / "neighbours.json").write_text(json.dumps(neighbours, indent=1))
    hits = pl.read_parquet(E302 / "shift_hits.parquet")
    stratum = set(hits.filter(pl.col("long_deep") == True)["card_id"].unique())  # noqa: E712
    shifts = dat.labelled_loads()
    print("shifts", shifts.height, "stratum cards", len(stratum))
    rows = []
    for r in shifts.iter_rows(named=True):
        pair = [int(x) for x in r["series"]]
        pair_m = [s for s in pair if s in metered]
        nb = sorted({n for s in pair_m for n in neighbours[s]} - set())
        counterpart = len(pair_m) == len(pair) and len(pair) >= 2
        partner = any(neighbours[s] for s in pair_m)
        outside = sorted({n for s in pair_m for n in neighbours[s]} - set(pair))
        rows.append(
            {
                "card_id": r["card_id"], "hours": r["hours"], "pair": pair,
                "stratum": r["card_id"] in stratum, "n_metered_in_pair": len(pair_m),
                "counterpart": counterpart, "partner": partner,
                "partners_outside_pair": outside,
            }
        )
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "ceiling.parquet")
    for name, expr in (("stratum", pl.col("stratum")), ("rest", ~pl.col("stratum")), ("all", pl.lit(True))):
        s = t.filter(expr)
        print(name, s.height, "partner", int(s["partner"].sum()), "counterpart",
              int(s["counterpart"].sum()), "partner outside pair", int((s["partners_outside_pair"].list.len() > 0).sum()))
    print(t.select("card_id", "hours", "pair", "stratum", "n_metered_in_pair", "partner", "partners_outside_pair"))


if __name__ == "__main__":
    main()
