"""Matches, alarms per match and MW-weighted matches on the long-and-deep stratum and the rest.

The stratum is experiment 302's: a real load shift of at least 2 days that moves at least 10% of
load (19 of the 35 labelled real transfers over 1 day; 8 have no recorded share). The stratum was
fixed in experiment 302 before this experiment's results were read. LR_b20 is experiment 302's `L0`
arm (a card counts as matched when most of its three held-out folds matched it); its alarms per
match are its mean alarm count over the three folds divided by its matches.

    uv run --no-sync python e00298_stratum.py > printed_stratum.txt
"""

from __future__ import annotations

import e00298_common as common
import e00298_detect as dt
import e00298_events as ev
import e00298_real as real
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E302 = ML_EXPERIMENTS_DATA_DIR / "00302-lsc-learned-score-without-nged-labels"
E217 = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"


def line(*, label: str, found: dict[str, bool], n_alarms: float, events: list[dict], strat: set[str]) -> str:
    """Format matches, alarms per match and MW-weighted matches for the stratum and the rest."""
    parts = [f"{label:22s} alarms={n_alarms:6.1f}"]
    for name, keep in (("stratum", lambda e: e["card_id"] in strat), ("rest", lambda e: e["card_id"] not in strat)):
        sel = [e for e in events if keep(e)]
        hits = [e for e in sel if found[e["card_id"]]]
        mw = sum(abs(e["moved_mw"]) for e in sel if np.isfinite(e["moved_mw"]))
        mw_hit = sum(abs(e["moved_mw"]) for e in hits if np.isfinite(e["moved_mw"]))
        apm = n_alarms / len(hits) if hits else float("inf")
        parts.append(f"{name}: {len(hits)}/{len(sel)} apm={apm:5.1f} mw={mw_hit / mw if mw else 0:.3f}")
    return "  ".join(parts)


def main() -> None:
    """Print the stratum table for every detector."""
    events = [e for e in ev.load_events() if e["cls"] == "transfer"]
    hits302 = pl.read_parquet(E302 / "shift_hits.parquet")
    strat = set(hits302.filter(pl.col("long_deep") == True)["card_id"].unique())  # noqa: E712
    print(f"transfers: {len(events)}; stratum (experiment 302's): {len(strat)} cards, "
          f"{sum(e['card_id'] in strat for e in events)} among these events")  # fmt: skip
    # My variants at their tuned thresholds.
    for family, _ in real.FAMILIES:
        table = pl.read_parquet(common.EXPERIMENT_DIR / f"real_spans_{family}.parquet")
        spans: dict[str, list[dt.Span]] = {}
        for r in table.iter_rows(named=True):
            spans.setdefault(r["site"], []).append(dt.Span(r["start"], r["end"], r["fire"], 1, r["size_mw"]))
        found = {}
        for e in events:
            found[e["card_id"]] = any(
                s.start <= e["end"] + 1 and s.end >= e["start"] - 1 for site in e["sites"] for s in spans.get(site, [])
            )
        print(line(label=family, found=found, n_alarms=float(table.height), events=events, strat=strat))
    # Stored RO-NC, pair alarm and learned re-ranker (spans: consecutive alarm days merged).
    for label, name in (("RO-NC R2", "alarms_RO-NC_R2"), ("RO-NC current", "alarms_RO-NC_current"),
                        ("pair R2", "alarms_pair_R2"), ("pair current", "alarms_pair_current"),
                        ("learned re-ranker", "alarms_learned_R_re-ranker_stored")):  # fmt: skip
        table = pl.read_parquet(E217 / f"{name}.parquet")
        days: dict[str, list[int]] = {}
        for series, stamp in table.select("series", "stamp").iter_rows():
            site = common.SERIES_TO_SITE.get(int(series))
            if site:
                days.setdefault(site, []).append(ev.stamp_day(stamp=stamp))
        runs = {s: [d for i, d in enumerate(sorted(set(v))) if i == 0 or d - sorted(set(v))[i - 1] > 1] for s, v in days.items()}
        found = {e["card_id"]: any(e["start"] - 1 <= d <= e["end"] + 1 for s in e["sites"] for d in days.get(s, [])) for e in events}
        print(line(label=label, found=found, n_alarms=float(sum(len(v) for v in runs.values())), events=events, strat=strat))
    # LR_b20 = experiment 302's L0 arm.
    l0 = hits302.filter((pl.col("arm") == "L0") & (pl.col("variant") == "all"))
    rate = l0.group_by("card_id").agg(pl.col("found").mean())
    found = {r["card_id"]: r["found"] > 0.5 for r in rate.iter_rows(named=True)}
    counts = pl.read_parquet(E302 / "alarm_counts.parquet").filter((pl.col("arm") == "L0") & (pl.col("variant") == "all"))
    n_alarms = counts["alarms"].sum() / counts["fold"].n_unique()
    print(line(label="LR_b20 (302 L0)", found={e["card_id"]: found.get(e["card_id"], False) for e in events}, n_alarms=float(n_alarms), events=events, strat=strat))


if __name__ == "__main__":
    main()
