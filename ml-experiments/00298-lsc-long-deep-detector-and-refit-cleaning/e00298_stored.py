"""Score the stored alarms of RO-NC, the pair alarm and the learned score on experiment 298's events.

The alarm tables are experiment 217's `R2` operating points and the carried `current` ones, and
experiment 195's learned re-ranker. An alarm is a half-hour stamp counted from 2022-01-01. The
event windows, the 3-day exclusion for false alarms, and the Primary-years are the same as in
`e00298_real.py`.

    uv run --no-sync python e00298_stored.py > printed_stored.txt
"""

from __future__ import annotations

from typing import Final

import e00298_common as common
import e00298_detect as dt
import e00298_events as ev
import e00298_real as real
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DIR217: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
TABLES: Final[dict[str, str]] = {
    "RO-NC R2": "alarms_RO-NC_R2",
    "RO-NC current": "alarms_RO-NC_current",
    "pair alarm R2": "alarms_pair_R2",
    "pair alarm current": "alarms_pair_current",
    "learned re-ranker": "alarms_learned_R_re-ranker_stored",
}


def main() -> None:
    """Print recall by event class and false alarms per Primary-year for each stored alarm table."""
    streams = real.build_streams()
    events = ev.load_events()
    sites = sorted(streams)
    pooled = [s for s in sites if s not in real.SPECIAL]
    years_all = real.primary_years(streams=streams, odd=True, sites=pooled) + real.primary_years(
        streams=streams, odd=False, sites=pooled
    )
    masks = {s: real.event_mask(events=events, site=s, pad=real.EXCLUDE_DAYS) for s in sites}
    totals = {c: sum(e["cls"] == c for e in events) for c in ("block", "drop", "transfer")}
    print(f"events: {totals}; pooled Primary-years (excl. D8 and D9): {years_all:.2f}")
    for label, name in TABLES.items():
        table = pl.read_parquet(DIR217 / f"{name}.parquet")
        days: dict[str, list[int]] = {}
        for series, stamp in table.select("series", "stamp").iter_rows():
            site = common.SERIES_TO_SITE.get(int(series))
            if site in streams:
                days.setdefault(site, []).append(ev.stamp_day(stamp=stamp))
        found = {c: 0 for c in totals}
        for e in events:
            if any(e["start"] - 1 <= d <= e["end"] + 1 for s in e["sites"] for d in days.get(s, [])):
                found[e["cls"]] += 1
        # Merge consecutive alarm days into one span, as `e00298_real.py` counts spans.
        runs = {s: [d for i, d in enumerate(sorted(set(ds))) if i == 0 or d - sorted(set(ds))[i - 1] > 1]
                for s, ds in days.items()}  # fmt: skip
        false = {s: sum(not masks[s][d] for d in rs) for s, rs in runs.items()}
        pooled_false = sum(v for s, v in false.items() if s not in real.SPECIAL)
        print(
            f"{label:20s} alarms={sum(len(set(d)) for d in days.values()):4d} "
            f"blocks={found['block']}/{totals['block']} drops={found['drop']}/{totals['drop']} "
            f"transfers={found['transfer']}/{totals['transfer']} "
            f"false_per_primary_year={pooled_false / years_all:.2f}"
        )
    _ = np, dt


if __name__ == "__main__":
    main()
