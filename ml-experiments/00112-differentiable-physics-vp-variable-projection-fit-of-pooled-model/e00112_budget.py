"""Tally the GPU time of experiment 112 against its cap, and say what to cut if it overruns.

Reads the seconds of every saved fit (`work/fit_records.json`) and prints the total in hours, the
share by kind of fit, and the cut order that applies when the total passes the cap: P-map-polish and
the other exploratory comparisons, the second start of VP-S, then the replicate-year refits of the
secondary arms (the 2023-07-01 leave-one-out and short-history refits).

    uv run --no-sync python ml-experiments/00112-*/e00112_budget.py
"""

import json
from collections import defaultdict
from typing import Final

import e00112_common as common
import e00112_vp as vp

CUT_ORDER: Final[tuple[str, ...]] = (
    "exploratory comparisons (P-map-polish, VP-P-nonet, VP-P-pin)",
    "the second start of VP-S",
    "the 2023-07-01 leave-one-out and short-history refits (replicate-year spreads)",
)


def kind_of(key: str) -> str:
    """Return the kind of fit a record key names."""
    name = key.split("__", maxsplit=1)[0]
    for prefix in ("gate_", "profile_", "vp_loo_", "vp_short3m_", "vp_s_", "vp_p_", "p_map_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def main() -> None:
    """Print the tally."""
    records = json.loads(vp.RECORDS_PATH.read_text()) if vp.RECORDS_PATH.exists() else {}
    hours: dict[str, float] = defaultdict(float)
    for key, record in records.items():
        hours[kind_of(key)] += float(record.get("seconds", 0.0)) / 3600.0
    diagnostics = common.WORK_DIR / "gate_diagnostics.json"
    if diagnostics.exists():
        entries = json.loads(diagnostics.read_text()).values()
        hours["gate diagnostics"] = (
            sum(float(e.get("seconds", 0.0)) for e in entries if isinstance(e, dict)) / 3600.0
        )
    ladder = common.WORK_DIR / "ladder_objectives.json"
    if ladder.exists():
        hours["ladder"] = (
            sum(float(v["seconds"]) for v in json.loads(ladder.read_text()).values()) / 3600.0
        )
    total = sum(hours.values())
    for kind, value in sorted(hours.items(), key=lambda item: -item[1]):
        print(f"{kind:14}{value:7.2f} h")
    print(f"{'total':14}{total:7.2f} h of the {common.GPU_CAP_HOURS:.0f} h cap")
    if total > common.GPU_CAP_HOURS:
        print("over the cap; cut in this order:", *CUT_ORDER, sep="\n  ")


if __name__ == "__main__":
    main()
