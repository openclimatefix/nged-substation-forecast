"""Summarise the SW2-0b labels: counts by card kind, sentinel recall by missingness, precision.

uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_labelled_set.py
"""

from __future__ import annotations

import collections
import json
import re
from datetime import datetime

import numpy as np
from e00041_common import WORK_DIR
from e00041_diagnose import wilson

OUT = WORK_DIR / "real"
VIS = ("conserved", "one_sided")
CLOSE = re.compile(
    r"Logged close \(UK local\): (\S+) (\d\d):(\d\d); logged open \(UK local\): (\S+) (\d\d):(\d\d)"
)
MISSING_BINS = (
    (0.0, 1.0, "under 1"),
    (1.0, 2.0, "1 to 2"),
    (2.0, 3.0, "2 to 3"),
    (3.0, 1e9, "3 or more"),
)


def card_facts(card_id: str) -> dict[str, float]:
    """Unavailable values per hourly row, and the logged duration in hours, of one card."""
    text = (OUT / "cards" / f"{card_id}.md").read_text()
    m = CLOSE.search(text)
    assert m
    start = datetime.fromisoformat(f"{m[1]} {m[2]}:{m[3]}")
    end = datetime.fromisoformat(f"{m[4]} {m[5]}:{m[6]}")
    rows = sum(1 for x in text.splitlines() if x.startswith("| ") and "UTC" not in x)
    return {
        "na_per_row": text.count("n/a") / max(rows, 1),
        "hours": (end - start).total_seconds() / 3600,
    }


def _bin(value: float) -> str:
    return next(name for lo, hi, name in MISSING_BINS if lo <= value < hi)


def duration_band(hours: float) -> str:
    """The duration band used by the gate: 2 to 6 h, 6 to 24 h, over 1 d (under 2 h is its own)."""
    return (
        "under 2 h"
        if hours < 2
        else "2 to 6 h"
        if hours < 6
        else "6 to 24 h"
        if hours < 24
        else "over 1 d"
    )


def main() -> None:
    """Print the three summaries the write-up quotes."""
    meta = json.loads((OUT / "meta.json").read_text())
    final = json.loads((OUT / "labels_final.json").read_text())
    facts = {c: card_facts(c) for c in meta}
    kinds = collections.Counter((m["kind"], final[c]["visibility"]) for c, m in meta.items())
    for key in sorted(kinds):
        print(key, kinds[key])
    print("\nSentinel recall and placebo rate by unavailable values per row")
    real_ops = [facts[c]["na_per_row"] for c, m in meta.items() if m["kind"] == "real_operation"]
    real_share = {n: np.mean([_bin(v) == n for v in real_ops]) for *_, n in MISSING_BINS}
    weighted_recall, weighted_placebo = 0.0, 0.0
    for lo, hi, name in MISSING_BINS:
        inj = [
            c
            for c, m in meta.items()
            if m["kind"] == "sentinel_injected" and lo <= facts[c]["na_per_row"] < hi
        ]
        pla = [
            c
            for c, m in meta.items()
            if m["kind"] == "sentinel_placebo" and lo <= facts[c]["na_per_row"] < hi
        ]
        k_inj = sum(final[c]["visibility"] in VIS for c in inj)
        k_pla = sum(final[c]["visibility"] in VIS for c in pla)
        r = k_inj / len(inj) if inj else float("nan")
        p = k_pla / len(pla) if pla else float("nan")
        weighted_recall += real_share[name] * (0 if np.isnan(r) else r)
        weighted_placebo += real_share[name] * (0 if np.isnan(p) else p)
        print(
            f"  {name:10s} injected visible {k_inj}/{len(inj)} "
            f"placebo visible {k_pla}/{len(pla)} real share {real_share[name]:.2f}"
        )
    print(
        f"  reweighted to real operations: recall {weighted_recall:.3f}, "
        f"placebo rate {weighted_placebo:.3f}"
    )
    print("\nAlarm precision by detector (lower bracket: visible alarms over alarms)")
    by_det: dict[str, list[bool]] = collections.defaultdict(list)
    for c, m in meta.items():
        if m["kind"] == "alarm":
            by_det[m["detector"]].append(final[c]["visibility"] in VIS)
    for det, v in sorted(by_det.items()):
        lo, hi = wilson(sum(v), len(v))
        print(f"  {det}: {sum(v)}/{len(v)} ({lo:.2f} to {hi:.2f})")
    placebos = [
        final[c]["visibility"] in VIS for c, m in meta.items() if m["kind"] == "alarm_placebo"
    ]
    print(f"  alarm placebos visible: {sum(placebos)}/{len(placebos)}")
    print("\nAlarm visible by assumed duration band")
    bands: dict[str, list[bool]] = collections.defaultdict(list)
    for c, m in meta.items():
        if m["kind"] == "alarm":
            bands[duration_band(facts[c]["hours"])].append(final[c]["visibility"] in VIS)
    for band, v in sorted(bands.items()):
        print(f"  {band}: {sum(v)}/{len(v)}")
    print("\nReal operations labelled not visible, by logged duration band")
    nv = collections.Counter(
        duration_band(facts[c]["hours"])
        for c, m in meta.items()
        if m["kind"] == "real_operation" and final[c]["visibility"] == "not_visible"
    )
    for band, n in sorted(nv.items()):
        print(f"  {band}: {n}")


if __name__ == "__main__":
    main()
