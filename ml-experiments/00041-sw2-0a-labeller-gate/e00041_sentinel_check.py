"""Check that hidden known-answer cards cannot be told from real-operation cards by card text.

Compares real-operation cards with sentinel cards on the minute of the log time, an unmetered
side B, the clock hour, the logged duration, the number of neighbour rows, the number of
unavailable values, and the card's length, with a two-sample test for each.

    uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_sentinel_check.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
from e00041_common import WORK_DIR
from scipy import stats

OUT = WORK_DIR / "real"
CLOSE = re.compile(
    r"Logged close \(UK local\): (\S+) (\d\d):(\d\d); logged open \(UK local\): (\S+) (\d\d):(\d\d)"
)


def features_of(text: str) -> dict[str, float]:
    """Pull the fields a labeller could use as a giveaway out of one card's text."""
    m = CLOSE.search(text)
    assert m
    start = datetime.fromisoformat(f"{m[1]} {m[2]}:{m[3]}")
    end = datetime.fromisoformat(f"{m[4]} {m[5]}:{m[6]}")
    return {
        "close_minute_mod30": int(m[3]) % 30,
        "close_minute_is_05_or_35": float(int(m[3]) in (5, 35)),
        "side_b_unmetered": float("side B: not metered" in text),
        "close_hour": int(m[2]),
        "night": float(int(m[2]) < 6 or int(m[2]) >= 22),
        "duration_hours": (end - start).total_seconds() / 3600,
        "duration_is_whole_hours": float(int(m[6]) == int(m[3])),
        "neighbour_rows": sum(
            1 for line in text.splitlines() if re.match(r"\| \d+ \| \d+ \| M", line)
        ),
        "n_unavailable": text.count("n/a"),
        "unavailable_per_row": text.count("n/a")
        / max(sum(1 for x in text.splitlines() if x.startswith("| ") and "UTC" not in x), 1),
        "card_lines": len(text.splitlines()),
    }


def main() -> None:
    """Print each field's mean for real and sentinel cards, with a two-sample p-value."""
    meta = json.loads((OUT / "meta.json").read_text())
    groups: dict[str, list[dict[str, float]]] = {"real operation": [], "sentinel": []}
    for card_id, info in meta.items():
        kind = info["kind"]
        if kind == "real_operation":
            groups["real operation"].append(
                features_of((OUT / "cards" / f"{card_id}.md").read_text())
            )
        elif kind.startswith("sentinel"):
            groups["sentinel"].append(features_of((OUT / "cards" / f"{card_id}.md").read_text()))
    print({k: len(v) for k, v in groups.items()})
    for name in groups["sentinel"][0]:
        real = np.array([f[name] for f in groups["real operation"]])
        sentinel = np.array([f[name] for f in groups["sentinel"]])
        p = stats.ks_2samp(real, sentinel).pvalue
        print(
            f"{name:26s} real {real.mean():8.3f}  sentinel {sentinel.mean():8.3f}  KS p = {p:.3f}"
        )


if __name__ == "__main__":
    main()
