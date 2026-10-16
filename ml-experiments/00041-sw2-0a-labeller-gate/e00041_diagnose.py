"""Wilson intervals and the receivers-named breakdown for a labeller's labels on the test cards."""

from __future__ import annotations

import collections
import json
import sys
from math import sqrt

from e00041_common import BIG_SET, MVA_SERIES, TEST_SET, WORK_DIR

VISIBLE = ("conserved", "one_sided")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main(labels_name: str) -> None:
    """Print per-band recall diagnostics for the labels in `labels_name`."""
    for which in (TEST_SET, BIG_SET):
        d = WORK_DIR / which
        t = json.loads((d / "truths.json").read_text())
        path = d / labels_name
        if not path.exists():
            continue
        lab = json.loads((path).read_text())
        big = [
            c
            for c, tr in t.items()
            if tr["kind"] == "injected"
            and tr["moved_share_of_receiver"] >= 0.25
            and tr["end_stamp"] - tr["onset_stamp"] + 1 >= 12
            and c in lab
        ]
        k = sum(lab[c]["visibility"] in VISIBLE for c in big)
        lo, hi = wilson(k, len(big))
        print(
            f"{which} big-band recall {k}/{len(big)} = {k / len(big):.2f} (Wilson {lo:.2f} to "
            f"{hi:.2f})"
        )
    d = WORK_DIR / TEST_SET
    t = json.loads((d / "truths.json").read_text())
    lab = json.loads((d / labels_name).read_text())
    cnt: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for c, tr in t.items():
        if tr["kind"] != "injected" or c not in lab:
            continue
        label = lab[c]
        if label["visibility"] not in VISIBLE or label.get("onset_stamp") is None:
            continue
        names = [x["series"] for x in label.get("receivers") or []]
        for r in tr["receivers"]:
            if r["share"] < 0.3:
                continue
            keys = [
                "card with an unmetered receiver"
                if tr["unmetered_receiver"]
                else "all receivers metered",
                f"{len(tr['receivers'])} metered receiver(s) in truth",
                "receiver on MVA series" if r["series"] in MVA_SERIES else "receiver on MW series",
                label["visibility"],
            ]
            for key in keys:
                cnt[key][0] += r["series"] in names
                cnt[key][1] += 1
    for k_, (a, b) in sorted(cnt.items()):
        lo, hi = wilson(a, b)
        print(f"{k_}: {a}/{b} = {a / b:.2f} (Wilson {lo:.2f} to {hi:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
