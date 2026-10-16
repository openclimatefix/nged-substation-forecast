"""Cohen's kappa and the agreement rule between the two Sonnet labellers on shared cards."""

from __future__ import annotations

import collections
import contextlib
import json

from e00041_common import TEST_SET, WORK_DIR
from e00041_gate import score
from e00041_score_sonnet import load

VIS = ("conserved", "one_sided")


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two label lists."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return (po - pe) / (1 - pe)


def agree(x: dict, y: dict) -> bool:
    """Whether two labels agree on visibility, onset, end and moved MW."""
    if x["visibility"] != y["visibility"]:
        return False
    if x["visibility"] not in VIS:
        return True
    dur = (x["end_stamp"] - x["onset_stamp"] + 1) / 2
    ok_onset = abs(x["onset_stamp"] - y["onset_stamp"]) <= 2
    ok_end = abs(x["end_stamp"] - y["end_stamp"]) <= max(4, 0.1 * 2 * dur)
    mx, my = x.get("moved_mw_p50"), y.get("moved_mw_p50")
    ok_mw = mx and my and abs(mx - my) <= 0.3 * max(mx, my)
    return bool(ok_onset and ok_end and ok_mw)


if __name__ == "__main__":
    one = load(which=TEST_SET, folder="labels_sonnet1")
    two = load(which=TEST_SET, folder="labels_sonnet2")
    ids = sorted(set(one) & set(two))
    print(len(ids), "shared cards")
    k4 = kappa([one[c]["visibility"] for c in ids], [two[c]["visibility"] for c in ids])
    k2 = kappa(
        ["v" if one[c]["visibility"] in VIS else "n" for c in ids],
        ["v" if two[c]["visibility"] in VIS else "n" for c in ids],
    )
    print(f"kappa 4-class {k4:.3f}, visible/not {k2:.3f}")
    print("full agreement rule:", sum(agree(one[c], two[c]) for c in ids), "of", len(ids))
    # bootstrap interval for the binary kappa
    import random

    rng = random.Random(0)
    ks = []
    for _ in range(2000):
        s = [rng.choice(ids) for _ in ids]
        with contextlib.suppress(ZeroDivisionError):
            ks.append(
                kappa(
                    ["v" if one[c]["visibility"] in VIS else "n" for c in s],
                    ["v" if two[c]["visibility"] in VIS else "n" for c in s],
                )
            )
    ks.sort()
    print("binary kappa 95% bootstrap", ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))])
    truths = json.loads((WORK_DIR / TEST_SET / "truths.json").read_text())
    sub = {c: t for c, t in truths.items() if c in two}
    print(
        "labeller 2 on its subset:",
        {k: round(v, 3) for k, v in score(labels=two, truths=sub).items()},
    )
    (WORK_DIR / TEST_SET / "labels_sonnet2_merged.json").write_text(json.dumps(two, indent=1))
