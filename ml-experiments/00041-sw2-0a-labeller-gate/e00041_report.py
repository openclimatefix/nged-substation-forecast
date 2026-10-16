"""Print every number the SW2-0a write-up quotes: gate table, per-band recall, token cost."""

from __future__ import annotations

import json

import numpy as np
from e00041_common import BIG_SET, TEST_SET, WORK_DIR
from e00041_diagnose import wilson
from e00041_gate import score
from e00041_score_sonnet import load

VIS = ("conserved", "one_sided")


def band(tr: dict) -> tuple[str, str]:
    """The (share-of-receiver band, duration band) of one truth row."""
    hours = (tr["end_stamp"] - tr["onset_stamp"] + 1) / 2
    share = tr["moved_share_of_receiver"]
    return (
        "under 10%" if share < 0.10 else "10% to 25%" if share < 0.25 else "25% or more",
        "2 to 6 h" if hours < 6 else "6 to 24 h" if hours < 24 else "over 1 d",
    )


def main() -> None:
    """Print every number the write-up quotes."""
    truths = {
        **json.loads((WORK_DIR / TEST_SET / "truths.json").read_text()),
        **json.loads((WORK_DIR / BIG_SET / "truths.json").read_text()),
    }
    labellers = {
        "rules": {
            **json.loads((WORK_DIR / TEST_SET / "rules_labels.json").read_text()),
            **json.loads((WORK_DIR / BIG_SET / "rules_labels.json").read_text()),
        },
        "sonnet": {
            **load(which=TEST_SET, folder="labels_sonnet1"),
            **load(which=BIG_SET, folder="labels_sonnet1"),
        },
    }
    test_ids = set(json.loads((WORK_DIR / TEST_SET / "truths.json").read_text()))
    for name, labels in labellers.items():
        print(f"== {name}: 300 test cards")
        t = {c: truths[c] for c in test_ids}
        res = score(labels={c: labels[c] for c in test_ids}, truths=t)
        for k, v in res.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        # counts for Wilson intervals
        pl = [c for c in test_ids if t[c]["kind"] == "placebo"]
        k = sum(labels[c]["visibility"] in VIS for c in pl)
        print(f"  placebo visible {k}/{len(pl)} Wilson {wilson(k, len(pl))}")
        k = sum(labels[c]["visibility"] == "conserved" for c in pl)
        print(f"  placebo conserved {k}/{len(pl)}")
        big_all = [
            c
            for c, tr in truths.items()
            if tr["kind"] == "injected"
            and tr["moved_share_of_receiver"] >= 0.25
            and tr["end_stamp"] - tr["onset_stamp"] + 1 >= 12
        ]
        k = sum(labels[c]["visibility"] in VIS for c in big_all)
        print(
            f"  big-band recall pooled test+testbig {k}/{len(big_all)} = "
            f"{k / len(big_all):.3f} Wilson "
            f"{tuple(round(x, 3) for x in wilson(k, len(big_all)))}"
        )
        for name_, keep in (
            ("a metered receiver", lambda tr: bool(tr["receivers"])),
            ("no metered receiver (donor step only)", lambda tr: not tr["receivers"]),
        ):
            sub = [c for c in big_all if keep(truths[c])]
            k = sum(labels[c]["visibility"] in VIS for c in sub)
            lo, hi = wilson(k, len(sub))
            print(
                f"  big-band recall, {name_}: {k}/{len(sub)} = {k / len(sub):.3f} "
                f"({lo:.2f} to {hi:.2f})"
            )
        found = [
            c
            for c in test_ids
            if t[c]["kind"] == "injected"
            and labels[c]["visibility"] in VIS
            and labels[c].get("onset_stamp") is not None
        ]
        k = sum(abs(labels[c]["onset_stamp"] - t[c]["onset_stamp"]) <= 2 for c in found)
        print(
            f"  onset within 1 h {k}/{len(found)} Wilson "
            f"{tuple(round(x, 3) for x in wilson(k, len(found)))}"
        )
        [c for c in test_ids if t[c]["kind"] == "jittered"]
        # per-band recall on all injected cards in test + testbig
        bands: dict[tuple[str, str], list[bool]] = {}
        for c, tr in truths.items():
            if tr["kind"] == "injected":
                bands.setdefault(band(tr), []).append(labels[c]["visibility"] in VIS)
        print("  recall by band (share of receiver load, duration), test+testbig injected:")
        for key in sorted(
            bands,
            key=lambda b: (
                ("under 10%", "10% to 25%", "25% or more").index(b[0]) * 3
                + ("2 to 6 h", "6 to 24 h", "over 1 d").index(b[1])
            ),
        ):
            v = bands[key]
            lo, hi = wilson(sum(v), len(v))
            print(
                f"    {key[0]:12s} {key[1]:9s} {sum(v):3d}/{len(v):3d} = {np.mean(v):.2f} "
                f"({lo:.2f} to {hi:.2f})"
            )
    cost = [line.split("\t") for line in (WORK_DIR / "token_cost.tsv").read_text().splitlines()[1:]]
    total = sum(int(c[2]) for c in cost)
    print(
        "subagent tokens total:",
        total,
        "labeller 1:",
        sum(int(c[2]) for c in cost if c[0].startswith("sonnet1")),
        "labeller 2:",
        sum(int(c[2]) for c in cost if c[0].startswith("sonnet2")),
    )


if __name__ == "__main__":
    main()
