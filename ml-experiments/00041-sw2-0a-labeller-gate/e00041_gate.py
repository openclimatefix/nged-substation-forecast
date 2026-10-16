"""Score labels against known-answer truths with the pre-registered gate table."""

from __future__ import annotations

import numpy as np

VISIBLE = {"conserved", "one_sided"}
SHARE_BAND_MIN = 0.25
DURATION_BAND_MIN_STAMPS = 12


def is_hit(*, label: dict, truth: dict) -> bool:
    """A visible label whose onset is within 2 h of the truth and end within the end tolerance."""
    if label.get("visibility") not in VISIBLE or label.get("onset_stamp") is None:
        return False
    dur = truth["end_stamp"] - truth["onset_stamp"] + 1
    return abs(label["onset_stamp"] - truth["onset_stamp"]) <= 4 and abs(
        label["end_stamp"] - truth["end_stamp"]
    ) <= max(4, 0.1 * dur)


def score(*, labels: dict[str, dict], truths: dict[str, dict]) -> dict:
    """Return the gate table values for one labeller on the cards in `truths`."""
    placebo = [c for c, t in truths.items() if t["kind"] == "placebo" and c in labels]
    inj = [c for c, t in truths.items() if t["kind"] == "injected" and c in labels]
    jit = [c for c, t in truths.items() if t["kind"] == "jittered" and c in labels]
    out: dict = {}
    out["n_placebo"], out["n_injected"], out["n_jittered"] = len(placebo), len(inj), len(jit)
    out["placebo_visible"] = (
        np.mean([labels[c]["visibility"] in VISIBLE for c in placebo]) if placebo else float("nan")
    )
    out["placebo_conserved"] = (
        np.mean([labels[c]["visibility"] == "conserved" for c in placebo])
        if placebo
        else float("nan")
    )
    big = [
        c
        for c in inj
        if truths[c]["moved_share_of_receiver"] >= SHARE_BAND_MIN
        and truths[c]["end_stamp"] - truths[c]["onset_stamp"] + 1 >= 12
    ]
    out["n_big"] = len(big)
    out["recall_big"] = (
        np.mean([labels[c]["visibility"] in VISIBLE for c in big]) if big else float("nan")
    )
    out["recall_all"] = (
        np.mean([labels[c]["visibility"] in VISIBLE for c in inj]) if inj else float("nan")
    )
    found = [
        c
        for c in inj
        if labels[c]["visibility"] in VISIBLE and labels[c].get("onset_stamp") is not None
    ]
    out["onset_within_1h"] = (
        np.mean([abs(labels[c]["onset_stamp"] - truths[c]["onset_stamp"]) <= 2 for c in found])
        if found
        else float("nan")
    )
    out["end_within_tol"] = (
        np.mean(
            [
                abs(labels[c]["end_stamp"] - truths[c]["end_stamp"])
                <= max(4, 0.1 * (truths[c]["end_stamp"] - truths[c]["onset_stamp"] + 1))
                for c in found
            ]
        )
        if found
        else float("nan")
    )
    rel = [
        abs(labels[c]["moved_mw_p50"] - truths[c]["moved_mw_p50"]) / truths[c]["moved_mw_p50"]
        for c in found
        if labels[c].get("moved_mw_p50") is not None and truths[c]["moved_mw_p50"] > 0
    ]
    out["moved_mw_median_rel_err"] = float(np.median(rel)) if rel else float("nan")
    named = []
    for c in found:
        t = truths[c]
        if t["unmetered_receiver"]:
            continue
        named.extend(
            r["series"] in [x["series"] for x in labels[c].get("receivers") or []]
            for r in t["receivers"]
            if r["share"] >= 0.30
        )
    out["receivers_named_share_ge30"] = float(np.mean(named)) if named else float("nan")
    near = [
        abs(labels[c]["onset_stamp"] - truths[c]["onset_stamp"])
        < 2 * abs(truths[c]["log_jitter_hours"]) * 1.0 - 1e-9
        for c in jit
        if labels[c]["visibility"] in VISIBLE and labels[c].get("onset_stamp") is not None
    ]
    out["jittered_onset_nearer_than_log"] = float(np.mean(near)) if near else float("nan")
    # event-level F1 over all cards: hit = visible, onset and end within tolerance
    real = inj + jit
    tp = sum(is_hit(label=labels[c], truth=truths[c]) for c in real)
    pred_pos = sum(labels[c]["visibility"] in VISIBLE for c in real + placebo)
    precision = tp / pred_pos if pred_pos else float("nan")
    recall = tp / len(real) if real else float("nan")
    out["event_precision"], out["event_recall"] = precision, recall
    out["event_f1"] = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return out
