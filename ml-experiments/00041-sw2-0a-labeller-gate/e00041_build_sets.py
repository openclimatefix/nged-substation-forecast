"""Build the known-answer development and test sets: cards, truths, features, rules labels.

uv run python ml-experiments/00041-*/e00041_build_sets.py dev|test
"""

from __future__ import annotations

import json
import pickle
import sys
from typing import Final

import numpy as np
import polars as pl
from e00041_cards import (
    Truth,
    apply_injection,
    make_injection,
    make_placebo,
    real_operations,
    render_card,
    revert_injection,
)
from e00041_common import WORK_DIR
from e00041_features import CardSpec, Features, compute_features
from e00041_panel import INDEX, N_STAMPS, PAD_STAMPS, Panel, load_panel, time_of
from e00041_rules import label_card

COMPOSITION: Final[dict[str, dict[str, int]]] = {
    "dev": {"injected": 50, "placebo": 40, "jittered": 10, "seed": 11},
    "test": {"injected": 150, "placebo": 120, "jittered": 30, "seed": 22},
    "testbig": {"injected": 60, "placebo": 0, "jittered": 0, "seed": 33},
    "test2": {"injected": 150, "placebo": 120, "jittered": 30, "seed": 44},
    "testbig2": {"injected": 60, "placebo": 0, "jittered": 0, "seed": 55},
}


def hourly_table(*, panel: Panel, spec: CardSpec, features: Features) -> str:
    """Hourly means of obs, ref, and hindsight q10/q50/q90 per side around the event."""
    from e00041_panel import reference

    sides = [spec.side_a] + ([spec.side_b] if spec.side_b in INDEX else [])
    first, last = spec.first, spec.last
    span = last - first + 1
    lo, hi = first - 48, last + 49
    # events over 3 days: first and last 48 hours, daily means between
    windows = [(lo, first + 96), (last - 96, hi)] if span > 144 else [(lo, hi)]
    out = ["", "## Hourly values (UK-time-agnostic UTC hour ending; MW)", ""]
    header = (
        "| hour ending UTC | "
        + " | ".join(f"obs {s} | ref {s} | q10 {s} | q50 {s} | q90 {s}" for s in sides)
        + " |"
    )
    out += [header, "|---|" + "---|" * (5 * len(sides))]
    for a, b in windows:
        for h in range(a, b, 2):
            cells = []
            for sid in sides:
                row = INDEX[sid]
                st = np.arange(h, min(h + 2, N_STAMPS))
                ref = reference(panel=panel, row=row, stamps=st)
                vals = [
                    np.nanmean(panel.obs[row, st]),
                    np.nanmean(ref),
                    np.nanmean(panel.q10[row, st]),
                    np.nanmean(panel.q50[row, st]),
                    np.nanmean(panel.q90[row, st]),
                ]
                cells += ["n/a" if np.isnan(x) else f"{x:.2f}" for x in vals]
            marker = "*" if first <= h <= last else ""
            out.append(
                f"| {time_of(stamp=h + 1):%m-%d %H:%M}{marker} | " + " | ".join(cells) + " |"
            )
        if len(windows) == 2 and a == windows[0][0]:
            out.append("| ... (days between summarised below) |" + " |" * (5 * len(sides)))
    if len(windows) == 2:
        days = np.arange(first + 96, last - 96, 48)
        out += ["", "Daily means in the middle of the event (obs - ref, MW):", ""]
        out.append("| day starting UTC | " + " | ".join(f"excess {s}" for s in sides) + " |")
        out.append("|---|" + "---|" * len(sides))
        for d in days[:: max(1, len(days) // 20)]:
            st = np.arange(d, d + 48)
            row_vals = []
            for sid in sides:
                row = INDEX[sid]
                row_vals.append(
                    np.nanmean(panel.obs[row, st] - reference(panel=panel, row=row, stamps=st))
                )
            out.append(
                f"| {time_of(stamp=int(d)):%Y-%m-%d} | "
                + " | ".join("n/a" if np.isnan(x) else f"{x:.2f}" for x in row_vals)
                + " |"
            )
    out.append("")
    out.append("`*` marks rows inside the logged closure.")
    return "\n".join(out) + "\n"


def _draw_card(
    *, panel: Panel, ops: pl.DataFrame, rng: np.random.Generator, which: str, kind: str
) -> tuple[CardSpec, Truth, dict | None]:
    """Draw cards of `kind` until one is acceptable for the set `which`."""
    made = None
    while made is None:
        if kind == "placebo":
            m = make_placebo(panel=panel, ops=ops, rng=rng)
            made = None if m is None else (m[0], m[1], None)
        else:
            made = make_injection(panel=panel, ops=ops, rng=rng, jitter=(kind == "jittered"))
            if which.startswith("testbig") and made is not None:
                t = made[1]
                if not (
                    (t.moved_share_of_receiver or 0.0) >= 0.25
                    and (t.end_stamp or 0) - (t.onset_stamp or 0) + 1 >= 12
                ):
                    made = None
    return made


def main(which: str) -> None:
    """Build one set (dev, test or testbig): cards, truths, rules labels and features."""
    cfg = COMPOSITION[which]
    rng = np.random.default_rng(cfg["seed"])
    panel = load_panel()
    ops = real_operations()
    out_dir = WORK_DIR / which
    (out_dir / "cards").mkdir(parents=True, exist_ok=True)
    truths: dict[str, dict] = {}
    rules: dict[str, dict] = {}
    feats: dict[str, object] = {}
    specs: dict[str, object] = {}
    todo = (
        (["injected"] * cfg["injected"])
        + (["placebo"] * cfg["placebo"])
        + (["jittered"] * cfg["jittered"])
    )
    rng.shuffle(todo)
    for kind in todo:
        spec, truth, inj = _draw_card(panel=panel, ops=ops, rng=rng, which=which, kind=kind)
        rows = [INDEX[s] for s in (spec.side_a, spec.side_b) if s in INDEX]
        if inj is not None:
            rows = sorted(set(rows) | {INDEX[inj["donor"]]} | {INDEX[r] for r in inj["receivers"]})
        saved = panel.excluded[rows].copy()
        avoid = np.zeros(N_STAMPS, dtype=bool)
        lo, hi = max(spec.first - PAD_STAMPS - 48, 0), min(spec.last + PAD_STAMPS + 48, N_STAMPS)
        avoid[lo:hi] = True
        panel.excluded[
            rows, max(spec.first - PAD_STAMPS, 0) : min(spec.last + PAD_STAMPS + 1, N_STAMPS)
        ] = True
        if inj is not None:
            apply_injection(panel=panel, inj=inj)
        features = compute_features(
            panel=panel, spec=spec, avoid=avoid, seed=int(rng.integers(1 << 30))
        )
        label = label_card(panel=panel, spec=spec, features=features)
        text = render_card(spec=spec, features=features, panel=panel) + hourly_table(
            panel=panel, spec=spec, features=features
        )
        (out_dir / "cards" / f"{spec.card_id}.md").write_text(text)
        if inj is not None:
            revert_injection(panel=panel, inj=inj)
        panel.excluded[rows] = saved
        truths[spec.card_id] = truth.__dict__
        rules[spec.card_id] = label
        feats[spec.card_id] = features
        specs[spec.card_id] = spec
        print(len(truths), kind, spec.card_id, label["visibility"], flush=True)
    (out_dir / "truths.json").write_text(json.dumps(truths, indent=1, default=float))
    (out_dir / "rules_labels.json").write_text(json.dumps(rules, indent=1, default=float))
    with (out_dir / "features.pkl").open("wb") as handle:
        pickle.dump({"features": feats, "specs": specs}, handle)


if __name__ == "__main__":
    main(sys.argv[1])
