"""Merge the labellers' and the adjudicator's labels into the SW2-0b labelled set.

Writes `real/labels_final.json` (one final label per card) and `real/labelled_set.parquet`, with
the two moved-load shares that Python computes (the labeller does not).

    uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_finalize.py
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import polars as pl
from e00041_common import WORK_DIR
from e00041_panel import INDEX, Panel, excess, load_panel, reference, time_of

OUT = WORK_DIR / "real"
VIS = ("conserved", "one_sided")
DONOR_HISTORY_STAMPS = 365 * 48


def _load_jsonl(folder: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((OUT / folder).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["card_id"]] = row
    return rows


def _donor_range(*, panel: Panel, donor: int, onset: int) -> float:
    """P99 minus P1 of the donor's half-hourly power over the 365 days before the event."""
    row = INDEX[donor]
    lo = max(onset - DONOR_HISTORY_STAMPS, 0)
    values = np.where(panel.excluded[row, lo:onset], np.nan, panel.obs[row, lo:onset])
    values = values[~np.isnan(values)]
    return (
        float(np.percentile(values, 99) - np.percentile(values, 1))
        if len(values) > 100
        else float("nan")
    )


def _shares(*, panel: Panel, label: dict) -> tuple[float, float]:
    donor = label.get("donor")
    moved = label.get("moved_mw_p50")
    onset, end = label.get("onset_stamp"), label.get("end_stamp")
    if moved is None or onset is None or end is None or donor not in INDEX:
        return float("nan"), float("nan")
    donor_range = _donor_range(panel=panel, donor=int(donor), onset=int(onset))
    receivers = [r["series"] for r in label.get("receivers") or [] if r["series"] in INDEX]
    stamps = np.arange(int(onset), int(end) + 1)
    load = (
        float(
            np.nansum(
                [np.nanmean(reference(panel=panel, row=INDEX[r], stamps=stamps)) for r in receivers]
            )
        )
        if receivers
        else float("nan")
    )
    return (moved / load if load else float("nan")), (
        moved / donor_range if donor_range else float("nan")
    )


def _missing_share(*, panel: Panel, sides: list[int | None], first: int, last: int) -> float:
    """Largest share, over the card's metered sides, of event half-hours with no excess."""
    shares = [
        float(np.mean(np.isnan(excess(panel=panel, row=INDEX[s], lo=first, hi=last + 1))))
        for s in sides
        if s in INDEX
    ]
    return max(shares) if shares else float("nan")


def main() -> None:
    """Merge the labels and write the final set."""
    meta = json.loads((OUT / "meta.json").read_text())
    one = json.loads((OUT / "labels_sonnet1_merged.json").read_text())
    opus = _load_jsonl("labels_opus")
    adjudicated = set(json.loads((OUT / "adjudicate_cards.json").read_text()))
    final = {c: opus.get(c, one[c]) if c in adjudicated else one[c] for c in meta}
    missing = adjudicated - set(opus)
    print("adjudicated cards", len(adjudicated), "returned", len(opus), "missing", len(missing))
    (OUT / "labels_final.json").write_text(json.dumps(final, indent=1))
    panel = load_panel()
    specs = pickle.loads((OUT / "features.pkl").read_bytes())["specs"]
    rows = []
    for card_id, label in final.items():
        share_receiver, share_donor = _shares(panel=panel, label=label)
        spec = specs[card_id]
        rows.append(
            {
                "card_id": card_id,
                "kind": meta[card_id]["kind"],
                "side_a": spec.side_a,
                "side_b": spec.side_b,
                "logged_close_utc": spec.close_utc,
                "logged_open_utc": spec.open_utc,
                "visibility": label["visibility"],
                "confidence": label.get("confidence"),
                "onset_utc": time_of(stamp=int(label["onset_stamp"]))
                if label.get("onset_stamp") is not None
                else None,
                "end_utc": time_of(stamp=int(label["end_stamp"]))
                if label.get("end_stamp") is not None
                else None,
                "donor": label.get("donor"),
                "receivers": json.dumps(label.get("receivers") or []),
                "moved_mw_p50": label.get("moved_mw_p50"),
                "moved_mw_p10": label.get("moved_mw_p10"),
                "moved_mw_p90": label.get("moved_mw_p90"),
                "moved_share_of_receiver": share_receiver,
                "moved_share_of_donor_range": share_donor,
                "max_side_missing_share": _missing_share(
                    panel=panel, sides=[spec.side_a, spec.side_b], first=spec.first, last=spec.last
                ),
                "adjudicated": card_id in adjudicated,
                "is_sentinel": meta[card_id]["kind"].startswith("sentinel"),
            }
        )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    frame.write_parquet(OUT / "labelled_set.parquet")
    real = frame.filter(~pl.col("is_sentinel"), pl.col("visibility").is_in(list(VIS)))
    print("visible real events:", real.height)
    print(real.group_by("kind", "visibility").len().sort("kind", "visibility"))


if __name__ == "__main__":
    main()
