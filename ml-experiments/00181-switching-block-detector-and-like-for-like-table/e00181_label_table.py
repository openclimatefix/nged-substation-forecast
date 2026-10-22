"""Part 2: one candidate label table for NGED review, written to the private briefs folder.

Four sources: the 17 block days, the 21 balanced-opposite alarms of the false-alarm check, the 29
multi-day drops of D8 (experiment 159), and the 66 labelled real transfers (experiment 41). Rows
use D-labels only. The file is never committed.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_label_table.py
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00181_block import best_z, series_scores
from e00181_blocks17 import block_days, site_series
from e00181_common import ADJACENCY, GRID_START, ML_EXPERIMENTS_DATA_DIR
from e00181_detectors import level_shift, robust_scale
from e00181_variants import load_both

BRIEFS: Final = ML_EXPERIMENTS_DATA_DIR / "briefs"
OUT: Final = BRIEFS / "sw_block_detector_label_table.csv"
MARGIN: Final[int] = 6
PARTNER_Z: Final[float] = 2.0
"""Noise units of 24-hour step an interconnected series needs to count as an opposite partner."""


def when(*, stamp: int) -> str:
    """UTC start of a half-hour as text."""
    return (GRID_START + timedelta(minutes=30 * int(stamp))).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    """Build and write the table."""
    label_of = {series: site for site, series in site_series().items()}
    panel, _ = load_both()
    residual = {s: v for s, v in panel["residual"].items() if np.isfinite(v).any()}
    scores = {s: series_scores(values=v)[0] for s, v in residual.items()}
    step24 = {s: level_shift(values=v, window=48) for s, v in residual.items()}
    scale24 = {s: robust_scale(shift=step24[s]) for s in residual}
    rows: list[dict] = []

    def partners(*, series: int, stamp: int, sign: float) -> str:
        found = []
        for n in ADJACENCY.get(series, ()):
            if n not in step24:
                continue
            window = -sign * step24[n][max(stamp - 1, 0) : stamp + 2] / scale24[n]
            if np.isfinite(window).any() and np.nanmax(window) >= PARTNER_Z:
                found.append(label_of[n])
        return " ".join(sorted(found))

    for day in block_days().iter_rows(named=True):
        s = day["series"]
        lo, hi = day["day_first_stamp"] - MARGIN, day["day_last_stamp"] + MARGIN + 1
        z, length = best_z(scores=scores[s])
        pick = lo + int(np.argmax(np.abs(z[lo:hi])))
        sign = float(np.sign(z[pick]))
        span = int(length[pick])
        flank = min(span, 48)
        inside = np.nanmean(residual[s][pick : pick + span])
        around = np.nanmean(
            np.concatenate(
                [residual[s][pick - flank : pick], residual[s][pick + span : pick + span + flank]]
            )
        )
        rows.append(
            {
                "onset_utc": when(stamp=pick),
                "end_utc": when(stamp=pick + span),
                "depth_mw": round(float(inside - around), 2),
                "duration_hours": span / 2,
                "pair": f"{label_of[s]} with {partners(series=s, stamp=pick, sign=sign)}".strip(),
                "source": "block day, experiment 175",
                "confidence": "medium" if abs(z[pick]) >= 3.5 else "low",
            }
        )
    flagged = pl.read_parquet(BRIEFS / "sw_false_alarms" / "work" / "tags.parquet").filter(
        pl.col("kind") == "alarm", pl.col("transfer_like")
    )
    series_of = site_series()
    for a in flagged.iter_rows(named=True):
        s = series_of[a["site"]]
        partner_text = partners(series=s, stamp=a["stamp"], sign=np.sign(a["step_mw"]))
        rows.append(
            {
                "onset_utc": when(stamp=a["stamp"]),
                "end_utc": "",
                "depth_mw": round(float(a["step_mw"]), 2),
                "duration_hours": "",
                "pair": f"{a['site']} with {partner_text}".strip(),
                "source": "balanced opposite alarm, false-alarm check",
                "confidence": "low",
            }
        )
    drops = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00159-switching-d8-multi-day-drops-recurrence-and-report"
        / "drops.parquet"
    )
    rows.extend(
        {
            "onset_utc": when(stamp=d["first_stamp"]),
            "end_utc": when(stamp=d["last_stamp"] + 1),
            "depth_mw": round(-float(d["depth_mw"]), 2),
            "duration_hours": round(24 * float(d["duration_days"]), 1),
            "pair": f"D8 with {d['best_neighbour']}" if d["best_neighbour"] else "D8",
            "source": "D8 multi-day drop, experiment 159",
            "confidence": "medium" if d["closures"] else "low",
        }
        for d in drops.iter_rows(named=True)
    )
    real = (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR
            / "00130-ro-nc-detector-on-the-gam-k-residual"
            / "B"
            / "real_results.parquet"
        )
        .filter(pl.col("setting") == "XL")
        .join(
            pl.read_parquet(
                ML_EXPERIMENTS_DATA_DIR
                / "00041-sw2-0a-labeller-gate"
                / "real"
                / "labelled_set.parquet"
            ).select("card_id", "donor", "receivers", "confidence"),
            on="card_id",
        )
    )
    for r in real.iter_rows(named=True):
        donor = label_of.get(int(r["donor"])) if r["donor"] is not None else None
        receivers = [label_of.get(int(x["series"])) for x in json.loads(r["receivers"])]
        group = " ".join(label_of[int(x)] for x in str(r["series"]).split(","))
        rows.append(
            {
                "onset_utc": when(stamp=r["onset"]),
                "end_utc": when(stamp=r["end"]),
                "depth_mw": r["moved_mw_p50"],
                "duration_hours": round(float(r["hours"]), 1),
                "pair": f"{donor} to {' '.join(x for x in receivers if x)}" if donor else group,
                "source": "logged and labelled transfer, experiment 41",
                "confidence": {"med": "medium"}.get(r["confidence"], r["confidence"]),
            }
        )
    table = pl.DataFrame(rows).sort("source", "onset_utc")
    table.write_csv(OUT)
    print(
        table.group_by("source").agg(n=pl.len(), n_confident=(pl.col("confidence") != "low").sum())
    )
    print(f"wrote {OUT} ({table.height} rows)")


if __name__ == "__main__":
    main()
