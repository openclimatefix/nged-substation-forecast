"""Descriptive checks on the real stream, with no labelled MW in any score.

Part A scores forecasts issued inside a logged real transfer, with targets after the logged end,
against observed power: once the transfer has ended the observed power is the NRA, so no labelled
MW is needed. Part B is a no-harm contrast: every real row outside the logged transfers (and their
48 hours after), scored against observed power. Arms are K0, SUB-DET (the detected MW of the real
stream, with experiment 151's knowledge stamps), MASK (readings inside known intervals treated as
missing), and GAM-only (no correction). The hand-filtered switching mask is not applied to part A,
because it covers these transfers.

    OMP_NUM_THREADS=1 uv run --no-sync python e00151_real.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00151_fed as fed
import e00151_nra as nra
import numpy as np
import polars as pl
from e00151_common import GRID_START, LABELLED_PATH, WORK_DIR
from e00151_g_arms import quantile_score

INTERVALS: Final[Path] = (
    WORK_DIR.parent / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
ARMS: Final[tuple[str, ...]] = ("K0", "SUB_DET", "MASK", "GAM_ONLY")
AFTER: Final[int] = 336
"""One week of target time after the logged end (part A)."""


def stamp(when: Any) -> int:
    """The detector stamp of a UTC time."""
    return int((when - GRID_START).total_seconds() // 1800)


def logged_transfers() -> pl.DataFrame:
    """Labelled real transfers with their detector-stamp onset and end and the series involved."""
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    rows = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        donor = int(row["donor"]) if row["donor"] not in (None, "") else None
        series = {s for s in (donor, row["side_a"], row["side_b"], *receivers) if s is not None}
        rows.append(
            {
                "card": row["card_id"],
                "onset": stamp(row["onset_utc"]),
                "end": stamp(row["end_utc"]) + 1,
                "series": sorted(int(s) for s in series),
            }
        )
    return pl.DataFrame(rows)


def month_difference(*, months: np.ndarray, diff: np.ndarray) -> tuple[float, float, float]:
    """Mean difference with a 95% interval from resampling whole calendar months."""
    labels = np.unique(months)
    sums = np.array([diff[months == m].sum() for m in labels])
    counts = np.array([(months == m).sum() for m in labels], dtype=float)
    rng = np.random.default_rng(151)
    picks = rng.integers(0, len(labels), size=(10_000, len(labels)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(diff.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:  # noqa: PLR0915
    """Print part A per transfer, and part B pooled."""
    transfers = logged_transfers()
    intervals = pl.read_parquet(INTERVALS)
    a_rows, b_frames = [], []
    for site, series in fed.SERIES_OF_SITE.items():
        data = fed.get_site(site)
        offset = fed.stamp_offset(data)
        n = len(data.stream)
        spans = nra.detected_spans(data=data, site=site, intervals=intervals)
        cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
        stream = data.stream
        spread = fed.k0_spread(site)

        def corrections(
            rows: np.ndarray,
            *,
            stream: Any = stream,
            data: Any = data,
            spans: Any = spans,
            cutoffs: Any = cutoffs,
            n: Any = n,
        ) -> Any:
            return {
                "K0": fed.correction_for(
                    data=data, stream=stream, prior=np.zeros(n), multiplier=1.0
                ),
                "SUB_DET": nra.subtraction_correction(
                    data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows
                ),
                "MASK": nra.subtraction_correction(
                    data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
                ),
                "GAM_ONLY": np.zeros(len(data.run_of_row)),
            }

        def score(rows: np.ndarray, corr: dict, *, data: Any = data, spread: Any = spread) -> dict:
            out = {}
            for arm, c in corr.items():
                q = fed.quantiles_at(data=data, correction=c, spread=spread, rows=rows)
                out[arm] = quantile_score(actual=data.actual[rows], quantiles=q)
            return out

        mine = transfers.filter(pl.col("series").list.contains(series))
        blocked = np.zeros(n, dtype=bool)
        for row in transfers.filter(pl.col("series").list.contains(series)).iter_rows(named=True):
            lo, hi = row["onset"] + offset, row["end"] + offset
            blocked[max(lo, 0) : min(hi + nra.TAIL, n)] = True
        for row in mine.iter_rows(named=True):
            lo, hi = row["onset"] + offset, row["end"] + offset
            if lo < 0 or hi + AFTER >= n:
                continue
            target = np.zeros(n, dtype=bool)
            target[hi : hi + AFTER] = True
            rows = np.flatnonzero(target[data.position] & np.isfinite(data.actual))
            rows = rows[(cutoffs[rows] >= lo) & (cutoffs[rows] < hi)]
            if not len(rows):
                continue
            scored = score(rows, corrections(rows))
            a_rows.append(
                {"card": row["card"], "site": site, "rows": len(rows)}
                | {arm: float(v.mean()) for arm, v in scored.items()}
            )
        # part B: all real rows outside the logged transfers, the hand-filtered mask applied
        for period in ("replicate", "scored"):
            rows = fed.rows_where(data=data, mask_by_position=~blocked, period=period)
            if not len(rows):
                continue
            scored = score(rows, corrections(rows))
            run = data.runs[data.run_of_row[rows]]
            b_frames.append(
                pl.DataFrame(
                    {"site": site, "run": run, "lead_hours": data.lead_hours[rows]} | dict(scored)
                )
            )
    pl.Config.set_tbl_rows(40)
    a = pl.DataFrame(a_rows, infer_schema_length=None)
    print("== part A: forecasts issued inside a logged real transfer, targets after its end ==")
    print(a)
    if a.height:
        w = a["rows"]
        print(
            "row-weighted mean quantile score:",
            {arm: round(float((a[arm] * w).sum() / w.sum()), 3) for arm in ARMS},  # ty: ignore[unsupported-operator]
        )
    b = pl.concat(b_frames)
    b = b.with_columns(
        month=(
            pl.lit(datetime(2024, 4, 1, 9))  # noqa: DTZ001
            + pl.duration(days=pl.col("run"))
            + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
        ).dt.strftime("%Y-%m")
    )
    print(f"\n== part B: no-harm contrast on {b.height} real rows outside logged transfers ==")
    months = b["month"].to_numpy()
    for arm in ARMS:
        print(f"  {arm:9s} mean quantile score {b[arm].mean():.3f}")
    for arm in ("SUB_DET", "MASK", "GAM_ONLY"):
        mean, lo, hi = month_difference(months=months, diff=(b[arm] - b["K0"]).to_numpy())
        print(f"  {arm} minus K0: {mean:+.3f} (95% {lo:+.3f} to {hi:+.3f}); positive is worse")
    a.write_parquet(WORK_DIR / "real_a.parquet")


if __name__ == "__main__":
    main()
