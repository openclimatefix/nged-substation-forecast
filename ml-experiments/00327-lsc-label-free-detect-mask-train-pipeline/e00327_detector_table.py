"""The detectors alone on the 35 labelled real load shifts: matches, alarms per match, strata.

Labels are read here only to score. A shift is matched when an alarm opens on either Primary of its
pair from 2 hours before its onset to its end. A false alarm opens outside every logged closure and
block day of its series. Intervals resample whole shifts, 20,000 draws.

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth \
        uv run --no-sync python e00327_detector_table.py
"""

from __future__ import annotations

import json
from typing import Final

import e00309_data as dat
import e00309_detect as det
import e00309_harness as h
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
OPEN_DELAY: Final[int] = 8
"""A span's `start` is the data cutoff; the alarm opens at the origin, 8 half-hours later."""
VARIANTS: Final[tuple[str, ...]] = ("A_plain", "A", "B_plain", "B", "E_plain", "E")
N_DRAWS: Final[int] = 20_000


def opens_of(*, name: str) -> dict[int, np.ndarray]:
    """Alarm opening stamps per series."""
    t = pl.read_parquet(WORK / "spans" / f"{name}.parquet")
    return {
        s: np.sort(t.filter(pl.col("series") == s)["start"].to_numpy() + OPEN_DELAY)
        for s in h.SERIES
    }


def main() -> None:
    """Print and store the detector-alone table."""
    shifts = dat.labelled_loads().filter(
        pl.col("series").list.eval(pl.element().is_in(list(h.SERIES))).list.any()
    )
    groups = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR / "00315-lsc-neighbour-residual-filter" / "ceiling.parquet"
    )
    g = {r["card_id"]: r for r in groups.iter_rows(named=True)}
    stratum = np.array([g[r["card_id"]]["stratum"] for r in shifts.iter_rows(named=True)])
    hours = shifts["hours"].to_numpy()
    rng = np.random.default_rng(327)
    blocks = {s: det.closure_blocks(series=s) | det.extra_blocks(series=s) for s in h.SERIES}
    out: dict = {}
    matched_by: dict[str, np.ndarray] = {}
    for name in VARIANTS:
        opens = opens_of(name=name)
        matched = np.zeros(shifts.height, dtype=bool)
        lag = np.full(shifts.height, np.nan)
        for i, r in enumerate(shifts.iter_rows(named=True)):
            best = np.inf
            for s in r["series"]:
                o = opens.get(s)
                if o is None:
                    continue
                hit = o[(o >= r["onset"] - det.MATCH_EARLY) & (o <= r["end"])]
                if len(hit):
                    best = min(best, hit.min() - r["onset"])
            if np.isfinite(best):
                matched[i], lag[i] = True, max(best, 0) / 2.0
        n_alarms = sum(len(o) for o in opens.values())
        false = {s: int((~blocks[s][np.clip(o, 0, h.N_GRID - 1)]).sum()) for s, o in opens.items()}
        n = shifts.height
        idx = rng.integers(0, n, size=(N_DRAWS, n))
        draws = matched[idx].sum(axis=1)
        out[name] = {
            "alarms": n_alarms,
            "false_alarms": sum(false.values()),
            "d8_false": false[26],
            "d9_false": false[7],
            "matched": int(matched.sum()),
            "matched_lo": float(np.quantile(draws, 0.025)),
            "matched_hi": float(np.quantile(draws, 0.975)),
            "matched_stratum": int((matched & stratum).sum()),
            "stratum_n": int(stratum.sum()),
            "median_lag_hours": float(np.nanmedian(lag)) if matched.any() else None,
            "alarms_per_match": float(n_alarms / max(matched.sum(), 1)),
            "matched_by_duration": {
                lab: int((matched & m).sum())
                for lab, m in (
                    ("24-48h", (hours <= 48)),
                    ("48-96h", (hours > 48) & (hours <= 96)),
                    ("over 96h", hours > 96),
                )
            },
            "n_by_duration": {
                lab: int(m.sum())
                for lab, m in (
                    ("24-48h", (hours <= 48)),
                    ("48-96h", (hours > 48) & (hours <= 96)),
                    ("over 96h", hours > 96),
                )
            },
        }
        matched_by[name] = matched
    for a, b in (("A", "B"), ("E", "A"), ("A", "A_plain")):
        out[f"{a}-{b}"] = det.bootstrap_difference(
            a=matched_by[a], b=matched_by[b], rng=rng, n_draws=N_DRAWS
        )
        sa = matched_by[a] & stratum
        sb = matched_by[b] & stratum
        out[f"{a}-{b}_stratum"] = det.bootstrap_difference(
            a=sa[stratum], b=sb[stratum], rng=rng, n_draws=N_DRAWS
        )
    (WORK / "results" / "detector_alone.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("shifts", shifts.height, "stratum", int(stratum.sum()))


if __name__ == "__main__":
    main()
