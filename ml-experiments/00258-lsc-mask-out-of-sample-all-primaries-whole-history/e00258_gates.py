# ruff: noqa: E501
"""The two known-answer gates, run before any arm is read.

G1: on the 13 pairs of load shift and Primary series that experiment 227 scored, the pooled K0
minus ORACLE score on the rows after each load shift is within 30% of 227's 5.339 points.

G2: on experiment 141's 44 stored injected worlds at the six Primaries, the pooled K0 minus PERFECT
score in the AFTER phase of transfers of 25% of load or more lasting over 1 day is within 30% of
experiment 208's 1.50 points.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00258_gates.py
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Final

import e00258_data as dat
import e00258_harness as h
import numpy as np
import polars as pl

E227_DIR: Final = dat.E227_DIR
WORLDS: Final = dat.DATA / "00141-switching-detector-fed-into-gam-k" / "worlds"
SERIES_OF_SITE: Final[dict[str, int]] = {
    "D5": 17,
    "D10": 10,
    "D11": 8,
    "D14": 16,
    "D16": 18,
    "D9": 7,
}
E227_GAIN: Final[float] = 5.339
E208_GAIN: Final[float] = 1.50
TOLERANCE: Final[float] = 0.30
BEFORE_ALARM_STAMPS: Final[int] = 96
AFTER_STAMPS: Final[int] = 96
SCORED_FIRST_STAMP: Final[int] = int(
    (datetime(2025, 7, 1, tzinfo=UTC) - dat.GRID_START).total_seconds() // 1800
)
"""The first stamp of experiment 208's scored year, used by a diagnostic only."""


def eleven_load_shifts() -> pl.DataFrame:
    """The pairs of load shift and series that experiment 227 scored in its scored and replicate months."""
    out = []
    for site in SERIES_OF_SITE:
        after = pl.read_parquet(E227_DIR / f"after_{site}.parquet")
        meta = pl.read_parquet(E227_DIR / f"meta_{site}.parquet").select("row", "run_period")
        kept = after.join(meta, on="row").filter(
            pl.col("run_period").is_in(["scored", "replicate"])
        )
        out.append(kept.group_by("card", "site").len())
    return pl.concat(out).with_columns(
        series=pl.col("site").replace_strict(SERIES_OF_SITE, return_dtype=pl.Int64)
    )


def gate_g1() -> dict:
    """Pooled K0 minus ORACLE on the rows after the 13 pairs, all leads and by lead band."""
    pairs = eleven_load_shifts()
    tables = [
        pl.read_parquet(h.HARNESS_DIR / f"after_{s}.parquet") for s in pairs["series"].unique()
    ]
    after = pl.concat(tables).join(pairs.select("card", "series"), on=["card", "series"])
    sums = (
        after.filter(pl.col("arm").is_in(["K0", "ORACLE"]))
        .group_by("arm", "lead_band")
        .agg(n=pl.col("n").sum(), qs=pl.col("qs").sum())
    )

    def gain(frame: pl.DataFrame) -> float:
        k0 = frame.filter(pl.col("arm") == "K0")
        oracle = frame.filter(pl.col("arm") == "ORACLE")
        return float(k0["qs"].sum() / k0["n"].sum() - oracle["qs"].sum() / oracle["n"].sum())

    out = {
        "pairs": pairs.height,
        "load_shifts": int(pairs["card"].n_unique()),
        "rows": int(sums.filter(pl.col("arm") == "K0")["n"].sum()),
        "k0_score": float(
            sums.filter(pl.col("arm") == "K0")["qs"].sum()
            / sums.filter(pl.col("arm") == "K0")["n"].sum()
        ),
        "gain_all_leads": gain(sums),
    }
    for band in range(len(h.LEAD_BAND_UPPER)):
        out[f"gain_lead_band_{band}"] = gain(sums.filter(pl.col("lead_band") == band))
    out["target"] = E227_GAIN
    out["passes"] = bool(
        abs(out["gain_all_leads"] - E227_GAIN) <= TOLERANCE * E227_GAIN
        and out["gain_all_leads"] > 0
    )
    return out


def _qs_rows(
    *,
    data: h.SeriesData,
    correction: np.ndarray,
    offsets: np.ndarray,
    band: np.ndarray,
    ks: np.ndarray,
) -> np.ndarray:
    """Quantile score of the origins `ks`, shape (len(ks), leads), against the clean readings."""
    _, _, targets = h.origin_grid()
    t = targets[ks]
    clipped = np.minimum(t, h.N_GRID - 1)
    actual = data.actual[clipped]
    location = data.expected[clipped] + correction[ks]
    cell = offsets[band[clipped], np.broadcast_to(h.lead_band_of_lead()[None, :], t.shape)]
    error = actual[..., None] - (location[..., None] + cell)
    pinball = np.maximum(h.LEVELS * error, (h.LEVELS - 1.0) * error)
    out = 2.0 * pinball.mean(axis=-1)
    out[(t >= h.N_GRID) | ~np.isfinite(actual)] = np.nan
    return out


def gate_g2(*, spread_scale: float = 1.0, first_stamp: int = 0) -> dict:
    """Pooled K0 minus PERFECT in the AFTER phase over the 44 worlds and six series.

    `spread_scale` multiplies the offset table and `first_stamp` drops transfers that start earlier
    (1 and 0 for the gate; other values are diagnostics).
    """
    band = h.band_grid()
    origins, cutoffs, targets = h.origin_grid()
    gains, k0_scores, perfect_scores, bands = [], [], [], []
    for series in SERIES_OF_SITE.values():
        data = h.load_series(series=series)
        offsets = np.load(h.HARNESS_DIR / f"series_{series}.npz")["offsets"] * spread_scale
        scale = 100.0 / data.capacity_mw
        for path in sorted(WORLDS.glob("world_*.npz")):
            seed = path.stem.split("_")[1]
            with np.load(path) as stored:
                delta = np.nan_to_num(stored[f"delta_{series}"])
                events = stored["events"]
            mine = [
                (int(r[2]), int(r[3]), float(r[4]))
                for r in events
                if np.any(delta[int(r[2]) : int(r[3])])
            ]
            mine = [
                (a, b, f) for a, b, f in mine if f >= 0.25 and (b - a) > 48 and a >= first_stamp
            ]
            if not mine:
                continue
            injected = dataclasses.replace(data, actual=data.actual + delta * scale)
            spans = [h.Span(a, b, a, max(b, a + 1)) for a, b, _ in mine]
            k0_trace = h.k0_trace(data=injected, band=band)
            k0 = h.k0_correction(trace=k0_trace, band=band)
            perfect = h.mask_correction(data=injected, spans=spans, band=band, k0=k0_trace)
            for a, b, _ in mine:
                ks = np.flatnonzero(
                    (cutoffs >= a + BEFORE_ALARM_STAMPS) & (origins <= b + AFTER_STAMPS)
                )
                if not len(ks):
                    continue
                t = targets[ks]
                window = (t >= b) & (t < b + AFTER_STAMPS)
                q0 = _qs_rows(data=data, correction=k0, offsets=offsets, band=band, ks=ks)
                qp = _qs_rows(data=data, correction=perfect, offsets=offsets, band=band, ks=ks)
                keep = window & np.isfinite(q0) & np.isfinite(qp)
                k0_scores.append(q0[keep])
                perfect_scores.append(qp[keep])
                gains.append(q0[keep] - qp[keep])
                bands.append(np.broadcast_to(h.lead_band_of_lead()[None, :], t.shape)[keep])
            print(f"series {series} world {seed}: {len(mine)} events", flush=True)
    gain = np.concatenate(gains)
    out = {
        "rows": len(gain),
        "k0_score": float(np.concatenate(k0_scores).mean()),
        "perfect_score": float(np.concatenate(perfect_scores).mean()),
        "gain": float(gain.mean()),
        "target": E208_GAIN,
    }
    out["passes"] = bool(abs(out["gain"] - E208_GAIN) <= TOLERANCE * E208_GAIN and out["gain"] > 0)
    band_all = np.concatenate(bands)
    for b in range(len(h.LEAD_BAND_UPPER)):
        out[f"gain_lead_band_{b}"] = float(gain[band_all == b].mean())
        out[f"rows_lead_band_{b}"] = int((band_all == b).sum())
    return out


def diagnostic() -> None:
    """Post hoc: G2's gain with the offset table scaled up, to test whether spread explains the miss."""
    results = {}
    for name, kwargs in {
        "spread_x1.2": {"spread_scale": 1.2},
        "spread_x1.5": {"spread_scale": 1.5},
        "scored_year_only": {"first_stamp": SCORED_FIRST_STAMP},
    }.items():
        results[name] = gate_g2(**kwargs)
        print("G2 diagnostic", name, json.dumps(results[name], indent=1), flush=True)
    (h.HARNESS_DIR / "g2_spread_diagnostic.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    """Run both gates and write `gates.json`."""
    g1 = gate_g1()
    print("G1", json.dumps(g1, indent=1), flush=True)
    g2 = gate_g2()
    print("G2", json.dumps(g2, indent=1), flush=True)
    (h.HARNESS_DIR / "gates.json").write_text(json.dumps({"G1": g1, "G2": g2}, indent=1))


if __name__ == "__main__":
    import sys

    diagnostic() if sys.argv[1:] == ["diagnostic"] else main()
