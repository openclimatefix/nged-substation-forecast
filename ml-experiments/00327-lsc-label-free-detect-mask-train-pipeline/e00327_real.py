# ruff: noqa: E501
"""Score every arm on the real record: per-unit metric sums by row group, for the bootstrap.

Each arm is a demand-model forecast (`forecast/<arm>`), the Kalman filter of
`studies.dp_assimilation` with the arm's alarm spans set aside (experiment 309's harness), and an
offset table fitted on that arm's own odd-month residuals over all half-hours. Arms: C (no mask),
D (the hand labels masked, reference), A (label-free detector, pass 1), B (24 hour residual
detector), E (pass 2), and AF (post hoc: the forecast of C with A's spans in the filter only).
Scored rows are even-month targets that no arm masks. Labels are read only to define the row groups.

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth:ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual \
        uv run --no-sync python e00327_real.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Final

import e00130_gam_panel as g  # ty: ignore[unresolved-import]
import e00309_data as dat
import e00309_harness as h
import e00309_real as real
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    MIN_PRIOR_SPREAD,
    FilterHyperparameters,
    fit_site_hyperparameters,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
ARMS: Final[dict[str, tuple[str, str]]] = {
    "C": ("C", "none"),
    "D": ("D", "oracle"),
    "A": ("A", "A"),
    "B": ("B", "B"),
    "E": ("E", "E"),
    "AF": ("C", "A"),
}
"""Arm -> (forecast folder, spans source)."""
METRICS: Final[tuple[str, ...]] = ("qs", "pin95", "pin99", "pin01", "mae", "below01", "above99")
AFTER_SHORT: Final[int] = 96
AFTER_LONG: Final[int] = 336
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})


def spans_of(*, source: str, series: int) -> list[h.Span]:
    """The arm's spans for one series."""
    if source == "none":
        return []
    if source == "oracle":
        return h.oracle_spans(events=h.labelled_events(), series=series)
    table = pl.read_parquet(WORK / "spans" / f"{source}.parquet").filter(pl.col("series") == series)
    return [
        h.Span(int(r["onset"]), int(r["end"]), int(r["start"]), int(r["known"]))
        for r in table.iter_rows(named=True)
    ]


def fitted_hyper(
    *, residual_points: np.ndarray, spans: list[h.Span], band: np.ndarray, site: str
) -> FilterHyperparameters:
    """Filter hyperparameters on the arm's odd-month residual with its spans set aside."""
    tuned = json.loads((g.PART_4_DIR / "work" / "tuned.json").read_text())["filter_hyper"]["a0"]
    stacked = np.array(list(tuned.values()))
    prior_mean = stacked.mean(axis=0)
    prior_spread = np.maximum(stacked.std(axis=0, ddof=1), MIN_PRIOR_SPREAD)
    resid = residual_points.copy()
    for s in spans:
        resid[s.onset : s.end] = np.nan
    odd = np.isin(h.utc_month_grid(), list(h.TUNING_MONTHS))
    resid = np.where(odd, resid, np.nan)
    result = fit_site_hyperparameters(
        residual=resid, band=band, prior_mean=prior_mean, prior_spread=prior_spread,
        start=np.array(tuned[site]) if site in tuned else prior_mean,
    )  # fmt: skip
    return FilterHyperparameters.from_array(values=result.as_array())


def arm_metric_arrays(
    *, data: h.SeriesData, correction: np.ndarray, offsets: np.ndarray, band: np.ndarray
) -> dict[str, np.ndarray]:
    """Per (origin, lead) metrics, NaN where the target has no reading."""
    _, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    actual = data.actual[clipped].astype(np.float64)
    location = data.expected[clipped] + correction
    cell = offsets[band[clipped], np.broadcast_to(h.lead_band_of_lead()[None, :], targets.shape)]
    bad = (targets >= h.N_GRID) | ~np.isfinite(actual)
    quant = location[..., None] + cell
    error = actual[..., None] - quant
    pin = np.maximum(h.LEVELS * error, (h.LEVELS - 1.0) * error)
    out = {
        "qs": 2.0 * pin.mean(axis=-1),
        "pin95": pin[..., 9],
        "pin99": pin[..., 10],
        "pin01": pin[..., 0],
        "mae": np.abs(quant[..., 5] - actual),
        "below01": (actual < quant[..., 0]).astype(float),
        "above99": (actual > quant[..., 10]).astype(float),
    }
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


def after_masks(
    *, series: int, origins: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows whose target is within 48 hours and within 7 days after a labelled shift, and its label."""
    shifts = (
        dat.labelled_loads()
        .with_row_index("shift_index")
        .filter(pl.col("series").list.contains(series))
    )
    short = np.zeros(targets.shape, dtype=bool)
    long = np.zeros(targets.shape, dtype=bool)
    label = np.full(targets.shape, -1, dtype=np.int64)
    for r in shifts.iter_rows(named=True):
        base = (origins[:, None] >= r["end"]) & (targets >= r["end"])
        s = base & (targets < r["end"] + AFTER_SHORT)
        lg = base & (targets < r["end"] + AFTER_LONG)
        short |= s
        long |= lg
        label = np.where(lg & (label < 0), r["shift_index"], label)
    return short, long, label


def series_units(series: int) -> pl.DataFrame:
    """Per-unit metric sums of every arm and row group for one series."""
    caps = h.capacities()
    site = g.site_of_series()[series][0]
    band = h.band_grid()
    month = h.utc_month_grid()
    origins, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    power = np.load(WORK / "power" / f"power_{series}.npy") * 100.0 / caps[series]
    spans = {name: spans_of(source=src, series=series) for name, (_, src) in ARMS.items()}
    masked = np.zeros(h.N_GRID, dtype=bool)
    for name in ("A", "B", "E", "D"):
        for s in spans[name]:
            masked[s.onset : s.end] = True
    masks, _, _ = real.group_masks(series=series, origins=origins, targets=targets, state={})
    short, long, shift_label = after_masks(series=series, origins=origins, targets=targets)
    even = ~np.isin(month[clipped], list(h.TUNING_MONTHS))
    ok_rows = ~masked[clipped] & even
    groups = {
        "ordinary": masks["ordinary"] & ok_rows,
        "after48": short & ok_rows,
        "after7d": long & ok_rows,
        "all_rows": ok_rows & (targets < h.N_GRID),
    }
    year_month = dat.stamp_year_month()[clipped]
    winter = np.isin(month[clipped], list(WINTER))
    rows = []
    mets = {}
    for name, (fdir, _) in ARMS.items():
        expected = (
            np.load(WORK / "forecast" / fdir / f"forecast_{series}.npy") * 100.0 / caps[series]
        )
        hyper = fitted_hyper(
            residual_points=power - expected, spans=spans[name], band=band, site=site
        )
        data = h.SeriesData(series, caps[series], power, expected, hyper)
        k0 = h.k0_trace(data=data, band=band)
        corr = h.mask_correction(data=data, spans=spans[name], band=band, k0=k0).astype(np.float64)
        offsets = h.fit_offsets(data=data, k0=corr, band=band)
        mets[name] = arm_metric_arrays(data=data, correction=corr, offsets=offsets, band=band)
    every_arm = np.all([np.isfinite(m["qs"]) for m in mets.values()], axis=0)
    for name, met in mets.items():
        for gname, gmask in groups.items():
            m = gmask & every_arm
            unit = shift_label if gname.startswith("after") else year_month
            for w in (True, False):
                m2 = m & (winter == w)
                if not m2.any():
                    continue
                codes, inv = np.unique(unit[m2], return_inverse=True)
                cols = {"n": np.bincount(inv).astype(float)} | {
                    k: np.bincount(inv, weights=met[k][m2]) for k in METRICS
                }
                rows.append(
                    pl.DataFrame(
                        {
                            "series": series,
                            "arm": name,
                            "group": gname,
                            "winter": w,
                            "unit": codes.astype(np.int64),
                            **cols,
                        }
                    )
                )
    print("series", series, "done", flush=True)
    return pl.concat(rows)


def main() -> None:
    """Score every series in four spawned workers and write the unit table."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        tables = list(pool.imap_unordered(series_units, h.SERIES))
    table = pl.concat(tables)
    out = WORK / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "real_units.parquet")
    print(table.height, "unit rows")


if __name__ == "__main__":
    main()
