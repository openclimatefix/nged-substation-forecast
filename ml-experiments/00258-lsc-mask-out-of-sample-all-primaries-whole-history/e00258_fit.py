"""Compute every arm's correction at every (origin, lead) pair for one series, and store it.

Arms: K0 (no mask), GAM_ONLY (the GAM forecast with no filter), ORACLE (the 66 labelled load
shifts as logged), LR_b20 and LR_b80 for each fold that holds the series' Primary pair out
(experiment 195's re-ranker), RO_b80 (RO-NC), and HALF (the average of the K0 and LR_b80
corrections).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00258_fit.py 10 3 7
"""

from __future__ import annotations

import sys
import time

import e00258_data as dat
import e00258_harness as h
import numpy as np
import polars as pl


def main(series: int) -> None:
    """Fit one series and write `series_<id>.npz` and `spans_<id>.parquet` to the harness folder."""
    started = time.time()
    band = h.band_grid()
    data = h.load_series(series=series)
    events = h.labelled_events()
    origins, _, _ = h.origin_grid()
    k0_trace = h.k0_trace(data=data, band=band)
    arms: dict[str, np.ndarray] = {"K0": h.k0_correction(trace=k0_trace, band=band)}
    arms["GAM_ONLY"] = np.zeros_like(arms["K0"])
    span_rows = []
    sources: dict[str, list[h.Span]] = {"ORACLE": h.oracle_spans(events=events, series=series)}
    folds = dat.series_folds()[series]
    lr20, lr80, ro80 = (dat.alarm_tables(source=s) for s in ("LR_b20", "LR_b80", "RO_b80"))
    for fold in folds:
        sources[f"LR_b20_f{fold}"] = h.spans_from_table(table=lr20[fold], series=series)
        sources[f"LR_b80_f{fold}"] = h.spans_from_table(table=lr80[fold], series=series)
    sources["RO_b80"] = h.spans_from_table(table=ro80[0], series=series)
    for name, spans in sources.items():
        arms[name] = h.mask_correction(data=data, spans=spans, band=band, k0=k0_trace)
        span_rows += [
            {"arm": name, "onset": s.onset, "end": s.end, "start": s.start, "known": s.known}
            for s in spans
        ]
        print(
            f"series {series} {name}: {len(spans)} spans at {time.time() - started:.0f} s",
            flush=True,
        )
    for fold in folds:
        arms[f"HALF_f{fold}"] = 0.5 * (arms["K0"] + arms[f"LR_b80_f{fold}"])
    offsets = h.fit_offsets(data=data, k0=arms["K0"], band=band)
    h.HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = h.HARNESS_DIR / f"series_{series}.tmp.npz"
    np.savez(tmp, origins=origins, offsets=offsets, folds=np.array(folds), **arms)
    tmp.rename(h.HARNESS_DIR / f"series_{series}.npz")
    pl.DataFrame(
        span_rows,
        schema={
            "arm": pl.String,
            "onset": pl.Int64,
            "end": pl.Int64,
            "start": pl.Int64,
            "known": pl.Int64,
        },
    ).write_parquet(h.HARNESS_DIR / f"spans_{series}.parquet")
    print(f"series {series} done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(int(arg))
