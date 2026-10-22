"""Look first: what an engine start or stop looks like at half-hourly resolution.

Compares G2's recorded edges with the edges of the 35 labelled real switching transfers and with
large unlabelled steps in D8's clean history.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00199-*/e00199_shape.py
"""

import json
from datetime import UTC
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00199_common import CLEAN_END, G2_FIRST, G2_LAST, g2_output, key_of, load_panel, stamp
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PANEL_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "panel.parquet"
LABELLED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
MIN_STEP_MW: Final[float] = 1.0


def edge_stats(*, x: np.ndarray, centre: int) -> dict | None:
    """Edge statistics around the largest one-half-hour change within two of `centre`."""
    lo, hi = centre - 2, centre + 3
    if lo < 6 or hi + 6 >= len(x):
        return None
    d = np.diff(x)
    window = d[lo:hi]
    if not np.isfinite(window).all():
        return None
    j = lo + int(np.argmax(np.abs(window)))
    pre = np.nanmean(x[j - 5 : j - 1])
    post = np.nanmean(x[j + 3 : j + 7])
    step = post - pre
    if not np.isfinite(step) or abs(step) < MIN_STEP_MW:
        return None
    local = d[j - 1 : j + 2]
    biggest = d[j]
    return {
        "step_mw": float(step),
        "ramp_share": float(1.0 - abs(biggest) / max(abs(local.sum()), 1e-9)),
        "biggest_share": float(biggest / step),
        "intermediate": bool(abs(biggest) < 0.8 * abs(step)),
    }


def g2_edges_in_noise() -> list[dict]:
    """The same edges after G2's output is injected into D8's clean-history reading (K2 windows)."""
    from e00199_inject import build

    panel = load_panel()
    p = panel[key_of(panel=panel, label="D8")]
    y = np.asarray(p["power"], dtype=float)
    exp = np.asarray(p["expected"], dtype=float)
    g2 = g2_output(n=len(y))
    inj = build(y=y, g2=g2, with_k1=False)
    resid = inj.y - exp
    record = np.clip(np.nan_to_num(g2[stamp(G2_FIRST) : stamp(G2_LAST)]), 0, None)
    out = []
    for w in range(1, 10):
        idx = np.flatnonzero(inj.window_id == w)
        a = idx[0]
        i = 6
        while i < len(record) - 8:
            if abs(record[i + 1] - record[i]) > 0.3:
                s = edge_stats(x=resid, centre=a + i)
                if s:
                    out.append({**s, "kind": "engine G2 injected into D8 noise"})
                    i += 4
                    continue
            i += 1
    return out


def g2_edges() -> list[dict]:
    """Edges of G2's recorded output (steps of at least one unit)."""
    g2 = np.clip(
        np.nan_to_num(g2_output(n=len(next(iter(load_panel().values()))["power"]))), 0, None
    )
    lo, hi = stamp(G2_FIRST), stamp(G2_LAST)
    out = []
    i = lo + 6
    while i < hi - 8:
        if abs(g2[i + 1] - g2[i]) > 0.3:
            s = edge_stats(x=g2, centre=i)
            if s:
                out.append({**s, "kind": "engine G2"})
                i += 4
                continue
        i += 1
    return out


def transfer_edges() -> list[dict]:
    """Onset and end edges of the 35 real transfers over a day, on the largest-step series."""
    panel = pl.read_parquet(PANEL_PATH)
    series = {int(k[0]): v for k, v in panel.partition_by("series", as_dict=True).items()}
    arr = {}
    for sid, v in series.items():
        x = np.full(panel["stamp"].len() + 1, np.nan)
        x[v["stamp"].to_numpy()] = v["res_mw"].to_numpy()
        arr[sid] = x
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    from datetime import datetime

    g0 = datetime(2022, 1, 1, tzinfo=UTC)
    out = []
    for row in frame.iter_rows(named=True):
        hours = (row["logged_open_utc"] - row["logged_close_utc"]).total_seconds() / 3600
        if hours <= 24:
            continue
        ids = {int(r["series"]) for r in json.loads(row["receivers"])}
        ids |= {int(s) for s in (row["donor"], row["side_a"], row["side_b"]) if s is not None}
        for label, when in (("onset", row["onset_utc"]), ("end", row["end_utc"])):
            centre = int((when - g0).total_seconds() // 1800)
            best = None
            for sid in ids & set(arr):
                s = edge_stats(x=arr[sid], centre=centre)
                if s and (best is None or abs(s["step_mw"]) > abs(best["step_mw"])):
                    best = s
            if best:
                out.append({**best, "kind": f"real transfer {label}"})
    return out


def unlabelled_d8_steps() -> list[dict]:
    """Large steps in D8's clean history (reading, 2022-01 to 2023-06) with no label."""
    panel = load_panel()
    p = panel[key_of(panel=panel, label="D8")]
    x = np.asarray(p["power"], dtype=float) - np.asarray(p["expected"], dtype=float)
    out = []
    i = 8
    end = stamp(CLEAN_END) - 8
    while i < end:
        if abs(x[i + 1] - x[i]) > 1.0:
            s = edge_stats(x=x, centre=i)
            if s:
                out.append({**s, "kind": "unlabelled D8 clean-history step"})
                i += 6
                continue
        i += 1
    return out


def main() -> None:
    """Print the edge statistics by kind."""
    rows = [*g2_edges(), *g2_edges_in_noise(), *transfer_edges(), *unlabelled_d8_steps()]
    frame = pl.DataFrame(rows)
    print(
        frame.group_by("kind")
        .agg(
            n=pl.len(),
            mean_abs_step_mw=pl.col("step_mw").abs().mean(),
            share_with_intermediate_half_hour=pl.col("intermediate").mean(),
            median_ramp_share=pl.col("ramp_share").median(),
            mean_biggest_share=pl.col("biggest_share").abs().mean(),
        )
        .sort("kind")
    )
    frame.write_parquet(Path("/tmp/x/shape_edges.parquet"))


if __name__ == "__main__":
    main()
