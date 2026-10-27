"""Step 5 of experiment 00356: build the quantiles of default 1, default 2, and arms u1 to u4.

Reads the case table (`frame.parquet`), default 1 (`d1.npy`, 3 seeds), default 2's heads
(`parts/d2_h<seed>.npy`, 5 seeds), and the monthly heads (`parts/u3_s<seed>_<month>.npy`). Writes
`arms/<name>.npy` (5 seeds, scored rows, 11) for the scored rows (the 53 Tuesday runs of
2025-07 to 2026-06 and the Tuesday runs of 2026-07 to 2026-10, in table order), `arms/rows.parquet`,
and `arms/leakage.json`. Default 1 seed `s % 3` pairs with default 2 seed `s`.

- `d1`, `d2`: the two defaults. `u1`: experiment 348's z5 (the plant's residual band blended per
  quantile and lead band with default 1; weights from earlier scored months). `u2`: default 2
  blended with default 1 per quantile and lead band, weight in steps of 0.1 chosen by pinball loss
  on the finished daily runs since 2025-04-16 and refitted on the first day of each month; `u2p`
  is its placebo (each quantile and lead band takes another month's weight). `u4`: default 2 plus
  an offset per farm, lead day, and quantile, refitted monthly on the same window. `u3`: default 2's
  head retrained every month.

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_arms.py
"""

import json
import os
from datetime import UTC, datetime

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00348_z5 as z5  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

GRID = np.round(np.linspace(0.0, 1.0, 11), 1)
HALF_HOUR_US = 1_800_000_000
ARMS_DIR = common.DATA_DIR / "arms"


def month_labels() -> list[str]:
    """Return every scored month, `YYYY-MM`."""
    out, (y, m) = [], common.FIRST_SCORED_MONTH
    while (y, m) <= common.LAST_SCORED_MONTH:
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def start_of(*, label: str) -> datetime:
    """Return 00:00 UTC on the first day of month `label`."""
    return datetime(int(label[:4]), int(label[5:7]), 1, tzinfo=UTC)


def to_us(*, moment: datetime) -> int:
    """Return `moment` in microseconds since the epoch."""
    return int(moment.timestamp()) * 1_000_000


def finish(*, q: np.ndarray) -> np.ndarray:
    """Return quantiles clipped to the unit interval and sorted."""
    return np.sort(np.clip(q, 0.0, 1.0), axis=-1).astype(np.float32)


def blend_weights(
    *, d2: np.ndarray, d1: np.ndarray, frame: pl.DataFrame, labels: list[str], leakage: list[dict]
) -> np.ndarray:
    """Return the blend weights (months, 4 lead bands, 11) of default 2 against default 1."""
    init = frame["power_fcst_init_time"]
    base = ((init >= common.WINDOW_START) & frame["kept"]).to_numpy()
    valid_us = frame["valid_time"].dt.epoch("us").to_numpy()
    first = valid_us[base].min()
    bucket = ((valid_us[base] - first) // HALF_HOUR_US).astype(np.int64)
    n_buckets = int(bucket.max()) + 1
    band = z5.lead_band(hours=frame["lead_hours"].to_numpy().astype(np.float64))[base]
    key = band * n_buckets + bucket
    actual = frame["actual_norm"].to_numpy().astype(np.float64)[base]
    levels = c303.LEVELS
    loss = np.zeros((len(GRID), 11, 4, n_buckets))
    for gi, g in enumerate(GRID):
        q = g * d2[base].astype(np.float64) + (1.0 - g) * d1[base].astype(np.float64)
        diff = actual[:, None] - q
        pin = np.maximum(levels * diff, (levels - 1.0) * diff)
        for k in range(11):
            loss[gi, k] = np.bincount(key, weights=pin[:, k], minlength=4 * n_buckets).reshape(
                4, n_buckets
            )
    counts = np.bincount(key, minlength=4 * n_buckets).reshape(4, n_buckets)
    cum_loss, cum_count = loss.cumsum(axis=-1), counts.cumsum(axis=-1)
    weights = np.full((len(labels), 4, 11), 0.5)
    valid_base = valid_us[base]
    for mi, label in enumerate(labels):
        cutoff_us = to_us(moment=start_of(label=label) - common.DELAY)
        b = int((cutoff_us - first) // HALF_HOUR_US)
        b = min(b, n_buckets - 1)
        used = valid_base[bucket <= b]
        latest = int(used.max()) if used.size else None
        if latest is not None and latest > cutoff_us:
            msg = f"leak: u2 {label} uses valid time {latest}"
            raise ValueError(msg)
        leakage.append(
            {
                "arm": "u2",
                "month": label,
                "rows": int(used.size),
                "latest_valid_used": None
                if latest is None
                else datetime.fromtimestamp(latest / 1e6, tz=UTC).isoformat(),
                "month_start": start_of(label=label).isoformat(),
            }
        )
        for band_i in range(4):
            if b < 0 or cum_count[band_i, b] < common.MIN_WINDOW_ROWS:
                continue
            weights[mi, band_i] = GRID[np.argmin(cum_loss[:, :, band_i, b], axis=0)]
    return weights


def apply_weights(
    *, d2: np.ndarray, d1: np.ndarray, rows: pl.DataFrame, labels: list[str], weights: np.ndarray
) -> np.ndarray:
    """Return default 2 blended with default 1 by each row's month and lead band."""
    month = np.array([labels.index(m) for m in rows["month"].to_list()])
    band = z5.lead_band(hours=rows["lead_hours"].to_numpy().astype(np.float64))
    w = weights[month, band]
    return finish(q=w * d2.astype(np.float64) + (1.0 - w) * d1.astype(np.float64))


def plant_band(*, frame: pl.DataFrame, summary: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Return experiment 348's z5b for the scored rows: the plant's median plus a residual band.

    Per farm, the offset of each quantile is the residual quantile in each of 6 bins of the plant's
    mean and 4 lead bands of the hindcast runs (init before 2025-06-17), as in `e00348_z5`.
    """
    levels = c303.LEVELS
    hind = (frame["power_fcst_init_time"] < common.FIRST_NEW_RUN).to_numpy()
    train_all = hind & frame["kept"].to_numpy()
    actual_h = frame["actual_norm"].to_numpy().astype(np.float64)
    lead_all = z5.lead_band(hours=frame["lead_hours"].to_numpy().astype(np.float64))
    site = frame["site"].to_numpy()
    out = np.zeros((len(t_idx), 11))
    for farm in common.FARMS:
        rows_h = train_all & (site == farm)
        edges = np.quantile(summary[rows_h, 1], np.linspace(0, 1, z5.MEAN_BINS + 1)[1:-1])
        ids_all = z5.bin_ids(values=summary[:, 1], edges=edges) * 4 + lead_all
        rows_s = site[t_idx] == farm
        ids_s = ids_all[t_idx]
        diff = actual_h - summary[:, 0]
        for k, tau in enumerate(levels):
            fallback = np.quantile(diff[rows_h], tau)
            offset = np.full(z5.MEAN_BINS * 4 + 1, fallback)
            for b in np.unique(ids_all[rows_h]):
                sel = rows_h & (ids_all == b)
                if sel.sum() >= z5.MIN_BIN_ROWS:
                    offset[b] = np.quantile(diff[sel], tau)
            out[rows_s, k] = summary[t_idx[rows_s], 0] + offset[ids_s[rows_s]]
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def offset_layer(
    *,
    d2: np.ndarray,
    frame: pl.DataFrame,
    rows: pl.DataFrame,
    t_idx: np.ndarray,
    labels: list[str],
    leakage: list[dict],
) -> np.ndarray:
    """Return arm u4: default 2 plus a monthly refitted offset per farm, lead day, and quantile."""
    base = ((frame["power_fcst_init_time"] >= common.WINDOW_START) & frame["kept"]).to_numpy()
    base_idx = np.flatnonzero(base)
    base_frame = frame[base_idx]
    base_valid = base_frame["valid_time"]
    out = np.zeros((len(t_idx), 11), dtype=np.float32)
    months = rows["month"].to_numpy()
    for label in labels:
        cutoff = start_of(label=label) - common.DELAY
        sel = (base_valid <= cutoff).to_numpy()
        latest = base_valid.filter(pl.Series(sel)).max()
        if latest is not None and not isinstance(latest, datetime):
            raise TypeError(latest)
        if latest is not None and latest > cutoff:
            msg = f"leak: u4 {label} uses valid time {latest}"
            raise ValueError(msg)
        leakage.append(
            {
                "arm": "u4",
                "month": label,
                "rows": int(sel.sum()),
                "latest_valid_used": None if latest is None else latest.isoformat(),
                "month_start": start_of(label=label).isoformat(),
            }
        )
        table, keys = ens_final.offsets_table(
            hindcast=base_frame.filter(pl.Series(sel)),
            quantiles=d2[base_idx[sel]].astype(np.float64),
        )
        target = months == label
        out[target] = ens_final.apply_offsets(
            cases=rows.filter(pl.Series(target)),
            quantiles=d2[t_idx[target]],
            table=table,
            keys=keys,
        )
    return out


def main() -> None:
    """Build every arm and store it."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(common.DATA_DIR / "frame.parquet")
    summary = np.load(common.DATA_DIR / "summary.npy").astype(np.float64)
    d1_all = np.load(common.DATA_DIR / "d1.npy")
    d2_all = np.stack([np.load(common.PARTS_DIR / f"d2_h{s}.npy") for s in common.SEEDS])
    labels = month_labels()
    t_mask = (
        frame["scored"]
        | (frame["tuesday"] & (frame["power_fcst_init_time"] >= common.OUT_OF_TIME_FIRST_RUN))
    ).to_numpy()
    t_idx = np.flatnonzero(t_mask)
    rows = frame[t_idx].with_columns(masked=~pl.col("kept"))
    rows.write_parquet(ARMS_DIR / "rows.parquet")
    print(
        f"{len(t_idx):,} scored rows; {int(frame['scored'].sum()):,} in the scored year", flush=True
    )
    leakage: list[dict] = []
    pos = np.full(frame.height, -1)
    pos[t_idx] = np.arange(len(t_idx))
    results: dict[str, list[np.ndarray]] = {
        n: [] for n in ("d1", "d2", "u1", "u2", "u2p", "u3", "u4")
    }
    plant = plant_band(frame=frame, summary=summary, t_idx=t_idx)
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    for s in common.SEEDS:
        d1, d2 = d1_all[s % 3], d2_all[s]
        print(f"seed {s}: blend weights", flush=True)
        weights = blend_weights(
            d2=d2, d1=d1, frame=frame, labels=labels, leakage=leakage if s == 0 else []
        )
        np.save(ARMS_DIR / f"u2_weights_s{s}.npy", weights)
        placebo = weights.copy()
        for mi in range(len(labels)):
            for band in range(4):
                for k in range(11):
                    others = [j for j in range(len(labels)) if j != mi]
                    placebo[mi, band, k] = weights[
                        others[int(rng.integers(0, len(others)))], band, k
                    ]
        results["d1"].append(d1[t_idx])
        results["d2"].append(d2[t_idx])
        results["u2"].append(
            apply_weights(d2=d2[t_idx], d1=d1[t_idx], rows=rows, labels=labels, weights=weights)
        )
        results["u2p"].append(
            apply_weights(d2=d2[t_idx], d1=d1[t_idx], rows=rows, labels=labels, weights=placebo)
        )
        blend, used = z5.blend(plant=plant, default=d1[t_idx].astype(np.float64), scored=rows)
        np.save(ARMS_DIR / f"u1_weights_s{s}.npy", used)
        results["u1"].append(blend)
        print(f"seed {s}: offset layer", flush=True)
        results["u4"].append(
            offset_layer(
                d2=d2,
                frame=frame,
                rows=rows,
                t_idx=t_idx,
                labels=labels,
                leakage=leakage if s == 0 else [],
            )
        )
        u3 = np.zeros((len(t_idx), 11), dtype=np.float32)
        for label in labels:
            part = np.load(common.PARTS_DIR / f"u3_s{s}_{label}.npy")
            target = np.flatnonzero(((frame["month"] == label) & frame["tuesday"]).to_numpy())
            keep_rows = pos[target] >= 0
            part = part[keep_rows]
            target = target[keep_rows]
            u3[pos[target]] = part
        results["u3"].append(u3)
    for name, values in results.items():
        np.save(ARMS_DIR / f"{name}.npy", np.stack(values).astype(np.float32))
    (ARMS_DIR / "leakage.json").write_text(json.dumps(leakage, indent=1))
    print("arms stored", flush=True)


if __name__ == "__main__":
    main()
