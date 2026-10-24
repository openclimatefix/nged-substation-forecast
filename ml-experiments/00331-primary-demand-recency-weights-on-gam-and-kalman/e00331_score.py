# ruff: noqa: E501, RUF059, PTH119, PTH123, SIM115, D208
r"""Stage 2: Kalman level, quantile offsets and scores of every arm, ridge and gain.

    PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-*:ml-experiments/00331-* OMP_NUM_THREADS=1 \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00331-*/e00331_score.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import re
import sys
from datetime import UTC, datetime

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00331_common as c
import numpy as np
import polars as pl
from studies.dp_ens_weather import replace_site
from studies.dp_features import naive_utc

NO_FILTER = "--no-filter" in sys.argv
GAINS = (0.0,) if NO_FILTER else c.GAINS
OUT_DIR = c.WORK_DIR / "scores_nofilter" if NO_FILTER else c.SCORE_DIR
MIN_BAND_ROWS = 50
MIN_PAST_ROWS = 200


def levels_by_day(*, daily: np.ndarray, gain: float) -> np.ndarray:
    """Return the Kalman level used for each day: it sees only earlier days' mean residuals."""
    out = np.empty(len(daily))
    level = 0.0
    for i, value in enumerate(daily):
        out[i] = level
        level += gain * (value - level)
    return out


def band_table(*, resid: np.ndarray, hour_band: np.ndarray) -> np.ndarray:
    """Return the 11 residual quantiles for each of the four six-hour bands, shape (4, 11)."""
    out = np.zeros((4, len(c.LEVELS)))
    for b in range(4):
        v = resid[hour_band == b]
        out[b] = (
            np.quantile(v, c.LEVELS) if len(v) >= MIN_BAND_ROWS else np.quantile(resid, c.LEVELS)
        )
    return out


def score_stream(
    *, y_site: np.ndarray, end_times: np.ndarray, streams: dict, world: str, site: str,
    onset: datetime | None, score_cuts: list[datetime], ridges: tuple[float, ...], arms: tuple[str, ...],
) -> list[dict]:  # fmt: skip
    """Score every `arm|ridge` at every gain over `score_cuts`."""
    cuts = sorted(k for k in streams if isinstance(k, datetime))
    idx_all = np.concatenate([streams[k]["idx"] for k in cuts])
    t_all = end_times[idx_all]
    y_all = y_site[idx_all]
    start = t_all - np.timedelta64(1, "s")
    day_all = start.astype("datetime64[D]")
    band_all = (((start - day_all.astype("datetime64[us]")) // np.timedelta64(1, "h")) // 6).astype(
        int
    )
    days, inv = np.unique(day_all, return_inverse=True)
    onset_day = None if onset is None else np.datetime64(onset.replace(tzinfo=None), "D")
    out: list[dict] = []
    for arm in arms:
        for ridge in ridges:
            key = f"{arm}|{ridge}"
            missing = [k for k in cuts if key not in streams[k]]
            if missing and onset is None:
                msg = f"{site} {key} has no forecast at {len(missing)} cuts"
                raise ValueError(msg)
            p_all = np.concatenate(
                [
                    streams[k][key]
                    if key in streams[k]
                    else np.full(len(streams[k]["idx"]), np.nan)
                    for k in cuts
                ]
            )
            resid0 = y_all - p_all
            ok = np.isfinite(resid0)
            daily = np.bincount(inv, weights=np.where(ok, resid0, 0.0)) / np.maximum(
                np.bincount(inv, weights=ok.astype(float)), 1
            )
            for gain in GAINS:
                level = levels_by_day(daily=daily, gain=gain)[inv]
                filtered = resid0 - level
                for cut in score_cuts:
                    if cut not in streams or key not in streams[cut]:
                        continue
                    cut_day = np.datetime64(cut.replace(tzinfo=None), "D")
                    past = (
                        (day_all < cut_day)
                        & (day_all >= cut_day - np.timedelta64(c.BAND_DAYS, "D"))
                        & np.isfinite(filtered)
                    )
                    if past.sum() < MIN_PAST_ROWS:
                        continue
                    table = band_table(resid=filtered[past], hour_band=band_all[past])
                    now = np.isin(idx_all, streams[cut]["idx"])
                    q = (p_all[now] + level[now])[:, None] + table[band_all[now]]
                    y = y_all[now]
                    e = y[:, None] - q
                    pinball = np.maximum(c.LEVELS * e, (c.LEVELS - 1) * e)
                    if onset_day is None:
                        group = np.full(len(y), "all")
                    else:
                        since = (day_all[now] - onset_day) / np.timedelta64(1, "D")
                        group = np.select(
                            [since < 0, since < 30, since < 90], ["pre", "d0_30", "d30_90"], "d90p"
                        )
                    for g in np.unique(group):
                        m = group == g
                        out.append({
                            "world": world, "site": site, "month": str(cut.date()), "arm": arm, "ridge": ridge,
                            "gain": gain, "group": str(g), "n": int(m.sum()),
                            "pinball_sum": float(pinball[m].mean(axis=1).sum()), "abs_sum": float(np.abs(e[m][:, 5]).sum()),
                            "p01_sum": float(pinball[m][:, 0].sum()), "p95_sum": float(pinball[m][:, 9].sum()),
                            "p99_sum": float(pinball[m][:, 10].sum()), "below01": int((y[m] < q[m][:, 0]).sum()),
                            "above99": int((y[m] > q[m][:, 10]).sum()),
                        })  # fmt: skip
    return out


def job(path: str) -> str:
    """Score one stream pickle and write a parquet beside the scores folder."""
    name = os.path.basename(path).removesuffix(".pkl")
    target = OUT_DIR / f"{name}.parquet"
    if target.exists():
        return f"{name}: exists"
    streams = pickle.loads(open(path, "rb").read())
    cache = c295.load_cache()
    m = re.fullmatch(r"inj_(\w+?)_(\d\.\d\d)_(\d{4})_(\d\d)", name)
    if m is None:
        site = name.removeprefix("real_")
        world, onset, y_site = "real", None, cache.sites[site].y
        score_cuts = [k for k in c.monthly_cuts() if k >= c.SCORE_FROM]
        ridges, arms = c.RIDGES, c.REAL_ARMS
    else:
        site, size, year, month = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
        onset = datetime(year, month, 1, tzinfo=UTC)
        world = f"step_{size:.2f}_{year}-{month:02d}"
        before = (cache.end_times <= naive_utc(moment=onset)) & (
            cache.end_times > naive_utc(moment=onset) - np.timedelta64(365, "D")
        )
        y0 = cache.sites[site].y
        delta = np.where(
            cache.end_times > naive_utc(moment=onset), size * float(np.nanmean(y0[before])), 0.0
        )
        y_site = replace_site(cache=cache, site=site, y=y0 + delta).sites[site].y
        score_cuts = [
            k
            for k in c.monthly_cuts()
            if onset <= k <= c.add_months(moment=onset, months=c.MONTHS_AFTER)
        ]
        ridges, arms = (0.1,), c.INJECT_ARMS
    rows = score_stream(
        y_site=y_site,
        end_times=cache.end_times,
        streams=streams,
        world=world,
        site=site,
        onset=onset,
        score_cuts=score_cuts,
        ridges=ridges,
        arms=arms,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    tmp.rename(target)
    return f"{name}: {len(rows)} rows"


def main() -> None:
    """Score every stream pickle with a spawn pool."""
    paths = sorted(str(p) for p in c.STREAM_DIR.glob("*.pkl"))
    with mp.get_context("spawn").Pool(6) as pool:
        for line in pool.imap_unordered(job, paths, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
