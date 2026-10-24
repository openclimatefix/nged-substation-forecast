# ruff: noqa: E501, ANN001, ANN201, C416, RUF059
r"""Stage 2: forward-chained forecasts of the arms on real records and on injected steps.

Pass 1 computes, for each monthly origin, the pre-filter forecast of each base: A (all history),
B (last 24 months), C1 (old effects frozen at the detected change, new effects fitted on the
later residual), C2 (joint backfitting) and C2o (C2 with the true change date, an oracle). Pass 2
adds a fixed-gain local-level Kalman filter (the arms D1, D2, D2o reset it at the change), builds
the quantile bands from the arm's own earlier residuals, and scores by days since the change.

    PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-* OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00328-*/e00328_model.py --stage real|inject
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from datetime import UTC, datetime

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00328_common as c
import e00328_detect as d328
import e00328_detectors as det
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam

LEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
GAIN = 0.10
KAPPA = 10.0
MIN_REGIME_DAYS = 30
BAND_DAYS = 365
REGIME_BAND_DAYS = 60
SCORE_FROM = datetime(2021, 9, 1, tzinfo=UTC)
MONTHS_AFTER = 10
PRE_REGISTERED = ("A", "B", "C1", "C2", "C2o", "D1", "D2", "D2o")
POST_HOC = ("An", "Bn", "C1n", "C2n", "C2on", "Ar", "C1t", "C2t", "C2ot", "D1t", "D2t", "D2ot")
"""Post hoc arms: `Ar` is A with the regime-band rule, and `t` arms are the C and D arms with the
trailing 365-day bands of arm A. They separate the band rule from the regime effects."""
ARMS = (*PRE_REGISTERED, *POST_HOC)
BASE_OF = {
    "A": "A", "B": "B", "C1": "C1", "C2": "C2", "C2o": "C2o", "D1": "C1", "D2": "C2", "D2o": "C2o",
    "An": "A", "Bn": "B", "C1n": "C1", "C2n": "C2", "C2on": "C2o",
    "Ar": "A", "C1t": "C1", "C2t": "C2", "C2ot": "C2o", "D1t": "C1", "D2t": "C2", "D2ot": "C2o",
}  # fmt: skip
RESETS = {"D1", "D2", "D2o", "D1t", "D2t", "D2ot"}
ORACLE = {"C2o", "D2o", "C2ot", "D2ot", "C2on"}
NO_FILTER = {"An", "Bn", "C1n", "C2n", "C2on"}
REGIME_BANDS = {"C1", "C2", "C2o", "D1", "D2", "D2o", "Ar"}
MODEL_DIR = c.WORK_DIR / "model"
SPEC = GamSpec(ridge=c295.RIDGE, drift="none", capacity="fitted", final_knot="free")


def cell_of(*, cache: FeatureCache, idx: np.ndarray) -> np.ndarray:
    """Return the half-hour of day (0 to 47) plus 48 for a weekend, for grid positions `idx`."""
    start = cache.end_times[idx] - np.timedelta64(1, "s")
    day = start.astype("datetime64[D]")
    half = ((start - day.astype("datetime64[us]")) // np.timedelta64(30, "m")).astype(int)
    weekend = ((day.astype(np.int64) + 3) % 7 >= 5).astype(int)
    return half + 48 * weekend


def effect_table(*, cell: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Return the shrunken mean residual of each of the 96 cells (zero where there is no data)."""
    n = np.bincount(cell, minlength=96).astype(float)
    s = np.bincount(cell, weights=resid, minlength=96)
    return s / (n + KAPPA)


def fit_at(*, cache: FeatureCache, site: str, cut: datetime, months: int | None):
    """Fit the lagless GAM before `cut` on all history (`months` None) or a rolling window."""
    return fit_gam(
        cache=cache, site=site, specs=[SPEC], cut=cut,
        window=TrainingWindow(months=months), season_years=c295.SEASON_YEARS,
    )  # fmt: skip


def month_rows(*, cache: FeatureCache, rows: np.ndarray, cut: datetime) -> np.ndarray:
    """Return the grid rows of the month after `cut`."""
    start, stop = naive_utc(moment=cut), naive_utc(moment=c.add_months(moment=cut, months=1))
    return rows[(cache.end_times[rows] > start) & (cache.end_times[rows] <= stop)]


def known_regime(*, alarms: list[det.Alarm], cut: datetime) -> np.datetime64 | None:
    """Return the change day of the latest alarm raised at least 30 days before `cut`."""
    cut_day = np.datetime64(cut.replace(tzinfo=None), "D")
    old = [a for a in alarms if a.alarm_day < cut_day]
    if not old:
        return None
    start = old[-1].change_day
    return start if (cut_day - start) / np.timedelta64(1, "D") >= MIN_REGIME_DAYS else None


def arm_c1(*, cache, site, cut, rows, target, start, fits):
    """Return C1's forecast: old effects frozen at `start`, new effects on the later residual."""
    r_cut = datetime(*[int(x) for x in str(start).split("-")], tzinfo=UTC)
    if str(start) not in fits:
        fits[str(start)] = fit_at(cache=cache, site=site, cut=r_cut, months=None)
    fit = fits[str(start)]
    regime = rows[
        (cache.end_times[rows] > naive_utc(moment=r_cut))
        & (cache.end_times[rows] <= naive_utc(moment=cut))
    ]
    resid = cache.sites[site].y[regime] - predict_gam(cache=cache, fit=fit, index=regime)
    table = effect_table(cell=cell_of(cache=cache, idx=regime), resid=resid)
    return predict_gam(cache=cache, fit=fit, index=target) + table[cell_of(cache=cache, idx=target)]


def arm_c2(*, cache, site, cut, rows, target, start):
    """Return C2's forecast: the GAM and the regime effects fitted jointly by backfitting."""
    r_cut = datetime(*[int(x) for x in str(start).split("-")], tzinfo=UTC)
    regime = rows[
        (cache.end_times[rows] > naive_utc(moment=r_cut))
        & (cache.end_times[rows] <= naive_utc(moment=cut))
    ]
    y_full = cache.sites[site].y
    table = np.zeros(96)
    fit = None
    for _ in range(3):
        y_adj = y_full.copy()
        y_adj[regime] = y_full[regime] - table[cell_of(cache=cache, idx=regime)]
        adjusted = replace_site(cache=cache, site=site, y=y_adj)
        fit = fit_at(cache=adjusted, site=site, cut=cut, months=None)
        resid = y_full[regime] - predict_gam(cache=adjusted, fit=fit, index=regime)
        table = effect_table(cell=cell_of(cache=cache, idx=regime), resid=resid)
    return predict_gam(cache=cache, fit=fit, index=target) + table[cell_of(cache=cache, idx=target)]


def pass_one(
    *, cache: FeatureCache, site: str, months: list[datetime], truth: np.datetime64 | None,
    spliced: dict[datetime, dict], delta_start: datetime | None,
) -> dict[datetime, dict]:  # fmt: skip
    """Return `{cut: {"idx": rows, arm: forecast}}` for the given monthly origins."""
    rows = c295.all_rows(cache=cache, site=site)
    out: dict[datetime, dict] = {}
    # B first: its residual stream is the detector's, so the regime log comes from it.
    streams: dict[datetime, dict] = {}
    for cut in c.monthly_cuts():
        if cut < c.FIRST_CUT or cut > months[-1]:
            continue
        if delta_start is not None and cut < delta_start and cut in spliced:
            streams[cut] = spliced[cut]
            continue
        target = month_rows(cache=cache, rows=rows, cut=cut)
        if len(target) == 0:
            continue
        try:
            pa = predict_gam(
                cache=cache, fit=fit_at(cache=cache, site=site, cut=cut, months=None), index=target
            )
            pb = predict_gam(
                cache=cache,
                fit=fit_at(cache=cache, site=site, cut=cut, months=c.WINDOW_MONTHS),
                index=target,
            )
        except ValueError:
            continue
        streams[cut] = {"idx": target, "A": pa, "B": pb}
    times = np.concatenate([cache.end_times[s["idx"]] for s in streams.values()])
    resid = np.concatenate([cache.sites[site].y[s["idx"]] - s["B"] for s in streams.values()])
    days, means = det.daily_means(end_times=times, residual=resid)
    alarms = det.cusum_alarms(
        days=days,
        means=means,
        h=json.loads((c.WORK_DIR / "chosen_detectors.json").read_text())["rolling_gam"],
    )
    fits: dict[str, object] = {}
    for cut in months:
        if cut not in streams:
            continue
        s = dict(streams[cut])
        if delta_start is not None and cut < delta_start and "C1" in s:
            out[cut] = s
            continue
        target = s["idx"]
        start = known_regime(alarms=alarms, cut=cut)
        s["regime_start"] = None if start is None else str(start)
        s["C1"], s["C2"] = s["A"], s["A"]
        if start is not None:
            s["C1"] = arm_c1(
                cache=cache, site=site, cut=cut, rows=rows, target=target, start=start, fits=fits
            )
            s["C2"] = arm_c2(cache=cache, site=site, cut=cut, rows=rows, target=target, start=start)
        s["C2o"] = s["A"]
        if (
            truth is not None
            and (np.datetime64(cut.replace(tzinfo=None), "D") - truth) / np.timedelta64(1, "D")
            >= 14
        ):
            s["C2o"] = arm_c2(
                cache=cache, site=site, cut=cut, rows=rows, target=target, start=truth
            )
        out[cut] = s
    out["alarms"] = [(str(a.alarm_day), str(a.change_day)) for a in alarms]  # type: ignore[index]  # ty: ignore[invalid-assignment]
    return out


def filter_days(
    *, days: np.ndarray, daily: np.ndarray, until: np.datetime64, reset: np.datetime64 | None
):
    """Run the fixed-gain local-level filter; return `{day: state used for that day's forecast}`.

    The state before a day sees only earlier days' mean residuals. With a reset day the state
    returns to zero at its start and the gain is 1/n for the first n days, so the filter forgets
    the old regime; otherwise the gain is fixed.
    """
    level, n_since, used = 0.0, 0, {}
    for day, value in zip(days, daily, strict=True):
        if day > until:
            break
        if reset is not None and day == reset:
            level, n_since = 0.0, 0
        used[day] = level
        n_since += 1
        gain = max(GAIN, 1.0 / n_since) if reset is not None and day >= reset else GAIN
        level += gain * (value - level)
    return used


def band_table(*, resid: np.ndarray, hour_band: np.ndarray) -> np.ndarray:
    """Return the 11 residual quantiles for each of the four six-hour bands, shape (4, 11)."""
    out = np.zeros((4, len(LEVELS)))
    for b in range(4):
        v = resid[hour_band == b]
        out[b] = np.quantile(v, LEVELS) if len(v) >= 50 else np.quantile(resid, LEVELS)
    return out


def pass_two(
    *, cache: FeatureCache, site: str, streams: dict, truth: np.datetime64 | None,
    score_months: list[datetime], world: str,
) -> list[dict]:  # fmt: skip
    """Score every arm at every month in `score_months`."""
    cuts = sorted(k for k in streams if isinstance(k, datetime))
    idx_all = np.concatenate([streams[k]["idx"] for k in cuts])
    y_all = cache.sites[site].y[idx_all]
    t_all = cache.end_times[idx_all]
    day_all = c.day_of(end_times=t_all)
    start = t_all - np.timedelta64(1, "s")
    band_all = (
        ((start - start.astype("datetime64[D]").astype("datetime64[us]")) // np.timedelta64(1, "h"))
        // 6
    ).astype(int)
    out: list[dict] = []
    for arm in ARMS:
        p_all = np.concatenate([streams[k][BASE_OF[arm]] for k in cuts])
        resid0 = y_all - p_all
        days, inv = np.unique(day_all, return_inverse=True)
        daily = np.bincount(inv, weights=resid0) / np.bincount(inv)
        for cut in score_months:
            if cut not in streams:
                continue
            s = streams[cut]
            cut_day = np.datetime64(cut.replace(tzinfo=None), "D")
            reset = None
            if arm in RESETS:
                reset = (
                    truth
                    if arm in ORACLE
                    else (np.datetime64(s["regime_start"]) if s.get("regime_start") else None)
                )
                if arm in ORACLE and not (
                    truth is not None and (cut_day - truth) / np.timedelta64(1, "D") >= 14
                ):
                    reset = None
            month_end = np.datetime64(c.add_months(moment=cut, months=1).replace(tzinfo=None), "D")
            used = filter_days(days=days, daily=daily, until=month_end, reset=reset)
            level = np.array([used.get(d, 0.0) for d in day_all])
            if arm in NO_FILTER:
                level = np.zeros(len(day_all))
            filtered = resid0 - level
            past = (day_all < cut_day) & (day_all >= cut_day - np.timedelta64(BAND_DAYS, "D"))
            regime_start = (
                reset
                if reset is not None
                else (np.datetime64(s["regime_start"]) if s.get("regime_start") else None)
            )
            if arm in ORACLE and not (
                truth is not None and (cut_day - truth) / np.timedelta64(1, "D") >= 14
            ):
                regime_start = None
            use = past
            if (arm in REGIME_BANDS and regime_start is not None
                    and (cut_day - regime_start) / np.timedelta64(1, "D") >= REGIME_BAND_DAYS):  # fmt: skip
                regime = (day_all < cut_day) & (day_all >= regime_start)
                if regime.sum() >= 200:
                    use = regime
            if use.sum() < 200:
                continue
            table = band_table(resid=filtered[use], hour_band=band_all[use])
            now = np.isin(idx_all, s["idx"])
            q = (p_all[now] + level[now])[:, None] + table[band_all[now]]
            y = y_all[now]
            e = y[:, None] - q
            pinball = np.maximum(LEVELS * e, (LEVELS - 1) * e)
            day_now = day_all[now]
            since = (
                (day_now - truth) / np.timedelta64(1, "D")
                if truth is not None
                else np.full(len(y), np.nan)
            )
            group = np.where(
                np.isnan(since),
                "all",
                np.select(
                    [since < 0, since < 30, since < 90, since < 180],
                    ["pre", "d0_30", "d30_90", "d90_180"],
                    "d180p",
                ),
            )
            for g in np.unique(group):
                m = group == g
                out.append({
                    "world": world, "site": site, "month": str(cut.date()), "arm": arm, "group": str(g),
                    "n": int(m.sum()), "pinball_sum": float(pinball[m].mean(axis=1).sum()),
                    "abs_sum": float(np.abs(e[m][:, 5]).sum()), "p01_sum": float(pinball[m][:, 0].sum()),
                    "p99_sum": float(pinball[m][:, 10].sum()), "below01": int((y[m] < q[m][:, 0]).sum()),
                    "above99": int((y[m] > q[m][:, 10]).sum()),
                })  # fmt: skip
    return out


def job(args: tuple[str, str, int, int, float, str]) -> str:
    """Run one Primary's real record (`kind` real) or one injected world."""
    kind, site, year, month, size, parity = args
    torch.set_num_threads(1)
    cache = c295.load_cache()
    allm = [m for m in c.monthly_cuts()]
    if kind == "real":
        world, truth, delta_start = "real", (c.D8_BREAK if site == "D8" else None), None
        target = MODEL_DIR / f"real_{site}.parquet"
        score_months = [m for m in allm if m >= SCORE_FROM]
        months = allm
        spliced: dict = {}
        use_cache = cache
    else:
        world = f"step_{size:.2f}_{year}-{month:02d}"
        target = MODEL_DIR / f"inj_{site}_{size:.2f}_{year}_{month:02d}.parquet"
        onset = datetime(year, month, 1, tzinfo=UTC)
        base_pickle = MODEL_DIR / f"real_{site}.pkl"
        spliced = pickle.loads(base_pickle.read_bytes())
        before = (cache.end_times <= naive_utc(moment=onset)) & (
            cache.end_times > naive_utc(moment=onset) - np.timedelta64(365, "D")
        )
        y_site = cache.sites[site].y
        if np.isfinite(y_site[before]).sum() < 0.5 * 17520:
            return f"{site} {world}: too little history"
        level = float(np.nanmean(y_site[before]))
        delta = np.where(cache.end_times > naive_utc(moment=onset), size * level, 0.0)
        use_cache = replace_site(
            cache=cache, site=site, y=np.nan_to_num(y_site, nan=np.nan) + delta
        )
        truth = np.datetime64(onset.replace(tzinfo=None), "D")
        delta_start = onset
        months = [
            m
            for m in allm
            if m <= c.add_months(moment=onset, months=MONTHS_AFTER) and m <= c.LAST_CUT
        ]
        score_months = [m for m in months if m >= onset]
    if target.exists():
        return f"{site} {world}: exists"
    stream_file = MODEL_DIR / (
        f"real_{site}.pkl" if kind == "real" else f"inj_{site}_{size:.2f}_{year}_{month:02d}.pkl"
    )
    if stream_file.exists():
        streams = pickle.loads(stream_file.read_bytes())
        alarms = streams.pop("alarms", [])
    else:
        streams = pass_one(
            cache=use_cache,
            site=site,
            months=months,
            truth=truth,
            spliced=spliced,
            delta_start=delta_start,
        )
        alarms = streams.pop("alarms")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = stream_file.with_name(f"{stream_file.name}.{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps({**streams, "alarms": alarms}))
        tmp.rename(stream_file)
    rows = pass_two(
        cache=use_cache,
        site=site,
        streams=streams,
        truth=truth,
        score_months=score_months,
        world=world,
    )
    frame = pl.DataFrame(rows)
    d328._atomic(frame=frame, path=target)
    return f"{site} {world}: {frame.height} rows, {len(alarms)} alarms"


def main() -> None:
    """Run one stage in parallel with the spawn start method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("real", "inject"), required=True)
    parser.add_argument("--workers", type=int, default=c.n_workers())
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--sites", nargs="+", default=None)
    args = parser.parse_args()
    sites = args.sites or json.loads((c295.EXPERIMENT_DIR / "usable_sites.json").read_text())
    if args.stage == "real":
        jobs = [("real", s, 0, 0, 0.0, "") for s in sites]
    else:
        jobs = [
            ("inj", s, y, m, size, p)
            for s in sites
            if s != "D8"
            for (y, m, p) in d328.ONSETS
            if p == "report"
            for size in c.STEP_SIZES
        ]
    if args.count:
        jobs = jobs[: args.count]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(job, jobs, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
