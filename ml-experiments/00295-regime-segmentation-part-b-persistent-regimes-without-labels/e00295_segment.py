# ruff: noqa: PERF401, PLR0912, PLR0915, B007
r"""Segment every Primary's residual, inject known steps, and record how each is found and labelled.

systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00295_segment.py --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta

import e00295_common as c
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_ens_weather import replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import TrainingWindow

ODD_ONSETS = [
    (2021, 9),
    (2021, 11),
    (2022, 5),
    (2022, 9),
    (2024, 1),
    (2024, 7),
    (2025, 3),
    (2025, 9),
]
EVEN_ONSETS = [
    (2021, 10),
    (2021, 12),
    (2022, 6),
    (2022, 12),
    (2024, 2),
    (2024, 8),
    (2025, 4),
    (2025, 10),
]
SIZE = 0.25
STEP_SIZES = (0.10, 0.25)
TEMPORARY_DAYS = (14, 30, 60)
CONFIGS = [(b, m) for b in c.BETAS for m in c.MIN_LENGTHS]
MATCH_DAYS = 14
_STATE: dict = {}


def clean_state(*, cache: FeatureCache, site: str) -> dict:
    """Fit on the whole history and return the Primary's clean daily series and segmentations."""
    idx = c.all_rows(cache=cache, site=site)
    pred = c.fit_and_predict(
        cache=cache, site=site, cut=c.END_OF_DATA, window=TrainingWindow(), index=idx
    )
    residual = cache.sites[site].y[idx] - pred
    return series_state(index=idx, end_times=cache.end_times[idx], residual=residual) | {
        "index": idx,
        "residual": residual,
    }


def series_state(*, index: np.ndarray, end_times: np.ndarray, residual: np.ndarray) -> dict:
    """Return the daily series, noise scale, and unlabelled boundaries for every configuration."""
    series = c.daily_series(end_times=end_times, residual=residual)
    sigma = c.noise_scale(values=series.values)
    bounds = {
        config: c.boundaries_of(
            series=series,
            starts=c.segment(
                values=series.values, sigma=sigma, beta=config[0], min_length=config[1]
            ),
        )
        for config in CONFIGS
    }
    return {"series": series, "sigma": sigma, "bounds": bounds}


def _clean_job(site: str) -> tuple[str, dict]:
    torch.set_num_threads(1)
    try:
        return site, clean_state(cache=_STATE["cache"], site=site)
    except ValueError as error:  # a Primary with no usable training rows
        print(f"{site} skipped: {error}", flush=True)
        return site, {}


def neighbours_of(*, states: dict[str, dict]) -> dict[str, list[str]]:
    """Return each Primary's five most correlated others, by the daily mean residual."""
    days = sorted({d for s in states.values() for d in s["series"].days.tolist()})
    index = {d: i for i, d in enumerate(days)}
    matrix = np.full((len(states), len(days)), np.nan)
    names = sorted(states)
    for row, name in enumerate(names):
        s = states[name]["series"]
        matrix[row, [index[d] for d in s.days.tolist()]] = s.values[:, 0]
    out = {}
    for i, name in enumerate(names):
        corr = []
        for j, other in enumerate(names):
            ok = np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            corr.append(
                -9.0
                if i == j or ok.sum() < 200
                else np.corrcoef(matrix[i, ok], matrix[j, ok])[0, 1]
            )
        out[name] = [names[k] for k in np.argsort(corr)[::-1][:5]]
    return out


def delta_of(
    *, cache: FeatureCache, site: str, onset: datetime, kind: str, size: float, days: int | None
) -> np.ndarray:
    """Return the injected power change at the receiver on the whole grid, in points."""
    y = np.nan_to_num(cache.sites[site].y)
    t = cache.end_times
    active = t > naive_utc(moment=onset)
    if days is not None:
        active = active & (t <= naive_utc(moment=onset + timedelta(days=days)))
    if kind == "flat":
        before = (t <= naive_utc(moment=onset)) & (
            t > naive_utc(moment=onset - timedelta(days=365))
        )
        if not np.isfinite(cache.sites[site].y[before]).any():
            return np.full(len(t), np.nan)
        level = float(np.nanmean(cache.sites[site].y[before]))
        return np.where(active, size * level, 0.0)
    return np.where(active, size * y, 0.0)


def score_scenario(
    *,
    state: dict,
    boundaries_by_config: dict,
    neighbour_bounds: list[dict],
    onset: np.datetime64,
    expected: str,
    injected_mean: float,
    last_day: np.datetime64,
) -> list[dict]:
    """Label the receiver's boundaries per configuration and find the injection's boundary."""
    rows = []
    for config in CONFIGS:
        labelled = c.label_boundaries(
            boundaries=boundaries_by_config[config],
            neighbours=[nb[config] for nb in neighbour_bounds],
            sigma=state["sigma"],
            last_day=last_day,
        )
        near = [b for b in labelled if abs(c.days_between(onset, b.date)) <= MATCH_DAYS]
        if near:
            best = max(near, key=lambda b: float(np.abs(b.shift / state["sigma"]).sum()))
            rows.append(
                {
                    "beta": config[0],
                    "min_length": config[1],
                    "found": True,
                    "label": best.label,
                    "dating_error_days": c.days_between(onset, best.date),
                    "survive": float(best.centre_shift / injected_mean)
                    if injected_mean
                    else float("nan"),
                    "expected": expected,
                }
            )
        else:
            rows.append(
                {
                    "beta": config[0],
                    "min_length": config[1],
                    "found": False,
                    "label": "missed",
                    "dating_error_days": float("nan"),
                    "survive": float("nan"),
                    "expected": expected,
                }
            )
    return rows


def _site_job(args: tuple[str, int]) -> str:
    site, max_onsets = args
    torch.set_num_threads(1)
    cache: FeatureCache = _STATE["cache"]
    states, nbrs = _STATE["states"], _STATE["neighbours"]
    target = c.EXPERIMENT_DIR / "segmentation" / f"{site}.parquet"
    if target.exists():
        return f"{site}: already done"
    last_day = states[site]["series"].days[-1]
    donor = next(n for n in nbrs[site] if n != "D8")
    rows: list[dict] = []
    # False breaks on the unmodified series.
    for config in CONFIGS:
        labelled = c.label_boundaries(
            boundaries=states[site]["bounds"][config],
            neighbours=[states[n]["bounds"][config] for n in nbrs[site]],
            sigma=states[site]["sigma"],
            last_day=last_day,
        )
        for b in labelled:
            rows.append(
                {
                    "record": "clean_break",
                    "site": site,
                    "beta": config[0],
                    "min_length": config[1],
                    "date": str(b.date),
                    "label": b.label,
                    "shift_centre": b.centre_shift,
                    "shift_shape": float(b.shift[3]),
                    "shift_dip_rate": float(b.shift[0]),
                }
            )
    onsets = [(y, m, "odd") for y, m in ODD_ONSETS[:max_onsets]] + [
        (y, m, "even") for y, m in EVEN_ONSETS[:max_onsets]
    ]
    for year, month, parity in onsets:
        onset = datetime(year, month, 1, tzinfo=UTC)
        onset_day = np.datetime64(onset.date(), "D") + np.timedelta64(1, "D")
        year_before = (cache.end_times <= naive_utc(moment=onset)) & (
            cache.end_times > naive_utc(moment=onset - timedelta(days=365))
        )
        if np.isfinite(cache.sites[site].y[year_before]).sum() < 0.5 * 17520:
            continue  # less than half a year of readings before this onset
        scenarios = [
            ("persistent", k, s, None, "persistent") for s in STEP_SIZES for k in ("flat", "share")
        ]
        scenarios += [("transfer", "share", SIZE, None, "none")]
        scenarios += [("temporary", "share", SIZE, d, "temporary") for d in TEMPORARY_DAYS]
        for family, kind, size, days, expected in scenarios:
            delta = delta_of(cache=cache, site=site, onset=onset, kind=kind, size=size, days=days)
            if not np.isfinite(delta).all():
                continue  # the Primary has no readings in the year before this onset
            mirrored = family in ("temporary", "transfer")
            window = delta != 0
            injected_mean = float(delta[window][: 48 * 30].mean()) if window.any() else 0.0
            for absorption in ("fit", "residual"):
                if absorption == "fit":
                    injected_cache = replace_site(
                        cache=cache, site=site, y=cache.sites[site].y + delta
                    )
                    if mirrored:
                        injected_cache = replace_site(
                            cache=injected_cache, site=donor, y=cache.sites[donor].y - delta
                        )
                    idx = states[site]["index"]
                    pred = c.fit_and_predict(
                        cache=injected_cache,
                        site=site,
                        cut=c.END_OF_DATA,
                        window=TrainingWindow(),
                        index=idx,
                    )
                    residual = injected_cache.sites[site].y[idx] - pred
                    donor_state = None
                    if mirrored:
                        didx = states[donor]["index"]
                        dpred = c.fit_and_predict(
                            cache=injected_cache,
                            site=donor,
                            cut=c.END_OF_DATA,
                            window=TrainingWindow(),
                            index=didx,
                        )
                        donor_state = series_state(
                            index=didx,
                            end_times=cache.end_times[didx],
                            residual=injected_cache.sites[donor].y[didx] - dpred,
                        )
                else:
                    idx = states[site]["index"]
                    residual = states[site]["residual"] + delta[idx]
                    donor_state = None
                    if mirrored:
                        didx = states[donor]["index"]
                        donor_state = series_state(
                            index=didx,
                            end_times=cache.end_times[didx],
                            residual=states[donor]["residual"] - delta[didx],
                        )
                own = series_state(index=idx, end_times=cache.end_times[idx], residual=residual)
                neighbour_bounds = [
                    donor_state["bounds"]
                    if (n == donor and donor_state is not None)
                    else states[n]["bounds"]
                    for n in nbrs[site]
                ]
                scored = score_scenario(
                    state=own,
                    boundaries_by_config=own["bounds"],
                    neighbour_bounds=neighbour_bounds,
                    onset=onset_day,
                    expected=expected,
                    injected_mean=injected_mean,
                    last_day=last_day,
                )
                for r in scored:
                    rows.append(
                        {
                            "record": "injection",
                            "site": site,
                            "parity": parity,
                            "onset": str(onset.date()),
                            "family": family,
                            "shape": kind,
                            "size": size,
                            "duration_days": -1 if days is None else days,
                            "absorption": absorption,
                            **r,
                        }
                    )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{site}: {frame.height} rows"


def main() -> None:
    """Run the clean segmentations, then every injection, in parallel by Primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--max-onsets", type=int, default=8)
    args = parser.parse_args()
    (c.EXPERIMENT_DIR / "segmentation").mkdir(parents=True, exist_ok=True)
    _STATE["cache"] = cache = c.load_cache()
    context = multiprocessing.get_context("fork")
    state_file = c.EXPERIMENT_DIR / "clean_states.pkl"
    if state_file.exists():
        _STATE["states"] = pickle.loads(state_file.read_bytes())
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            _STATE["states"] = {
                site: state for site, state in pool.map(_clean_job, sorted(cache.sites)) if state
            }
        state_file.write_bytes(pickle.dumps(_STATE["states"]))
    _STATE["neighbours"] = neighbours_of(states=_STATE["states"])
    print(_STATE["neighbours"], flush=True)
    sites = args.sites or sorted(s for s in _STATE["states"] if s != "D8")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_site_job, [(s, args.max_onsets) for s in sites]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
