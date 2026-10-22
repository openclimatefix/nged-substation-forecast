"""Injected-transfer worlds: refit the three arms with drawn transfers added to the training readings.

A world is one of experiment 195's `eval_held` worlds (fold k, world w < 4). Its drawn transfers on
a substation's series add moved megawatts (MW) to the training readings only; the Kalman filter and
every score read the true readings. In a world, arm A trains on everything (with the injected MW in
it), arm B masks the true interval of every injected transfer plus the hand labels, and arm C masks
the real-record alarms of fold k plus the injected transfers that the re-ranker `found`, from the
step stamp once known (192 half-hours) to the end once known (end plus 60 half-hours).

Lab shortcuts: the moved profile is the drawn shape without experiment 195's random drift; a
receiver takes an equal share of the moved MW (the stored events do not record the weights); the
ramp shape is a ramp up over 12 half-hours and down at the end.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00212_worlds.py <fold> <world> <site>
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import replace

import e00212_fed as fed
import e00212_fits as fits_module
import e00212_g_common as common
import e00212_g_members as members
import numpy as np
import polars as pl
import torch
from e00212_common import E195_DIR, GRID_LENGTH, GRID_START, WORK_DIR, load_panel, stamp_months
from e00212_g_arms import quantile_score
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

import json

MAD_TO_STD = 1.4826
ONSET_KNOWN = 192
END_KNOWN = 60
N_WORLDS = 4
LATITUDE_DEG = 52.5
RAMP = 12
TUNING_MONTHS = {1, 3, 5, 7, 9, 11}


def noise_units(*, residual: dict[int, np.ndarray]) -> dict[int, float]:
    """Each series' noise unit: 1.4826 times the MAD of its residual in even-numbered months."""
    even = ~np.isin(stamp_months(), list(TUNING_MONTHS))
    out = {}
    for series, values in residual.items():
        chosen = values[even & np.isfinite(values)]
        out[series] = (
            float(MAD_TO_STD * np.median(np.abs(chosen - np.median(chosen))))
            if chosen.size > 1000
            else float("nan")
        )
    return out


def pv_profile(*, local_hours: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """A daylight bell: zero before sunrise and after sunset, one at solar noon."""
    declination = np.radians(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    cos_h0 = np.clip(-np.tan(np.radians(LATITUDE_DEG)) * np.tan(declination), -1.0, 1.0)
    day_length = 2 * np.degrees(np.arccos(cos_h0)) / 15.0
    return np.clip(np.cos(np.pi * (local_hours - 12.0) / np.maximum(day_length, 1.0)), 0.0, None)


def moved_mw(*, event: dict, sigma: dict[int, float], power: dict[int, np.ndarray], local) -> np.ndarray:
    """The MW moved at each stamp of an event: mean `depth_z` noise units of the donor."""
    onset, end = event["onset"], event["end"]
    length = end - onset
    shape = event["shape"]
    if shape == "constant":
        base = np.ones(length)
    elif shape == "demand":
        donor = power[event["donor"]][onset:end]
        if np.isfinite(donor).any():
            filled = np.nan_to_num(donor, nan=np.nanmedian(donor))
            base = np.clip(filled, 0.05 * max(float(np.mean(filled)), 1e-6), None)
        else:
            base = np.ones(length)
    elif shape == "pv":
        base = pv_profile(local_hours=local[0][onset:end], day_of_year=local[1][onset:end]) + 0.02
    else:
        t = np.arange(length)
        base = np.minimum(np.clip((t + 1) / RAMP, 0, 1), np.clip((length - t) / RAMP, 0, 1))
    return event["depth_z"] * sigma[event["donor"]] * base / max(float(base.mean()), 1e-9)


def local_calendar() -> tuple[np.ndarray, np.ndarray]:
    """Local hour of day and day of year of every detector stamp."""
    index = pl.datetime_range(
        GRID_START, GRID_START.replace(year=2026, month=7), interval="30m", closed="left",
        time_zone="UTC", eager=True,
    ).dt.convert_time_zone("Europe/London")
    hour = index.dt.hour().to_numpy() + index.dt.minute().to_numpy() / 60.0
    return hour, index.dt.ordinal_day().to_numpy()


def world_events(*, fold: int, world: int, series: int) -> pl.DataFrame:
    """The world's drawn transfers on a series, with the re-ranker's `found` flag."""
    events = pl.read_parquet(
        E195_DIR / "worlds" / f"f{fold}_eval_held_{world}_events.parquet"
    ).filter(pl.col("series").list.contains(series))
    found = (
        pl.read_parquet(E195_DIR / f"injected_f{fold}.parquet")
        .filter((pl.col("kind") == "eval_held") & (pl.col("arm") == "R re-ranker") & (pl.col("world") == world))
        .select("event_id", "found")
    )
    return events.join(found, on="event_id", how="left").with_columns(pl.col("found").fill_null(False))


def delta_points(*, events: pl.DataFrame, series: int, site: str, offset: int, n: int) -> np.ndarray:
    """The MW the world adds at a site's series, in points, on the cache grid."""
    panel = load_panel()
    sigma = noise_units(residual=panel["residual"])
    local = local_calendar()
    out = np.zeros(n)
    for event in events.iter_rows(named=True):
        if not np.isfinite(sigma[event["donor"]]):
            msg = f"donor {event['donor']} has no noise unit"
            raise ValueError(msg)
        moved = moved_mw(event=event, sigma=sigma, power=panel["power"], local=local)
        sign = -1.0 if event["donor"] == series else 1.0 / max(event["n_receivers"], 1)
        out[event["onset"] + offset : event["end"] + offset] += (
            sign * 100.0 * moved / fed.CAPACITY_MW[site]
        )
    return out


def interval_mask(*, intervals: list[tuple[int, int, int, int]], n: int, cut_position: int) -> np.ndarray:
    """Mask (onset, end, start, known) intervals as the alarms known at a cut."""
    mask = np.zeros(n, dtype=bool)
    for onset, end, start, known in intervals:
        if start > cut_position:
            continue
        stop = end if known <= cut_position else cut_position + 1
        mask[max(onset, 0) : max(min(stop, n), 0)] = True
    return mask


def fit_world_arm(*, cache: FeatureCache, site: str, arm: str, name: str, y: np.ndarray, truth: list, found: list, spans) -> None:  # noqa: ANN001
    """Fit one world arm at every monthly cut and store the fits under the arm's name."""
    settings = json.loads(common.PART_4_TUNED.read_text())
    drift = json.loads((common.SRC_WORK_DIR / "drift.json").read_text())["chosen_step"]
    spec = GamSpec(ridge=0.1, drift_step=float(drift), step_penalty=settings["step_penalty"],
                   capacity="pinned", final_knot="free", drift="none")
    vintages = read_vintages()
    features = cache.sites[site]
    base_rows = features.default_rows
    all_rows = base_rows | (features.switching_mask & features.in_pool & ~features.lockdown)
    cuts = [f.cut for f in members.load_fits(site=site)]
    fits = []
    n = len(base_rows)
    for cut in cuts:
        position = int(np.searchsorted(cache.end_times, naive_utc(moment=cut), side="right"))
        if arm == "A":
            rows = all_rows
        elif arm == "B":
            rows = base_rows & ~interval_mask(intervals=truth, n=n, cut_position=position)
        else:
            rows = all_rows & ~fits_module.detector_mask(spans=spans, n=n, cut_position=position)
            rows = rows & ~interval_mask(intervals=found, n=n, cut_position=position)
        world_cache = replace(
            cache, sites={**cache.sites, site: replace(features, y=y, default_rows=rows)}
        )
        fits.append(
            fit_gam(cache=world_cache, site=site, specs=[spec], cut=cut,
                    window=TrainingWindow(months=fits_module.WINDOW_MONTHS),
                    season_years=common.SEASON_YEARS,
                    pinned=pinned_capacity(vintages=vintages, table=features.table, cut=cut))
        )
    target = WORK_DIR / "fits" / name / f"{site}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pickle.dumps(fits))


def main(fold: int, world: int, site: str) -> None:
    """Fit, forecast, and score the three arms of one (fold, world, site)."""
    torch.set_num_threads(1)
    started = time.time()
    series = fed.SERIES_OF_SITE[site]
    events = world_events(fold=fold, world=world, series=series)
    if events.height == 0:
        print(f"f{fold} w{world} {site}: no events on the series")
        return
    cache = common.load_cache()
    base = fed.get_site(site, "B")
    offset = fed.stamp_offset(base)
    n = len(cache.end_times)
    delta = delta_points(events=events, series=series, site=site, offset=offset, n=n)
    y = np.where(np.isfinite(cache.sites[site].y), cache.sites[site].y + delta, np.nan)
    truth = [(r["onset"] + offset, r["end"] + offset, r["onset"] + offset, r["end"] + offset)
             for r in events.iter_rows(named=True)]
    found = [(r["onset"] + offset, r["end"] + offset, r["onset"] + offset + ONSET_KNOWN,
              r["end"] + offset + END_KNOWN)
             for r in events.iter_rows(named=True) if r["found"]]
    spans = fits_module.detector_spans(cache=cache, site=site, fold=fold)
    meta = pl.read_parquet(WORK_DIR / f"meta_{site}.parquet").filter(pl.col("run_period") == "scored")
    rows = meta["row"].to_numpy()
    spread = fed.k0_spread(site)
    out = []
    for arm in ("A", "B", "C"):
        name = f"W{fold}{world}{arm}"
        fit_world_arm(cache=cache, site=site, arm=arm, name=name, y=y, truth=truth, found=found, spans=spans)
        members.build_site(site, name)
        data = fed.get_site(site, name)
        correction = fed.correction_for(data=data, stream=data.stream, prior=np.zeros(len(data.stream)), multiplier=1.0)
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        out.append(pl.DataFrame({
            "arm": arm, "row": rows,
            "qs": quantile_score(actual=data.actual[rows], quantiles=q).astype(np.float32),
            "mean_forecast": (data.member_mean[rows] + correction[rows]).astype(np.float32),
        }))
        (WORK_DIR / "fits" / name / f"{site}.pkl").unlink()
        members.members_path(site=site, arm=name).unlink()
    target = WORK_DIR / "worlds" / f"scores_f{fold}_w{world}_{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(out).with_columns(fold=pl.lit(fold), world=pl.lit(world), site=pl.lit(site),
                                n_events=pl.lit(events.height), n_found=pl.lit(int(events["found"].sum()))).write_parquet(target)
    print(f"f{fold} w{world} {site}: {events.height} events, {int(events['found'].sum())} found, {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
