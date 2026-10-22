"""Refit experiment 111's GAM-K at every monthly cut with three training masks.

Arms: `A` trains on the rows of arm `B` plus the hand-filtered switching rows (no training mask);
`B` is experiment 111's `gamk` fit (hand labels masked), reused from its store and refitted here
only as a positive control; `C<k>` trains on the rows of `A` minus the half-hours inside the
learned-score alarm intervals of experiment 195's fold `k` that are known at the cut.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00212_fits.py D5 A
    uv run --no-sync python e00212_fits.py D5 B --control   # positive control: arm B must equal experiment 111's stored fits
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import e00212_alarms as al
import e00212_fed as fed
import e00212_g_common as common
import e00212_g_members as members
import e00212_nra as nra
import numpy as np
import torch
from e00212_common import WORK_DIR
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, GamSpec, TrainingWindow, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

RIDGE = 0.1
WINDOW_MONTHS = 24
ARMS = ("A", "B", "C0", "C1", "C2")
FITS_DIR = WORK_DIR / "fits"


@dataclass(frozen=True)
class Spans:
    """Detector alarm intervals of one fold in cache grid positions."""

    onset: np.ndarray
    end: np.ndarray
    start: np.ndarray
    known: np.ndarray


def detector_spans(*, cache: FeatureCache, site: str, fold: int) -> Spans:
    """Return a fold's learned-score alarm intervals at a site, as experiment 203 replays them."""
    table = al.learned_tables(arm="LEARNED_R")[fold]
    shim = SimpleNamespace(cache=cache)
    spans = nra.detected_spans(data=shim, site=site, intervals=table)
    return Spans(
        onset=np.array([s.onset for s in spans], dtype=np.int64),
        end=np.array([s.end for s in spans], dtype=np.int64),
        start=np.array([s.start for s in spans], dtype=np.int64),
        known=np.array([s.known for s in spans], dtype=np.int64),
    )


def detector_mask(*, spans: Spans, n: int, cut_position: int) -> np.ndarray:
    """Return the grid positions masked by the alarms known at a cut.

    An alarm counts once its step is known (`start`); its end counts only once known (`known`),
    and an interval whose end is not yet known runs to the cut.
    """
    mask = np.zeros(n, dtype=bool)
    for onset, end, start, known in zip(spans.onset, spans.end, spans.start, spans.known, strict=True):
        if start > cut_position:
            continue
        stop = end if known <= cut_position else cut_position + 1
        mask[max(onset, 0) : max(min(stop, n), 0)] = True
    return mask


def arm_cache(
    *, cache: FeatureCache, site: str, arm: str, cut_position: int, spans: Spans | None
) -> FeatureCache:
    """Return the cache with `default_rows` replaced by the arm's training rows at one cut."""
    features = cache.sites[site]
    if arm == "B":
        rows = features.default_rows
    else:
        rows = features.default_rows | (features.switching_mask & features.in_pool & ~features.lockdown)
        if arm != "A":
            assert spans is not None
            rows = rows & ~detector_mask(spans=spans, n=len(rows), cut_position=cut_position)
    return replace(cache, sites={**cache.sites, site: replace(features, default_rows=rows)})


def fit_arm(*, site: str, arm: str, control: bool = False) -> str:
    """Fit one arm at one site for every monthly cut, and store the fits atomically."""
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    stored = members.load_fits(site=site)
    cuts = [f.cut for f in stored]
    settings = json.loads(common.PART_4_TUNED.read_text())
    drift = json.loads((common.SRC_WORK_DIR / "drift.json").read_text())["chosen_step"]
    spec = GamSpec(
        ridge=RIDGE,
        drift_step=float(drift),
        step_penalty=settings["step_penalty"],
        capacity="pinned",
        final_knot="free",
        drift="none",
    )
    spans = detector_spans(cache=cache, site=site, fold=int(arm[1])) if arm[0] == "C" else None
    fits: list[GamFit] = []
    for cut in cuts:
        position = int(np.searchsorted(cache.end_times, naive_utc(moment=cut), side="right"))
        fits.append(
            fit_gam(
                cache=arm_cache(cache=cache, site=site, arm=arm, cut_position=position, spans=spans),
                site=site,
                specs=[spec],
                cut=cut,
                window=TrainingWindow(months=WINDOW_MONTHS),
                season_years=common.SEASON_YEARS,
                pinned=pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut),
            )
        )
    if control:
        worst = max(
            float(np.max(np.abs(new.theta - old.theta) / np.maximum(np.abs(old.theta), 1e-12)))
            for new, old in zip(fits, stored, strict=True)
        )
        status = "PASS" if worst <= 1e-8 else "FAIL"
        return f"control {site}: {len(fits)} cuts, max relative theta diff {worst:.3e} {status}, {time.monotonic() - started:.0f} s"
    target = FITS_DIR / arm / f"{site}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return f"{arm} {site}: {len(fits)} fits, {time.monotonic() - started:.0f} s"


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(fit_arm(site=sys.argv[1], arm=sys.argv[2], control="--control" in sys.argv), flush=True)
