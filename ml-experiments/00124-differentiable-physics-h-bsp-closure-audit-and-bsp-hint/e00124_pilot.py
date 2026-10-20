"""The pilot's inputs: the joint fit of the better-closing BSP's Primaries, at one cut.

The step 0 rule made neither BSP informative, so step 1 runs as a pilot on one BSP (the second in
the hierarchy file, the one that closes in the training years). Every private fact (which Primaries
sit beneath it, which generation series it has) is read at run time and never printed.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
from e00124_audit import (
    MIN_SERIES_COVERAGE,
    build_design,
    robust_scale,
    select_series,
    usable_rows,
)
from e00124_audit import fit as audit_fit
from e00124_common import WORK_DIR, BspSeries, sum_all, to_64
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.dp_hierarchy import BspMember
from studies.dp_hierarchy import BspSeries as HierarchyBsp

PILOT_BSP: Final[int] = 1
"""The pilot BSP's position in the hierarchy file."""
SPEC: Final[GamSpec] = GamSpec(ridge=0.01, drift="d1", drift_step=3000.0)
"""Experiment 106's smooth drift with its tuned step; no L1 steps, so the joint problem has about
300 columns per Primary and not 800, and the capacity is fitted, not pinned. Both are labelled
simplifications of the pilot, which has no real-time capacity vintages."""
FULL_WINDOW: Final[TrainingWindow] = TrainingWindow()
DARK_WINDOW: Final[TrainingWindow] = TrainingWindow(dark_only=True)
WEIGHT_GRID: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
CLOSURE_STEP_PENALTY: Final[float] = 100.0


@dataclass(frozen=True)
class Pilot:
    """The pilot BSP's joint-fit inputs at one cut.

    Attributes:
        cache: The feature cache.
        bsp: The BSP for `studies.dp_hierarchy`.
        specs: The convex model's spec of each member.
        windows: The training window of each member.
        sigma_points: Each member's robust half-hourly residual scale in points, pre-cut.
        cut: The cut.
    """

    cache: FeatureCache
    bsp: HierarchyBsp
    specs: dict[str, GamSpec]
    windows: dict[str, TrainingWindow]
    sigma_points: dict[str, float]
    cut: datetime

    @property
    def mw_sites(self) -> list[str]:
        """The members that read real power."""
        return [m.site for m in self.bsp.members if m.kind == "mw"]


def _covered(
    *, series: list[np.ndarray], reading: np.ndarray, cache: FeatureCache, cut: np.datetime64
) -> list[int]:
    """Return the positions of the series that hold readings where the BSP does, before the cut."""
    window = np.isfinite(reading) & (cache.end_times <= cut)
    return [i for i, s in enumerate(series) if np.isfinite(s[window]).mean() >= MIN_SERIES_COVERAGE]


def _sigma(
    *, cache: FeatureCache, site: str, spec: GamSpec, window: TrainingWindow, cut: datetime
) -> float:
    """Return a member's pre-cut robust residual scale from its separate fit, cached on disk."""
    path = WORK_DIR / "sigma_points.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    key = f"{site}|{cut.date()}|{spec}|{window}"
    if key not in store:
        fit = fit_gam(
            cache=cache, site=site, specs=[spec], cut=cut, window=window, season_years=SEASON_YEARS
        )
        from studies.dp_gam import prepare_fit

        _, index = prepare_fit(
            cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=SEASON_YEARS
        )
        residual = cache.sites[site].y[index] - predict_gam(cache=cache, fit=fit, index=index)
        store[key] = robust_scale(values=residual)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(store))
        temporary.replace(path)
    return float(store[key])


def build_pilot(*, cache: FeatureCache, bsps: list[BspSeries], cut: datetime) -> Pilot:
    """Build the pilot's inputs at `cut` from the audit's series.

    Args:
        cache: The feature cache.
        bsps: The audit's BSP series, from `load_bsps`.
        cut: The cut.

    Returns:
        The pilot, whose members are the MW and MVA Primaries and whose generators are the
        generation series with enough history before the cut.
    """
    series = bsps[PILOT_BSP]
    cut64 = to_64(moment=cut)
    mw = _covered(series=series.mw_list, reading=series.reading, cache=cache, cut=cut64)
    mva = _covered(series=series.mva_list, reading=series.reading, cache=cache, cut=cut64)
    gen = _covered(series=series.generator_list, reading=series.reading, cache=cache, cut=cut64)
    members = tuple(BspMember(site=series.mw_labels[i], kind="mw") for i in mw) + tuple(
        BspMember(site=series.mva_labels[i], kind="mva_dark") for i in mva
    )
    specs = {m.site: SPEC for m in members}
    windows = {m.site: FULL_WINDOW if m.kind == "mw" else DARK_WINDOW for m in members}
    sigma_points = {
        m.site: _sigma(cache=cache, site=m.site, spec=SPEC, window=windows[m.site], cut=cut)
        for m in members
    }
    sums = select_series(cache=cache, bsp=series, cut=cut64)
    first_day = float(
        cache.end_times[np.flatnonzero(np.isfinite(series.reading) & (cache.end_times <= cut64))[0]]
        .astype("datetime64[s]")
        .astype(np.int64)
        / 86400.0
    )
    design = build_design(
        cache=cache,
        bsp=series,
        sums=sums,
        cut_day=float(cut64.astype("datetime64[s]").astype(np.int64) / 86400.0),
        first_day=first_day,
    )
    index = usable_rows(cache=cache, bsp=series, design=design, cut=cut64)
    theta = audit_fit(
        x=design.matrix[index],
        y=series.reading[index],
        design=design,
        step_penalty=CLOSURE_STEP_PENALTY,
    )
    sigma_b = robust_scale(values=series.reading[index] - design.matrix[index] @ theta)
    bsp = HierarchyBsp(
        name="pilot",
        members=members,
        reading_mw=series.reading,
        generators_mw=sum_all(
            arrays=[series.generator_list[i] for i in gen], n=len(cache.end_times)
        ),
        temperature_c=series.temperature_c,
        pv_shape=np.nanmean(np.stack([cache.sites[m.site].pv_unit_shape for m in members]), axis=0),
        sigma_mw=sigma_b,
    )
    return Pilot(
        cache=cache, bsp=bsp, specs=specs, windows=windows, sigma_points=sigma_points, cut=cut
    )
