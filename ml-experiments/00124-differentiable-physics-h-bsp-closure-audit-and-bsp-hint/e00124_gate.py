"""The blocking known-answer gate of experiment 124, on synthetic systems.

    uv run python ml-experiments/00124-.../e00124_gate.py
    uv run python ml-experiments/00124-.../e00124_gate_readings.py

Each synthetic system keeps the pilot's real calendars, weather, and generator series. The truth of
each Primary is its separate fit at the tuning cut, the remainder's truth is the joint fit's, and
the noise is as large and as autocorrelated as the real residuals. The BSP is built from the noisy
Primary series, so the Primaries' own noise reaches it. Half the systems close exactly; the other
half have a_P = 0.9, a remainder hump on winter evenings that the model lacks, and the same kind
of hump on the first MW Primary.

**The gate is reduced in labelled ways (made after the module was written, before any fit):** the
month-block bootstrap intervals of reading 1 and the heating-slope width of reading 2 are not run
(intervals go on no card, which is reading 1's own fallback); the score of readings 6 and 7 is the
frozen fit's mean absolute error in points and not the day-ahead quantile score; there is no
Kalman filter; and the injected Primary of readings 4 and 6 is always the first MW Primary, so the
median system is a median over noise draws. Sabotage (reading 5) drops the second MW Primary from
the BSP row.
"""

import json
import sys
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import torch
from e00124_common import TUNING_CUT, WORK_DIR, load_bsps, load_cache
from e00124_inject import ScoreRows, mae, score_rows
from e00124_pilot import SEASON_YEARS, Pilot, build_pilot
from studies.dp_features import FeatureCache
from studies.dp_gam import design_matrix
from studies.dp_hierarchy import (
    BspSeries,
    JointAssembly,
    JointFit,
    RemainderLayout,
    RemainderSpec,
    assemble_joint,
    remainder_design,
    remainder_prediction,
    solve_joint,
)

MAX_WEIGHT: Final[float] = 30.0
N_SYSTEMS: Final[int] = 100
HUMP_MW: Final[float] = 2.0
CLOSED_SHARE: Final[float] = 1.0
OPEN_SHARE: Final[float] = 0.9
SCALE_ERROR: Final[float] = 1.25
ERROR_ROWS: Final[int] = 30 * 48
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"


def ar1(*, n: int, sd: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return a stationary AR(1) series of standard deviation `sd` and lag-1 correlation `rho`."""
    shocks = rng.normal(scale=sd * np.sqrt(1.0 - rho**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(scale=sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + shocks[i]
    return out


def lag1(*, residual: np.ndarray) -> float:
    """Return the lag-1 autocorrelation of a series with gaps (pairs of adjacent finite values)."""
    a, b = residual[:-1], residual[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@dataclass(frozen=True)
class Truth:
    """What the synthetic systems share: the truths, the noise sizes, and the remainder."""

    net: dict[str, np.ndarray]
    demand: dict[str, np.ndarray]
    rho: dict[str, float]
    sd: dict[str, float]
    capacity: dict[str, float]
    r_truth: np.ndarray
    bsp_rho: float
    bsp_sd: float
    hump: np.ndarray
    rem_layout: RemainderLayout
    rem_theta: np.ndarray
    real: JointFit


def build_truth(*, cache: FeatureCache, pilot: Pilot, assembly: JointAssembly) -> Truth:
    """Fit the real data at weights 0 and 1 and take the truths and noise sizes from the fits."""
    real = solve_joint(assembly=assembly, weights=[0.0, 1.0])
    n = len(cache.end_times)
    rem_layout = assembly.remainder_layouts["pilot"]
    rem_theta = real.theta[1, real.remainder_slices["pilot"]]
    net, demand, rho, sd, capacity = {}, {}, {}, {}, {}
    for m in pilot.bsp.members:
        features = cache.sites[m.site]
        layout = assembly.layouts[m.site]
        usable = np.flatnonzero(
            np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        )
        x = design_matrix(cache=cache, features=features, layout=layout, index=usable)
        theta = real.theta[0, real.site_slices[m.site]]
        net[m.site] = np.full(n, np.nan)
        net[m.site][usable] = x @ theta
        x[:, layout.slices["capacity"]] = 0.0
        demand[m.site] = np.full(n, np.nan)
        demand[m.site][usable] = x @ theta
        index = assembly.indices[m.site]
        residual = np.full(n, np.nan)
        residual[index] = features.y[index] - net[m.site][index]
        rho[m.site] = lag1(residual=residual)
        sd[m.site] = float(np.nanstd(residual[index]))
        capacity[m.site] = features.table.effective_capacity_mw
    ok_rows = np.flatnonzero(
        np.isfinite(pilot.bsp.temperature_c)
        & np.isfinite(pilot.bsp.pv_shape)
        & np.isfinite(pilot.bsp.generators_mw)
    )
    r_truth = np.full(n, np.nan)
    r_truth[ok_rows] = (
        remainder_design(cache=cache, layout=rem_layout, bsp=pilot.bsp, index=ok_rows) @ rem_theta
    )
    bsp_index = assembly.bsp_indices["pilot"]
    modelled = r_truth - pilot.bsp.generators_mw
    for m in pilot.bsp.members:
        modelled = modelled + capacity[m.site] * (net if m.kind == "mw" else demand)[m.site]
    bsp_residual = np.full(n, np.nan)
    bsp_residual[bsp_index] = pilot.bsp.reading_mw[bsp_index] - modelled[bsp_index]
    hour = cache.calendar.half_hour / 2.0
    hump = HUMP_MW * np.exp(-(((hour - 18.5) / 1.5) ** 2)) * (pilot.bsp.temperature_c < 5.0)
    return Truth(
        net=net,
        demand=demand,
        rho=rho,
        sd=sd,
        capacity=capacity,
        r_truth=r_truth,
        bsp_rho=lag1(residual=bsp_residual),
        bsp_sd=float(np.nanstd(bsp_residual[bsp_index])),
        hump=hump,
        rem_layout=rem_layout,
        rem_theta=rem_theta,
        real=real,
    )


def simulate(
    *, truth: Truth, pilot: Pilot, first_mw: str, open_system: bool, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Draw one system: the Primaries' noisy series and the BSP reading built from them."""
    n = len(truth.r_truth)
    power = {
        m.site: truth.net[m.site] + ar1(n=n, sd=truth.sd[m.site], rho=truth.rho[m.site], rng=rng)
        for m in pilot.bsp.members
    }
    if open_system:
        power[first_mw] = power[first_mw] + truth.hump / truth.capacity[first_mw]
    share = OPEN_SHARE if open_system else CLOSED_SHARE
    reading = truth.r_truth - pilot.bsp.generators_mw + (truth.hump if open_system else 0.0)
    for m in pilot.bsp.members:
        noisy = power[m.site] + (
            0.0 if m.kind == "mw" else truth.demand[m.site] - truth.net[m.site]
        )
        reading = reading + share * truth.capacity[m.site] * noisy
    return power, reading + ar1(n=n, sd=truth.bsp_sd, rho=truth.bsp_rho, rng=rng)


def level(*, cache: FeatureCache, fit: JointFit, bsp: BspSeries, index: np.ndarray) -> list[float]:
    """Return the mean fitted remainder over `index` for each weight."""
    return [
        float(
            np.mean(
                remainder_prediction(cache=cache, fit=fit, bsp=bsp, index=index, weight_index=k)
            )
        )
        for k in range(len(fit.weights))
    ]


def capacity_bias(*, fit: JointFit, truth: Truth, site: str) -> list[float]:
    """Return each weight's error in the latest season-year's capacity at a Primary."""
    layout = fit.site_layouts[site]
    year = fit.last_season_years[site]
    true_cumulative = float(
        np.cumsum(truth.real.theta[0, truth.real.site_slices[site]][layout.slices["capacity"]])[
            layout.season_years.index(year)
        ]
    )
    site_fit = fit.site_fit(site=site)
    return [
        site_fit.capacity_points(member=k)[year] - true_cumulative for k in range(len(fit.weights))
    ]


def main() -> None:
    """Run the gate and write `gate.json`."""
    w_star = json.loads((WORK_DIR / "tuning.json").read_text())["w_star"]
    weights = [0.0, w_star, MAX_WEIGHT]
    cache = load_cache()
    pilot = build_pilot(cache=cache, bsps=load_bsps(cache=cache), cut=TUNING_CUT)

    def assemble(
        *, specs: dict, bsp: BspSeries, window: dict, sigma_points: dict[str, float]
    ) -> JointAssembly:
        return assemble_joint(
            specs=specs,
            bsps=[bsp],
            window=window,
            cache=cache,
            remainder_spec=RemainderSpec(),
            sigma_points=sigma_points,
            cut=TUNING_CUT,
            season_years=SEASON_YEARS,
            device=DEVICE,
        )

    assembly = assemble(
        specs=pilot.specs, bsp=pilot.bsp, window=pilot.windows, sigma_points=pilot.sigma_points
    )
    truth = build_truth(cache=cache, pilot=pilot, assembly=assembly)
    first_mw, second_mw, *_ = pilot.mw_sites
    smaller = replace(pilot.bsp, members=tuple(m for m in pilot.bsp.members if m.site != second_mw))
    kept = [m.site for m in smaller.members]
    sabotaged = assemble(
        specs={s: pilot.specs[s] for s in kept},
        bsp=smaller,
        window={s: pilot.windows[s] for s in kept},
        sigma_points={s: pilot.sigma_points[s] for s in kept},
    )
    probe = solve_joint(assembly=assembly, weights=weights)
    scores = {
        s: score_rows(cache=cache, assembly=assembly, fit=probe, site=s, cut=TUNING_CUT)
        for s in pilot.mw_sites
    }
    index = assembly.bsp_indices["pilot"][::20]
    master = np.random.default_rng(2024)
    rows = []
    for system in range(N_SYSTEMS):
        rng = np.random.default_rng(master.integers(2**32))
        open_system = system >= N_SYSTEMS // 2
        power, reading = simulate(
            truth=truth, pilot=pilot, first_mw=first_mw, open_system=open_system, rng=rng
        )
        clean = solve_joint(
            assembly=assembly,
            weights=weights,
            primary_power=power,
            bsp_readings={"pilot": reading},
        )
        start = int(rng.choice(assembly.indices[first_mw][-17520:-ERROR_ROWS]))
        bad = power[first_mw].copy()
        bad[start : start + ERROR_ROWS] *= SCALE_ERROR
        injected = solve_joint(
            assembly=assembly,
            weights=weights,
            primary_power={**power, first_mw: bad},
            bsp_readings={"pilot": reading},
        )
        broken = solve_joint(
            assembly=sabotaged,
            weights=weights,
            primary_power=power,
            bsp_readings={"pilot": reading},
        )

        def score(fit: JointFit, site: str, power: dict[str, np.ndarray] = power) -> np.ndarray:
            observed = ScoreRows(
                design=scores[site].design,
                observed=power[site][scores[site].rows],
                rows=scores[site].rows,
            )
            return mae(score=observed, fit=fit, site=site)

        rows.append(
            {
                "system": system,
                "open": open_system,
                "clean_mae": {s: score(clean, s).tolist() for s in pilot.mw_sites},
                "damage": (score(injected, first_mw) - score(clean, first_mw)).tolist(),
                "sibling_damage": {
                    s: (score(injected, s) - score(clean, s)).tolist()
                    for s in pilot.mw_sites
                    if s != first_mw
                },
                "level": level(cache=cache, fit=clean, bsp=pilot.bsp, index=index),
                "level_truth": float(np.mean(truth.r_truth[index])),
                "level_sabotaged": level(cache=cache, fit=broken, bsp=smaller, index=index),
                "capacity_bias": {
                    s: capacity_bias(fit=clean, truth=truth, site=s) for s in pilot.mw_sites
                },
                "pv": [
                    float(
                        clean.theta[k, clean.remainder_slices["pilot"]][
                            truth.rem_layout.slices["pv"]
                        ][0]
                    )
                    for k in range(len(weights))
                ],
            }
        )
        sys.stdout.write(f"system {system}\n")
        sys.stdout.flush()
    pv_truth = float(truth.rem_theta[truth.rem_layout.slices["pv"]][0])
    (WORK_DIR / "gate.json").write_text(
        json.dumps({"weights": weights, "pv_truth": pv_truth, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
