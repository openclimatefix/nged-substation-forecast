"""The capacity profile re-run (pre-registered, labelled pass or fail).

**Does VP find the profile minimum from any start?** VP solves the capacity exactly, so a start can
matter only through the non-linear parameters. At two configurations (experiment 105's diagnostic,
K = 4 at the 2024-07-01 cut, and VP-P's, K = 2 at the 2025-07-01 cut) one multiplier scales the
convex capacity path (Part 2's model on all years at the cut) at every MW substation, and the model
is refitted with the capacity pinned for the multipliers 0.5, 0.7, 0.85, 1.0, 1.15, and 1.3. Free
fits then run from three starts: the default, seed 1 with the thresholds at 17 degrees, and the
non-linear parameters of an Adam fit on the same objective (P-map-same's analogue at that K).

**Pass, every line.** (1) The starts' capacities agree within 3 points at every MW substation.
(2) Each free objective is no higher than the lowest profile point plus 1 nat per 10,000 rows.
(3) Each free capacity lies where the profile is within that tolerance of its minimum. The noise
standard deviation is round 1 of the cut (the per-site convex model's training residuals), so the
profile runs before the arms. Every objective is `vp_objective` minus its capacity prior term,
which a pinned fit does not have, so the pinned and the free fits share one objective. Each
substation's capacity is shown beside the convex model's, experiment 105's P (K = 4 at the
2024-07-01 cut), and the vintages' season-year mean.

    uv run --no-sync python ml-experiments/00112-*/e00112_profile.py [--configs k4_2024 k2_2025]
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import e00112_common as common
import e00112_vp as vp
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel, PopulationScales
from studies.dp_pooled_vp import (
    VpSettings,
    capacity_prior_term,
    fit_pooled_adam_on_vp_objective,
    fit_pooled_vp,
    vp_objective,
)

MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3)
AGREEMENT_POINTS: Final[float] = 3.0
TOLERANCE_NATS_PER_10000_ROWS: Final[float] = 1.0
PROFILE_FALLBACK_STEPS: Final[int] = 400


@dataclass(frozen=True)
class ProfileConfig:
    """One configuration of the profile.

    Attributes:
        name: The configuration's label.
        cut: The cut the profile is fitted at.
        shapes: The number of shared daily shapes `K`.
    """

    name: str
    cut: datetime
    shapes: int


CONFIGS: Final[dict[str, ProfileConfig]] = {
    "k4_2024": ProfileConfig(name="k4_2024", cut=common.REPLICATE_CUT, shapes=4),
    "k2_2025": ProfileConfig(name="k2_2025", cut=common.FINAL_CUT, shapes=2),
}


def pooled_config(*, shapes: int) -> DpConfig:
    """Return the VP-P configuration with `shapes` daily shapes."""
    tuned = common.tuned()
    return DpConfig(
        shared_shapes=shapes,
        interaction_hidden=common.INTERACTION,
        population_scales=PopulationScales(deviation=tuned["map_deviation"]),
    )


def fresh_model(*, batch: PooledBatch, shapes: int, start: str) -> PooledDemandModel:
    """Return a new model with `shapes` daily shapes, started as `vp.new_model` does."""
    model = vp.new_model(batch=batch, start=start)
    if shapes == model.config.shared_shapes:
        return model
    torch.manual_seed(0 if start == "s0" else 1)
    rebuilt = PooledDemandModel(
        config=pooled_config(shapes=shapes),
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    if start == "s1":
        with torch.no_grad():
            rebuilt.threshold.fill_(common.START_THRESHOLD_C)
            rebuilt.population_mean["threshold"].fill_(common.START_THRESHOLD_C)
    vp.freeze_operator_for_vp(model=rebuilt)
    return rebuilt


def convex_capacity_path(*, cache: FeatureCache, cut: datetime) -> dict[str, dict[int, float]]:
    """Return Part 2's convex model (all years, no drift) capacities per season-year, in points."""
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    out = {}
    for site in common.SCORED_SITES:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
        )
        out[site] = {int(year): float(value) for year, value in fit.capacity_points().items()}
    return out


def override_for(
    *, batch: PooledBatch, convex: dict[str, dict[int, float]], multiplier: float
) -> torch.Tensor:
    """Return `multiplier` times the convex capacity path, in points; `NaN` at other substations."""
    path = np.full((len(batch.site_labels), batch.n_season_years), np.nan)
    for site, by_year in convex.items():
        row = batch.site_labels.index(site)
        previous = 0.0
        for offset in range(batch.n_season_years):
            year = batch.first_season_year + offset
            previous = by_year.get(year, previous)
            path[row, offset] = multiplier * previous
    return torch.as_tensor(path, dtype=batch.y.dtype, device=batch.y.device)


def shared_objective(
    *, model: PooledDemandModel, batch: PooledBatch, settings: VpSettings
) -> float:
    """Return the objective in nats without the capacity prior, so pinned and free fits agree."""
    with torch.no_grad():
        return float(
            vp_objective(model=model, batch=batch, settings=settings)
            - capacity_prior_term(model=model, settings=settings)
        )


def final_capacity(*, model: PooledDemandModel, batch: PooledBatch) -> dict[str, float]:
    """Return each scored MW substation's capacity in its last season-year, in points."""
    capacity = model.capacities().detach().cpu().numpy()
    return {
        site: float(capacity[batch.site_labels.index(site), -1]) for site in common.SCORED_SITES
    }


def vintage_season_means(*, cache: FeatureCache, cut: datetime) -> dict[str, dict[int, float]]:
    """Return the vintages' mean capacity per season-year (July to June), in points."""
    series = pl.read_parquet(common.SERIES_TABLE).select("time_series_id", "site")
    vintages = (
        pl.read_parquet(common.VINTAGES)
        .filter(pl.col("variant") == "frozen")
        .filter(pl.col("origin_monday") <= cut.date())
        .filter(pl.col("week_start") <= cut.date())
        .join(series, on="time_series_id")
    )
    out: dict[str, dict[int, float]] = {}
    for site in common.SCORED_SITES:
        rows = vintages.filter(pl.col("site") == site)
        if rows.is_empty():
            continue
        latest = rows.filter(pl.col("origin_monday") == rows["origin_monday"].max())
        year = latest.with_columns(
            season_year=pl.when(pl.col("week_start").dt.month() >= 7)
            .then(pl.col("week_start").dt.year())
            .otherwise(pl.col("week_start").dt.year() - 1)
        )
        means = year.group_by("season_year").agg(pl.col("capacity_mw").mean())
        effective = common.effective_capacity_mw(cache=cache, site=site)
        out[site] = {
            int(row["season_year"]): float(100.0 * row["capacity_mw"] / effective)
            for row in means.iter_rows(named=True)
        }
    return out


def adam_start_state(*, batch: PooledBatch, settings: VpSettings, shapes: int) -> dict:
    """Return the state of an Adam fit on the VP objective, the third start."""
    model = fresh_model(batch=batch, shapes=shapes, start="s0")
    fit_pooled_adam_on_vp_objective(
        model=model,
        batch=batch,
        settings=settings,
        steps=common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
    )
    return model.state_dict()


def run_config(*, cache: FeatureCache, config: ProfileConfig) -> dict[str, Any]:
    """Fit the profile and the three free starts of one configuration; return the checks."""
    batch = common.pooled_batch_for(
        cache=cache, sites=common.POOL, cut=config.cut, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )
    settings = common.vp_settings(cut=config.cut, round_name="round1")
    convex = convex_capacity_path(cache=cache, cut=config.cut)
    rows = batch.n_rows
    tolerance = TOLERANCE_NATS_PER_10000_ROWS * rows / 1e4
    free: dict[str, dict[str, Any]] = {}
    states: dict[str, dict] = {}
    third = adam_start_state(batch=batch, settings=settings, shapes=config.shapes)
    for name, state in (("default", None), ("seed1_17c", None), ("adam", third)):
        start = "s1" if name == "seed1_17c" else "s0"
        model = fresh_model(batch=batch, shapes=config.shapes, start=start)
        if state is not None:
            model.load_state_dict(state)
        result = fit_pooled_vp(
            model=model,
            batch=batch,
            settings=settings,
            adam_steps=common.FALLBACK_STEPS_SECONDARY,
            warm_start_steps=0 if state is not None else common.WARM_START_STEPS,
        )
        states[name] = model.state_dict()
        free[name] = {
            "objective_without_capacity_prior": shared_objective(
                model=model, batch=batch, settings=settings
            ),
            "capacity_points": final_capacity(model=model, batch=batch),
            "converged": result.converged,
            "seconds": result.seconds,
            "thresholds": model.threshold.detach().cpu().tolist(),
        }
        print(
            f"{config.name} free {name}: {free[name]['objective_without_capacity_prior']:.0f}",
            flush=True,
        )
    profile: dict[float, dict[str, Any]] = {}
    for multiplier in MULTIPLIERS:
        model = fresh_model(batch=batch, shapes=config.shapes, start="s0")
        model.load_state_dict(states["default"])
        model.capacity_override = override_for(batch=batch, convex=convex, multiplier=multiplier)
        result = fit_pooled_vp(
            model=model,
            batch=batch,
            settings=settings,
            adam_steps=PROFILE_FALLBACK_STEPS,
            warm_start_steps=0,
        )
        profile[multiplier] = {
            "objective": shared_objective(model=model, batch=batch, settings=settings),
            "converged": result.converged,
        }
        print(
            f"{config.name} profile {multiplier}: {profile[multiplier]['objective']:.0f}",
            flush=True,
        )
    return {
        "rows": rows,
        "tolerance_nats": tolerance,
        "free": free,
        "profile": {str(m): v for m, v in profile.items()},
        "convex_capacity_points": {site: by_year[max(by_year)] for site, by_year in convex.items()},
        "vintage_season_means": vintage_season_means(cache=cache, cut=config.cut),
        "checks": checks(free=free, profile=profile, convex=convex, tolerance=tolerance),
    }


def checks(
    *,
    free: dict[str, dict[str, Any]],
    profile: dict[float, dict[str, Any]],
    convex: dict[str, dict[int, float]],
    tolerance: float,
) -> dict[str, Any]:
    """Return the three pre-registered lines, each labelled pass or fail."""
    capacities = {name: entry["capacity_points"] for name, entry in free.items()}
    spreads = {
        site: max(c[site] for c in capacities.values()) - min(c[site] for c in capacities.values())
        for site in common.SCORED_SITES
    }
    multipliers = np.array(sorted(profile))
    values = np.array([profile[m]["objective"] for m in multipliers])
    lowest = float(values.min())
    objective_line = {
        name: entry["objective_without_capacity_prior"] - lowest for name, entry in free.items()
    }
    near_minimum = {}
    for name, entry in free.items():
        implied = [
            entry["capacity_points"][site] / convex[site][max(convex[site])]
            for site in common.SCORED_SITES
        ]
        inside = [
            bool(
                multipliers[0] <= m <= multipliers[-1]
                and float(np.interp(m, multipliers, values)) - lowest <= tolerance
            )
            for m in implied
        ]
        near_minimum[name] = {"implied_multipliers": implied, "inside": inside}
    return {
        "start_spread_points": spreads,
        "line_1_starts_agree": bool(max(spreads.values()) <= AGREEMENT_POINTS),
        "objective_above_lowest_profile_point": objective_line,
        "line_2_objective": bool(max(objective_line.values()) <= tolerance),
        "near_minimum": near_minimum,
        "line_3_capacity_near_minimum": bool(all(all(v["inside"]) for v in near_minimum.values())),
    }


def main() -> None:
    """Run the requested configurations and write `work/profile.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    arguments = parser.parse_args()
    cache = common.load_cache()
    out = {name: run_config(cache=cache, config=CONFIGS[name]) for name in arguments.configs}
    common.write_json(name="profile.json", payload=out)
    for name, result in out.items():
        checks_ = result["checks"]
        print(
            name,
            "lines:",
            checks_["line_1_starts_agree"],
            checks_["line_2_objective"],
            checks_["line_3_capacity_near_minimum"],
            flush=True,
        )


if __name__ == "__main__":
    main()
