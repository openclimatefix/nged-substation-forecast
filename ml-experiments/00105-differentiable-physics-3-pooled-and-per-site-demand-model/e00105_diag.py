"""Diagnostics for the gate's capacity findings, and the labelled model variants.

The pre-registered model P is unchanged. This script fits post hoc variants at screening tier on
the tuning cut and compares them on capacity and on the injected-farm leak test:

- `base`: P, fitted again, with the plan's shuffled-capacity test (swap fitted capacity paths in
  MW among the six MW substations, rescore without refitting).
- `seed1`: P with another seed, to measure the noise floor of any two fits.
- `nonet`: P with the interaction network off.
- `a`: temperature hidden from the interaction network by day.
- `c`: capacity started at Part 2's convex capacities.
- `ac`: both.
- `vt`: a tight capacity prior centred on the vintage (past data only), scale fixed at 5.

Each variant's capacities per season-year are written beside Part 2's convex model and the
vintages' season-year means.

    uv run --no-sync python ml-experiments/00105-*/e00105_diag.py [--variants base seed1 ...]
"""

import argparse
import json
from datetime import date
from typing import Any, Final

import e00105_common as common
import e00105_gate as gate
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam
from studies.dp_onboarding import FitCopy, fit_many
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel

VARIANTS: Final[tuple[str, ...]] = ("base", "seed1", "nonet", "a", "c", "ac", "vt")
VINTAGE_PRIOR_SCALE: Final[float] = 5.0


def convex_capacities(*, cache: FeatureCache) -> dict[str, dict[int, float]]:
    """Return Part 2's convex model (all years, no drift) capacities per season-year, in points."""
    tuned_path = common.PART_2_DIR / "work" / "tuned.json"
    tuned = json.loads(tuned_path.read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    out = {}
    for site in gate.SCORED:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=common.TUNING_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
        )
        out[site] = {int(year): float(value) for year, value in fit.capacity_points().items()}
    return out


def vintage_season_means(*, cache: FeatureCache) -> dict[str, dict[int, float]]:
    """Return the vintages' mean capacity per season-year (July to June), in points."""
    series = pl.read_parquet(common.SERIES_TABLE).select("time_series_id", "site")
    vintages = (
        pl.read_parquet(common.VINTAGES)
        .filter(pl.col("variant") == "frozen")
        .filter(pl.col("origin_monday") <= common.TUNING_CUT.date())
        .filter(pl.col("week_start") <= common.TUNING_CUT.date())
        .join(series, on="time_series_id")
    )
    out: dict[str, dict[int, float]] = {}
    for site in gate.SCORED:
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


def capacity_init_from_convex(
    *, batch: PooledBatch, convex: dict[str, dict[int, float]]
) -> torch.Tensor:
    """Return `capacity_raw` starting values that reproduce Part 2's convex capacities."""
    raw = np.full((len(batch.site_labels), batch.n_season_years), -3.0)
    for site, by_year in convex.items():
        row = batch.site_labels.index(site)
        previous = 0.0
        for offset in range(batch.n_season_years):
            year = batch.first_season_year + offset
            current = by_year.get(year, previous)
            increment = max(current - previous, 0.05)
            raw[row, offset] = float(gate.inverse_softplus(increment))
            previous = max(current, previous)
    return torch.as_tensor(raw, dtype=torch.float32)


def capacity_rows(
    *, model: PooledDemandModel, batch: PooledBatch, cache: FeatureCache, name: str
) -> list[dict[str, Any]]:
    """Return a model's capacity per scored substation and season-year, in points and MW."""
    capacity = model.capacities().detach().cpu().numpy()
    rows = []
    for site in gate.SCORED:
        index = batch.site_labels.index(site)
        effective = common.effective_capacity_mw(cache=cache, site=site)
        for offset in range(batch.n_season_years):
            year = batch.first_season_year + offset
            if float(model.trained[index, offset]) > 0:
                rows.append(
                    {
                        "variant": name,
                        "site": site,
                        "season_year": year,
                        "capacity_points": float(capacity[index, offset]),
                        "capacity_mw": float(capacity[index, offset] * effective / 100.0),
                    }
                )
    return rows


def vintage_prior(*, batch: PooledBatch, cache: FeatureCache) -> torch.Tensor:
    """Return the tight vintage prior's centres (before the softplus), NaN for none."""
    points = gate.vintage_points(cache=cache)
    prior = torch.full((len(batch.site_labels),), float("nan"))
    for site, value in points.items():
        prior[batch.site_labels.index(site)] = float(gate.inverse_softplus(value))
    return prior


def run_variant(
    *, name: str, cache: FeatureCache, convex: dict, leak_test: bool, report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fit one variant's base model, run the shuffled-capacity test, and optionally test 2."""
    config = {
        "base": gate.config_p(),
        "seed1": gate.config_p(),
        "nonet": gate.config_p(net=False),
        "a": DpConfig(
            variational=True,
            shared_shapes=4,
            interaction_hidden=common.INTERACTION,
            network_temperature_dark_only=True,
        ),
        "c": gate.config_p(),
        "ac": DpConfig(
            variational=True,
            shared_shapes=4,
            interaction_hidden=common.INTERACTION,
            network_temperature_dark_only=True,
        ),
        "vt": gate.config_p(),
    }[name]
    batch = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.TUNING_CUT)
    init = capacity_init_from_convex(batch=batch, convex=convex) if name in {"c", "ac"} else None
    prior = vintage_prior(batch=batch, cache=cache) if name == "vt" else None
    fixed = {"capacity_first": VINTAGE_PRIOR_SCALE} if name == "vt" else None
    models, results = fit_many(
        batch=batch,
        copies=[FitCopy(name=name, capacity_init=init, capacity_prior=prior, fixed_scales=fixed)],
        config=config,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
        steps=common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
        seed=1 if name == "seed1" else 0,
    )
    model = models[0]
    context = {"train_batch": batch, "base_model": model}
    swap = gate.gate_4_shuffled(cache=cache, context=context)
    midday = gate.midday_summer(batch=batch, cache=cache)
    report[name] = {
        "fit_seconds": results[0].seconds,
        "shuffled_capacity_test": swap,
        "module_midday_output": gate.module_midday_output(model=model, batch=batch, mask=midday),
    }
    if leak_test:
        leak_context: dict[str, Any] = {}
        report[name]["leak_test"] = gate.gate_2_injected(
            cache=cache,
            context=leak_context,
            config=config,
            capacity_init=init,
            prior=prior,
            capacity_scale=VINTAGE_PRIOR_SCALE if name == "vt" else None,
        )
    return capacity_rows(model=model, batch=batch, cache=cache, name=name)


MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3)


def capacity_profile(*, cache: FeatureCache, convex: dict[str, dict[int, float]]) -> dict[str, Any]:
    """Fit P with every scored substation's capacity fixed at a multiple of the convex capacity.

    For each multiplier the capacity paths are Part 2's convex capacities (all years, no drift,
    fitted before the tuning cut) times the multiplier, and everything else is refitted. The
    result records the training loss in nats (the loss per row times the rows) and each
    substation's summer-midday mean absolute error on the tuning year. A flat loss within a few
    nats means the pooled model does not identify the capacity; a clear minimum means a fit of
    the capacity itself is slow to converge.
    """
    batch = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.TUNING_CUT)
    score = gate.arms.remap_sites(
        batch=common.scoring_batch_for(
            cache=cache, sites=gate.SCORED, after=common.TUNING_CUT, through=common.TUNING_END
        ),
        model_sites=common.POOL,
    )
    sunny = gate.midday_summer(batch=score, cache=cache).cpu().numpy()
    truth = score.y.cpu().numpy()
    site_of_row = score.site.cpu().numpy()
    base_path = np.full((len(batch.site_labels), batch.n_season_years), np.nan)
    for site, by_year in convex.items():
        row = batch.site_labels.index(site)
        previous = 0.0
        for offset in range(batch.n_season_years):
            previous = by_year.get(batch.first_season_year + offset, previous)
            base_path[row, offset] = previous
    out: dict[str, Any] = {}
    for multiplier in MULTIPLIERS:
        override = torch.as_tensor(base_path * multiplier, dtype=torch.float32)
        models, results = fit_many(
            batch=batch,
            copies=[FitCopy(name=f"x{multiplier}", capacity_override=override)],
            config=gate.config_p(),
            initial_level=torch.full((len(batch.site_labels),), 50.0),
            steps=common.FIT_STEPS,
            learning_rate=common.FIT_RATE,
            final_rate_fraction=common.FINAL_RATE_FRACTION,
        )
        model = models[0]
        predicted = gate.predict_pooled(model=model, batch=score)
        error = np.abs(truth - predicted)
        out[str(multiplier)] = {
            "loss_nats": float(np.mean(results[0].losses[-20:])) * batch.n_rows,
            "summer_midday_mae": {
                site: float(error[(site_of_row == batch.site_labels.index(site)) & sunny].mean())
                for site in gate.SCORED
            },
        }
        print(f"profile x{multiplier}: {out[str(multiplier)]['loss_nats']:.0f} nats", flush=True)
    return out


def main() -> None:
    """Run the requested variants and write `work/diagnostics.json` and the capacity table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--leak-test", nargs="*", default=[])
    parser.add_argument("--profile", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    cache = common.load_cache()
    convex = convex_capacities(cache=cache)
    vintages = vintage_season_means(cache=cache)
    path = common.WORK_DIR / "diagnostics.json"
    report: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    report |= {
        "convex_capacity_points": convex,
        "vintage_season_mean_points": vintages,
        "cut": date.isoformat(common.TUNING_CUT.date()),
    }
    table_path = common.WORK_DIR / "diagnostic_capacities.parquet"
    table: list[dict[str, Any]] = (
        pl.read_parquet(table_path).to_dicts() if table_path.exists() else []
    )
    if arguments.profile:
        common.write_json(
            name="capacity_profile.json", payload=capacity_profile(cache=cache, convex=convex)
        )
        return
    for name in arguments.variants:
        table = [row for row in table if row["variant"] != name]
        table += run_variant(
            name=name,
            cache=cache,
            convex=convex,
            leak_test=name in arguments.leak_test,
            report=report,
        )
        common.write_json(name="diagnostics.json", payload=report)
        pl.DataFrame(table).write_parquet(common.WORK_DIR / "diagnostic_capacities.parquet")
        print(f"variant {name}: done", flush=True)


if __name__ == "__main__":
    main()
