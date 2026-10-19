"""The blocking known-answer gate of experiment 105, before any scored fit.

Four tests, each fitted at the tuning cut (2024-07-01) so that nothing here sees the scored year:

1. **Synthetic pooled recovery.** Nine synthetic substations built from real calendar and weather
   tensors share four daily shapes and a heating threshold of 14 degrees. The pooled model must
   recover the threshold, every heating slope, every capacity, and every substation's daily
   profile. A ninth substation is onboarded from 4 June weeks, and must recover its heating slope
   where a per-site model on the same weeks cannot.
2. **Injected-farm recovery.** Scaled solar-farm output is added to one MW substation's demand at a
   time. The fitted capacity increment must come back, and no other substation or shared module may
   move. Farms carry only their anonymised letters, and only ratios are printed.
3. **Placebo twin.** The CAMS irradiance of every MW substation is replaced by its irradiance from
   the same calendar day a year earlier (a year later where the earlier year is missing), keeping
   the solar geometry. The fit must score worse on summer middays and its capacities must fall.
4. **Shuffled capacity.** Each MW substation's capacity prior is replaced by another substation's
   (a derangement). The fitted capacity must still lie closer to the substation's own vintage.

    uv run --no-sync python ml-experiments/00105-*/e00105_gate.py [--tests 1 2 3 4]
"""

import argparse
import time
from dataclasses import replace
from typing import Any, Final

import e00105_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache, default_input_paths, unit_pv_shape
from studies.dp_onboarding import (
    FitCopy,
    after_mask,
    fit_many,
    onboard_site,
    predict_pooled,
    select_rows,
    site_mask,
    with_targets,
)
from studies.dp_pooled import (
    DpConfig,
    PooledBatch,
    PooledDemandModel,
    fit_pooled,
)

SCORED: Final[tuple[str, ...]] = common.SCORED_SITES
INJECTED_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
FARM_OF_SITE: Final[dict[str, str]] = {"D10": "A", "D11": "B", "D14": "C", "D16": "D"}
INJECTED_POINTS: Final[float] = 10.0
"""The injected farm's output at its 99th percentile, in points of effective capacity."""
SUMMER_MONTHS: Final[range] = range(4, 10)
MIDDAY_HALF_HOURS: Final[range] = range(20, 29)
SYNTHETIC_THRESHOLD_C: Final[float] = 14.0
SYNTHETIC_SLOPE: Final[float] = 8.0
SYNTHETIC_NOISE: Final[float] = 1.0
GATE_ONBOARD_END: Final[np.datetime64] = np.datetime64("2023-08-12T00:00:00", "us")
"""The synthetic onboarding site's 4 weeks end here: midsummer, so its own weeks say little about
heating and a per-site fit cannot identify the heating slope."""
N_BOOTSTRAP: Final[int] = 4000


def config_p(*, k: int = 4, net: bool = True) -> DpConfig:
    """Return the pooled configuration P: variational site prior, optionally the network."""
    return DpConfig(
        variational=True,
        shared_shapes=k,
        interaction_hidden=common.INTERACTION if net else (),
    )


def inverse_softplus(value: np.ndarray | float) -> np.ndarray:
    """Return `x` with `softplus(x) = value`."""
    value = np.asarray(value, dtype=np.float64)
    return np.log(np.expm1(value))


def relabel_site(*, batch: PooledBatch, source: int, target: int) -> PooledBatch:
    """Return the rows of site `source` as site `target` of the same layout."""
    rows = select_rows(batch=batch, keep=batch.site == source)
    year = rows.season_slot % batch.n_season_years
    return replace(
        rows,
        site=torch.full_like(rows.site, target),
        season_slot=target * batch.n_season_years + year,
    )


def concat_batches(*, parts: list[PooledBatch], labels: tuple[str, ...]) -> PooledBatch:
    """Concatenate batches that share a layout, with the given site labels."""
    fields = {
        name: torch.cat([getattr(part, name) for part in parts])
        for name in (
            "site",
            "day_type",
            "half_hour",
            "band",
            "holiday",
            "tent",
            "harmonics",
            "smoothed_temperature",
            "lighting",
            "elevation",
            "pv_shape",
            "season_slot",
            "solar_gate",
            "is_apparent",
            "y",
            "quarter",
            "row",
        )
    }
    return replace(parts[0], site_labels=labels, **fields)


def single_site(*, batch: PooledBatch, site: int) -> PooledBatch:
    """Return one site's rows as a one-site batch."""
    rows = select_rows(batch=batch, keep=batch.site == site)
    year = rows.season_slot % batch.n_season_years
    return replace(
        rows,
        site=torch.zeros_like(rows.site),
        season_slot=year,
        site_labels=(batch.site_labels[site],),
    )


def month_block_interval(
    *, difference: np.ndarray, month: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Return the mean of `difference` and its 95% interval from resampling whole months."""
    months = np.unique(month)
    sums = np.array([difference[month == m].sum() for m in months])
    counts = np.array([(month == m).sum() for m in months])
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(months), size=(N_BOOTSTRAP, len(months)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def _fit(*, batch: PooledBatch, config: DpConfig, copy: FitCopy, steps: int | None = None):  # noqa: ANN202
    models, results = fit_many(
        batch=batch,
        copies=[copy],
        config=config,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
        steps=steps or common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
    )
    return models[0], results[0]


def build_synthetic(*, cache: FeatureCache) -> tuple[PooledBatch, PooledDemandModel, np.ndarray]:
    """Return the nine-site synthetic batch with its targets, the truth model, and true slopes."""
    real = ("D5", "D9", "D10", "D11", "D14", "D16")
    base = common.pooled_batch_for(cache=cache, sites=real, cut=common.TUNING_CUT, device="cpu")
    parts = [relabel_site(batch=base, source=i, target=i) for i in range(len(real))]
    parts += [relabel_site(batch=base, source=src, target=6 + i) for i, src in enumerate((0, 2, 3))]
    labels = (*real, "N1", "N2", "N3")
    n_sites = len(labels)
    batch = concat_batches(parts=parts, labels=labels)
    batch = replace(
        batch,
        solar_gate=torch.ones_like(batch.solar_gate),
        is_apparent=torch.zeros_like(batch.is_apparent),
    )
    generator = np.random.default_rng(0)
    torch.manual_seed(0)
    truth = PooledDemandModel(
        config=DpConfig(shared_shapes=4),
        n_sites=n_sites,
        n_season_years=batch.n_season_years,
        initial_level=torch.zeros(n_sites),
    )
    slopes = SYNTHETIC_SLOPE * (1.0 + 0.1 * generator.uniform(-1, 1, n_sites))
    first_capacity = generator.uniform(5.0, 25.0, n_sites)
    with torch.no_grad():
        truth.shapes.copy_(torch.as_tensor(2.0 * generator.normal(size=truth.shapes.shape)))
        truth.site_logits.copy_(torch.as_tensor(generator.normal(size=(n_sites, 4))))
        truth.level.copy_(torch.as_tensor(generator.uniform(40, 70, n_sites)))
        truth.threshold.fill_(SYNTHETIC_THRESHOLD_C)
        truth.slope_raw.copy_(torch.as_tensor(inverse_softplus(slopes))[:, None].expand(-1, 4))
        raw = np.full((n_sites, batch.n_season_years), float(inverse_softplus(0.05)))
        raw[:, int(batch.season_slot.min()) % batch.n_season_years] = inverse_softplus(
            first_capacity
        )
        truth.capacity_raw.copy_(torch.as_tensor(raw))
    truth.mark_trained_years(batch=batch)
    truth.eval()
    with torch.no_grad():
        clean = truth(batch)
    noise = torch.as_tensor(
        generator.normal(scale=SYNTHETIC_NOISE, size=clean.shape), dtype=clean.dtype
    )
    batch = with_targets(batch=batch, y=clean + noise)
    return batch, truth, slopes


def gate_1_synthetic(*, cache: FeatureCache) -> dict[str, Any]:
    """Return the synthetic pooled-recovery results."""
    batch, truth, slopes = build_synthetic(cache=cache)
    n_sites = len(batch.site_labels)
    truth_capacity = truth.capacities()[:, -1].detach().numpy()
    train_sites = n_sites - 1
    keep = batch.site < train_sites
    started = time.perf_counter()
    model, _ = _fit(batch=batch, config=config_p(net=False), copy=FitCopy(name="p", keep=keep))
    seconds = time.perf_counter() - started
    fitted_capacity = model.capacities()[:, -1].detach().numpy()
    fitted_slope = torch.nn.functional.softplus(model.slope_raw).mean(dim=1).detach().numpy()
    threshold = model.threshold.detach().numpy()
    profile = np.corrcoef
    shapes_fit = (torch.softmax(model.site_logits, 1) @ model.shapes).detach().numpy()
    shapes_true = (torch.softmax(truth.site_logits, 1) @ truth.shapes).detach().numpy()
    correlation = [float(profile(shapes_fit[i], shapes_true[i])[0, 1]) for i in range(train_sites)]
    out: dict[str, Any] = {
        "seconds": seconds,
        "noise_sd": float(torch.exp(model.log_noise).detach()),
        "fitted_threshold": threshold[:train_sites].tolist(),
        "fitted_slope": fitted_slope[:train_sites].tolist(),
        "threshold_error_c": [
            float(abs(t - SYNTHETIC_THRESHOLD_C)) for t in threshold[:train_sites]
        ],
        "slope_relative_error": [
            float(abs(fitted_slope[i] / slopes[i] - 1)) for i in range(train_sites)
        ],
        "capacity_relative_error": [
            float(abs(fitted_capacity[i] / truth_capacity[i] - 1)) for i in range(train_sites)
        ],
        "profile_correlation": correlation,
    }
    start = GATE_ONBOARD_END - np.timedelta64(28, "D")
    onboard_mask = (
        site_mask(batch=batch, label="N3")
        & after_mask(batch=batch, cache=cache, after=start)
        & ~after_mask(batch=batch, cache=cache, after=GATE_ONBOARD_END)
    )
    own = select_rows(batch=batch, keep=onboard_mask)
    onboard_site(
        model=model,
        batch=own,
        label="N3",
        cache=cache,
        history_end=GATE_ONBOARD_END,
        forecast_season_year=2023,
        steps=300,
    )
    onboarded_slope = float(torch.nn.functional.softplus(model.slope_raw[-1]).mean().detach())
    per_site = PooledDemandModel(
        config=DpConfig(shared_shapes=4),
        n_sites=1,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((1,), 50.0),
    )
    fit_pooled(model=per_site, batch=single_site(batch=own, site=n_sites - 1), steps=600)
    per_site_slope = float(torch.nn.functional.softplus(per_site.slope_raw[0]).mean().detach())
    truth_slope = float(slopes[-1])
    out["onboarded_slope_error"] = abs(onboarded_slope / truth_slope - 1)
    out["per_site_4w_slope_error"] = abs(per_site_slope / truth_slope - 1)
    out["passed"] = bool(
        max(out["threshold_error_c"]) < 1.0
        and max(out["slope_relative_error"]) < 0.15
        and max(out["capacity_relative_error"]) < 0.15
        and min(out["profile_correlation"]) >= 0.9
        and out["onboarded_slope_error"] < 0.25
        and out["per_site_4w_slope_error"] > 0.5
    )
    return out


def farm_series(*, farm: str, end_times: np.ndarray) -> np.ndarray:
    """Return a farm's normalised output at the rows' end times, 0 where the farm has no reading."""
    table = (
        pl.read_parquet(common.FARMS_FILE)
        .filter(pl.col("farm") == farm)
        .select("time", "farm_norm")
        .drop_nulls()
        .sort("time")
    )
    times = table["time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    values = table["farm_norm"].to_numpy().astype(np.float64)
    position = np.searchsorted(times, end_times)
    position = np.clip(position, 0, len(times) - 1)
    found = times[position] == end_times
    return np.where(found, values[position], 0.0)


def midday_summer(*, batch: PooledBatch, cache: FeatureCache) -> torch.Tensor:
    """Return a boolean tensor marking summer middays (10:00 to 14:00 local, April to September)."""
    end = cache.end_times[batch.row.cpu().numpy()]
    month = end.astype("datetime64[M]").astype(int) % 12 + 1
    in_summer = np.isin(month, list(SUMMER_MONTHS))
    in_midday = np.isin(batch.half_hour.cpu().numpy(), list(MIDDAY_HALF_HOURS))
    return torch.as_tensor(in_summer & in_midday, device=batch.y.device)


def shared_midday_output(
    *, model: PooledDemandModel, batch: PooledBatch, mask: torch.Tensor
) -> float:
    """Return the mean centred shared-module output over the masked rows, in points."""
    model.eval()
    with torch.no_grad():
        calendar, _ = model.calendar_terms(batch=batch)
        centred = calendar - model.centre_means[batch.season_slot]
    return float(centred[mask].mean())


def p99_unit_shape(*, batch: PooledBatch, site: int) -> float:
    """Return the 99th percentile of a site's daylight PV shape."""
    shape = batch.pv_shape[batch.site == site]
    daylight = shape[shape > 0.01]
    return float(torch.quantile(daylight.float().cpu(), 0.99))


def gate_2_injected(*, cache: FeatureCache, context: dict[str, Any]) -> dict[str, Any]:
    """Return the injected-farm recovery results."""
    batch = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.TUNING_CUT)
    context["train_batch"] = batch
    config = config_p()
    copies = [FitCopy(name="base")]
    coverage = {}
    for site in INJECTED_SITES:
        index = batch.site_labels.index(site)
        mask = batch.site == index
        end = cache.end_times[batch.row[mask].cpu().numpy()]
        farm = farm_series(farm=FARM_OF_SITE[site], end_times=end)
        coverage[site] = float((farm > 0).mean())
        injection = torch.zeros_like(batch.y)
        injection[mask] = torch.as_tensor(
            INJECTED_POINTS * farm, dtype=batch.y.dtype, device=batch.y.device
        )
        copies.append(FitCopy(name=f"inject {site}", y=batch.y - injection))
    models, results = fit_many(
        batch=batch,
        copies=copies,
        config=config,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
        steps=common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
    )
    base, injected = models[0], models[1:]
    context["base_model"] = base
    base_cap = base.capacities()[:, -1].detach().cpu().numpy()
    midday = midday_summer(batch=batch, cache=cache)
    base_shared = shared_midday_output(model=base, batch=batch, mask=midday)
    ratios, others, shared_moves = {}, {}, {}
    solar_sites = [s for s in common.MW_POOL if s in batch.site_labels]
    for site, model in zip(INJECTED_SITES, injected, strict=True):
        index = batch.site_labels.index(site)
        cap = model.capacities()[:, -1].detach().cpu().numpy()
        recovered = (cap[index] - base_cap[index]) * p99_unit_shape(batch=batch, site=index)
        ratios[site] = float(recovered / INJECTED_POINTS)
        others[site] = float(
            max(
                abs(cap[batch.site_labels.index(o)] / base_cap[batch.site_labels.index(o)] - 1)
                for o in solar_sites
                if o != site
            )
        )
        shared_moves[site] = abs(
            shared_midday_output(model=model, batch=batch, mask=midday) - base_shared
        )
    median = float(np.median(list(ratios.values())))
    return {
        "recovery_ratio": ratios,
        "median_ratio": median,
        "largest_other_capacity_change": others,
        "shared_midday_change_points": shared_moves,
        "farm_coverage_of_site_rows": coverage,
        "seconds": [r.seconds for r in results],
        "peak_gpu_gb": max(r.peak_memory_bytes for r in results) / 1e9,
        "passed": bool(
            0.75 <= median <= 1.25
            and max(others.values()) < 0.05
            and max(shared_moves.values()) < 0.1
        ),
    }


def placebo_cache(*, cache: FeatureCache) -> FeatureCache:
    """Return a cache whose CAMS irradiance is the same site's from a year earlier (or later)."""
    paths = default_input_paths()
    cams = pl.read_parquet(paths.cams, columns=["site", "time", "ghi_w_m2"])
    grid = cache.times_series
    sites = dict(cache.sites)
    for label in common.MW_POOL:
        features = cache.sites[label]
        own = cams.filter(pl.col("site") == label)
        earlier = own.with_columns(time=pl.col("time").dt.offset_by("1y"))
        later = own.with_columns(time=pl.col("time").dt.offset_by("-1y"))
        shifted = (
            pl.concat([earlier, later.join(earlier.select("time"), on="time", how="anti")])
            .unique(subset=["site", "time"], keep="first")
            .sort("time")
        )
        sites[label] = replace(
            features, pv_unit_shape=unit_pv_shape(cams=shifted, table=features.table, grid=grid)
        )
    return replace(cache, sites=sites)


def scored_rows(*, model: PooledDemandModel, batch: PooledBatch) -> np.ndarray:
    """Return the absolute error of the central forecast on every row of `batch`."""
    return np.abs(batch.y.cpu().numpy() - predict_pooled(model=model, batch=batch))


def gate_3_placebo(*, cache: FeatureCache, context: dict[str, Any]) -> dict[str, Any]:
    """Return the placebo-twin results."""
    cut, end = common.TUNING_CUT, common.TUNING_END
    twin = placebo_cache(cache=cache)
    real_train = context["train_batch"]
    twin_train = common.pooled_batch_for(cache=twin, sites=common.POOL, cut=cut)
    twin_model, _ = _fit(batch=twin_train, config=config_p(), copy=FitCopy(name="placebo"))
    real_model = context["base_model"]
    real_score = common.scoring_batch_for(cache=cache, sites=SCORED, after=cut, through=end)
    twin_score = common.scoring_batch_for(cache=twin, sites=SCORED, after=cut, through=end)
    common_rows = np.intersect1d(real_score.row.cpu().numpy(), twin_score.row.cpu().numpy())
    errors = {}
    for name, model, batch in (
        ("real", real_model, real_score),
        ("placebo", twin_model, twin_score),
    ):
        error = scored_rows(model=model, batch=batch)
        key = batch.site.cpu().numpy() * 10**7 + batch.row.cpu().numpy()
        errors[name] = (key, error, midday_summer(batch=batch, cache=cache).cpu().numpy())
    shared_keys = np.intersect1d(errors["real"][0], errors["placebo"][0])
    pick = {}
    for name, (key_all, error_all, midday_all) in errors.items():
        order = np.argsort(key_all)
        keep = np.isin(key_all[order], shared_keys) & midday_all[order]
        pick[name] = (key_all[order][keep], error_all[order][keep])
    difference = pick["placebo"][1] - pick["real"][1]
    rows = pick["real"][0] % 10**7
    month = (cache.end_times[rows].astype("datetime64[M]")).astype(int)
    mean, low, high = month_block_interval(difference=difference, month=month)
    index = [real_train.site_labels.index(s) for s in SCORED]
    real_cap = real_model.capacities()[index, -1].detach().cpu().numpy()
    twin_cap = twin_model.capacities()[index, -1].detach().cpu().numpy()
    falls = int((twin_cap < real_cap).sum())
    del common_rows
    return {
        "placebo_minus_real_summer_midday_mae": {"mean": mean, "low": low, "high": high},
        "capacity_falls_at_sites": falls,
        "capacity_ratio_by_site": dict(zip(SCORED, (twin_cap / real_cap).tolist(), strict=True)),
        "passed": bool(low > 0.0 and falls >= 4),
    }


def vintage_points(*, cache: FeatureCache) -> dict[str, float]:
    """Return each scored site's `frozen` vintage at the tuning cut, in points of capacity."""
    series = pl.read_parquet(common.SERIES_TABLE).select("time_series_id", "site")
    vintages = (
        pl.read_parquet(common.VINTAGES)
        .filter(pl.col("variant") == "frozen")
        .filter(pl.col("origin_monday") <= common.TUNING_CUT.date())
        .filter(pl.col("week_start") <= common.TUNING_CUT.date())
        .join(series, on="time_series_id")
    )
    out = {}
    for site in SCORED:
        rows = vintages.filter(pl.col("site") == site)
        if rows.is_empty():
            continue
        latest = rows.filter(pl.col("origin_monday") == rows["origin_monday"].max())
        mw = float(latest.sort("week_start")["capacity_mw"][-1])
        out[site] = 100.0 * mw / common.effective_capacity_mw(cache=cache, site=site)
    return out


def derangement(*, n: int) -> list[int]:
    """Return a permutation of `range(n)` that moves every element (a cyclic shift)."""
    return [(i + 1) % n for i in range(n)]


def gate_4_shuffled(*, cache: FeatureCache, context: dict[str, Any]) -> dict[str, Any]:
    """Return the shuffled-capacity results."""
    batch = context["train_batch"]
    own = vintage_points(cache=cache)
    sites = [s for s in SCORED if s in own]
    moved = derangement(n=len(sites))
    prior = torch.full((len(batch.site_labels),), float("nan"))
    shuffled = {}
    for position, site in enumerate(sites):
        shuffled[site] = own[sites[moved[position]]]
        prior[batch.site_labels.index(site)] = float(inverse_softplus(shuffled[site]))
    model, _ = _fit(
        batch=batch, config=config_p(), copy=FitCopy(name="shuffled", capacity_prior=prior)
    )
    capacity = model.capacities()[:, -1].detach().cpu().numpy()
    closer, separated, detail = 0, 0, {}
    for site in sites:
        fitted = float(capacity[batch.site_labels.index(site)])
        gap = abs(own[site] - shuffled[site]) / own[site]
        entry = {"fitted": fitted, "own_vintage": own[site], "shuffled_prior": shuffled[site]}
        if gap >= 0.1:
            separated += 1
            entry["closer_to_own"] = bool(abs(fitted - own[site]) < abs(fitted - shuffled[site]))
            closer += int(entry["closer_to_own"])
        detail[site] = entry
    needed = int(np.ceil(2 * separated / 3))
    return {
        "sites": detail,
        "closer_to_own_at": closer,
        "of_separated_sites": separated,
        "needed": needed,
        "passed": bool(separated >= 1 and closer >= needed),
    }


def main() -> None:
    """Run the requested gate tests and write `work/gate.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+", type=int, default=[1, 2, 3, 4])
    arguments = parser.parse_args()
    torch.set_num_threads(8)
    cache = common.load_cache()
    context: dict[str, Any] = {}
    report: dict[str, Any] = {}
    steps = {
        1: lambda: gate_1_synthetic(cache=cache),
        2: lambda: gate_2_injected(cache=cache, context=context),
        3: lambda: gate_3_placebo(cache=cache, context=context),
        4: lambda: gate_4_shuffled(cache=cache, context=context),
    }
    needs_two = {3, 4}
    order = sorted(arguments.tests)
    if needs_two & set(order) and 2 not in order:
        order = [2, *order]
    for number in order:
        started = time.perf_counter()
        report[f"test_{number}"] = steps[number]()
        report[f"test_{number}"]["wall_seconds"] = time.perf_counter() - started
        verdict = report[f"test_{number}"]["passed"]
        print(
            f"gate test {number}: passed={verdict} ({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
        common.write_json(name="gate.json", payload=report)
    print(report)


if __name__ == "__main__":
    main()
