"""Experiment 267: the capacity of unmetered solar per season-year, against the register floor.

For each decision Primary the script fits the convex model of `studies.dp_gam` with an increment
penalty of 0.1 (experiment 129's arm 4, the card's fit) to the real reading, under two solar
shapes: the satellite-based shape (A0) and the recorded output of the nearest metered farm within
25 km of the Primary (A2). It reads the divisors from experiment 129's known-answer table (A0, A1)
and experiment 238's injection results (A2), prints the number of fitted months per season-year
(A3), and writes `work/capacity.parquet`. The plan is in
`docs/ml-experiments/00267-pv-capacity-early-years-and-register-floor/result.md`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00267-*/e00267_run.py --device cpu
"""

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

import e00267_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamFit,
    GamSpec,
    accumulate_gram,
    last_signal_season_year,
    penalty_matrix,
    prepare_fit,
    solve_bounded_quadratic,
    training_rows,
)
from studies.pv_dataset import pv_sites

CUT: Final[datetime] = datetime(2026, 8, 1, tzinfo=UTC)
"""The cut of the card's fit: experiment 129's maintainer-approved exception, data to 2026-07-31."""
N_BOOTSTRAP: Final[int] = 40
INCREMENT_SHARE: Final[float] = 0.1
FIRST_INCREMENT_RIDGE_SHARE: Final[float] = 1e-6
RANGE_LEVELS: Final[tuple[float, float]] = (0.1, 0.9)
MAX_REFERENCE_KM: Final[float] = 25.0
"""A reference farm must lie within this distance of the Primary (the brief's rule)."""
THIN_MONTHS: Final[int] = 6
MIN_YEAR_ROWS: Final[int] = 12000
"""A season-year counts as a data year with at least this many training half-hours."""
EXPERIMENT_129: Final = (
    common.ML_EXPERIMENTS_DATA_DIR / "00129-differentiable-physics-pv-capacity-known-answer-test"
)
EXPERIMENT_238: Final = (
    common.ML_EXPERIMENTS_DATA_DIR / "00238-pv-capacity-nearest-farm-sunshine-reference" / "work"
)


@dataclass(frozen=True)
class ShapeFit:
    """The capacity per season-year under one solar shape, with its bootstrap.

    Attributes:
        point: Season-year to capacity in points of effective capacity.
        boot: One such mapping per bootstrap resample.
        months: Season-year to the number of calendar months with training rows.
        rows: Season-year to the number of training half-hours.
    """

    point: dict[int, float]
    boot: list[dict[int, float]]
    months: dict[int, int]
    rows: dict[int, int]


def load_cache_to_july() -> FeatureCache:
    """Return the feature cache that holds data to 2026-08-01, as experiment 129 builds it."""
    work = EXPERIMENT_129 / "work"
    paths = default_input_paths()
    return load_or_build_feature_cache(
        paths=replace(paths, rows=work / "rows_to_july.parquet"),
        cache_dir=work / "cache_to_july",
    )


def month_block_weights(
    *, cache: FeatureCache, index: np.ndarray, n_resamples: int, seed: int
) -> np.ndarray:
    """Return month-block bootstrap weights, one row per resample, one column per training row.

    Each resample draws, within each (season-year, quarter of the year) stratum, as many months as
    the stratum has, with replacement, so that no season-year loses its summer.
    """
    month = cache.end_times[index].astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    month_number = months.astype(np.int64)
    stratum = ((month_number + 6) // 12) * 4 + (month_number % 12) // 3
    generator = np.random.default_rng(seed)
    counts = np.zeros((n_resamples, len(months)))
    for resample in range(n_resamples):
        for value in np.unique(stratum):
            members = np.flatnonzero(stratum == value)
            drawn = generator.choice(members, size=len(members), replace=True)
            counts[resample] += np.bincount(drawn, minlength=len(months))
    return counts[:, inverse]


def fit_with_increment_penalty(
    *, cache: FeatureCache, site: str, weights: np.ndarray | None, device: str
) -> GamFit:
    """Fit the convex model with a ridge on the capacity increments (experiment 129's arm 4)."""
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=CUT,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    gram, rhs = accumulate_gram(
        cache=cache,
        features=cache.sites[site],
        layout=layout,
        index=index,
        weights=weights,
        device=device,
        dtype=torch.float64,
        pinned=None,
    )
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    capacity = layout.slices["capacity"]
    diagonal = torch.diagonal(gram[0])[capacity].mean().item()
    ridge = np.full(capacity.stop - capacity.start, INCREMENT_SHARE * diagonal)
    ridge[0] *= FIRST_INCREMENT_RIDGE_SHARE
    matrix[capacity, capacity] = np.diag(ridge)
    batch = gram.shape[0]
    penalty = torch.as_tensor(matrix, dtype=torch.float64, device=device)[None].expand(
        batch, -1, -1
    )
    linear = torch.as_tensor(cost, dtype=torch.float64, device=device)[None].expand(batch, -1)
    theta, violation = solve_bounded_quadratic(
        matrix=gram + penalty,
        rhs=rhs - 0.5 * linear,
        constrained=torch.as_tensor(layout.constrained),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    return GamFit(
        layout=layout,
        theta=theta.cpu().numpy(),
        cut=CUT,
        n_rows=len(index),
        site=site,
        last_season_year=last_signal_season_year(
            cache=cache, site=site, index=index, cut=CUT, pinned=None
        ),
        violation=violation.cpu().numpy(),
        pinned=None,
    )


def fit_shape(*, cache: FeatureCache, site: str, shape: np.ndarray, device: str) -> ShapeFit:
    """Fit the real reading under one solar shape: the point fit and the bootstrap."""
    features = cache.sites[site]
    shaped = replace(cache, sites={**cache.sites, site: replace(features, pv_unit_shape=shape)})
    index = training_rows(
        features=shaped.sites[site], cache=shaped, cut=CUT, window=common.TRAINING_WINDOW
    )
    years = np.asarray(shaped.calendar.season_year)[index]
    month = shaped.end_times[index].astype("datetime64[M]")
    months = {int(y): len(np.unique(month[years == y])) for y in np.unique(years)}
    rows = {int(y): int((years == y).sum()) for y in np.unique(years)}
    point = fit_with_increment_penalty(cache=shaped, site=site, weights=None, device=device)
    weights = month_block_weights(cache=shaped, index=index, n_resamples=N_BOOTSTRAP, seed=129)
    boot = fit_with_increment_penalty(cache=shaped, site=site, weights=weights, device=device)
    return ShapeFit(
        point=point.capacity_points(),
        boot=[boot.capacity_points(member=m) for m in range(N_BOOTSTRAP)],
        months=months,
        rows=rows,
    )


def nearest_farm(*, cache: FeatureCache, site: str) -> tuple[str, float] | None:
    """Return the Primary's nearest metered farm and its distance in km, within the limit."""
    table = cache.sites[site].table
    farms = {row["site"]: row for row in pv_sites().to_dicts()}
    distances = [
        (
            common.distance_km(
                lat_a=table.latitude,
                lon_a=table.longitude,
                lat_b=farms[label]["latitude"],
                lon_b=farms[label]["longitude"],
            ),
            label,
        )
        for label in common.FARMS
    ]
    distance, label = min(distances)
    return (label, distance) if distance <= MAX_REFERENCE_KM else None


def reference_recoveries(*, site: str, reference: str) -> pl.DataFrame:
    """Return experiment 238's capacity ratios of the injected farms under the reference shape."""
    table = pl.read_parquet(EXPERIMENT_238 / f"result_{site}.parquet").filter(
        pl.col("arm") == "reference",
        pl.col("stat") == "capacity_ratio",
        pl.col("reference") == reference,
    )
    return table.select("farm", "b", ratio="value")


def main() -> None:
    """Write `work/capacity.parquet` and `work/known_answer_reference.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache_to_july()
    stored = pl.read_parquet(EXPERIMENT_129 / "work" / "real_arm4@0.1_to_july.parquet")
    rows: list[dict] = []
    check_rows: list[dict] = []
    for site in arguments.sites:
        scale = cache.sites[site].table.effective_capacity_mw / 100.0
        chosen = nearest_farm(cache=cache, site=site)
        print(site, "nearest farm", chosen, flush=True)
        cams_fit = fit_shape(
            cache=cache, site=site, shape=cache.sites[site].pv_unit_shape, device=arguments.device
        )
        card = stored.filter(pl.col("site") == site)
        for row in card.iter_rows(named=True):
            gap = abs(cams_fit.point[row["season_year"]] - row["raw_points"])
            if gap > 1e-6 * max(1.0, abs(row["raw_points"])):
                msg = f"{site} {row['season_year']}: refit {gap} from the stored raw estimate"
                raise ValueError(msg)
        ref_fit = None
        ratios = pl.DataFrame()
        if chosen is not None:
            reference, distance = chosen
            shape = common.reference_unit_shape(cache=cache, site=site, reference=reference)
            ref_fit = fit_shape(cache=cache, site=site, shape=shape, device=arguments.device)
            ratios = reference_recoveries(site=site, reference=reference)
            point_ratios = ratios.filter(pl.col("b") == 0)["ratio"].to_numpy()
            for k, farm in enumerate(ratios.filter(pl.col("b") == 0)["farm"].to_list()):
                others = np.delete(point_ratios, k)
                check_rows.append(
                    {
                        "site": site,
                        "farm": farm,
                        "corrected_error": float(point_ratios[k] / np.median(others) - 1.0),
                    }
                )
        for row in card.iter_rows(named=True):
            year = row["season_year"]
            line = {
                "site": site,
                "season_year": year,
                "months": cams_fit.months.get(year, 0),
                "rows": cams_fit.rows.get(year, 0),
                "a0_mw": row["estimate_mw"],
                "a0_low_mw": row["range_low_mw"],
                "a0_high_mw": row["range_high_mw"],
                "a0_divisor": row["divisor"],
                "raw_points": row["raw_points"],
                "scale": scale,
            }
            if ref_fit is not None and year in ref_fit.point:
                divisor = float(np.median(ratios.filter(pl.col("b") == 0)["ratio"].to_numpy()))
                all_ratios = ratios["ratio"].to_numpy()
                draws = (
                    np.array([c[year] for c in ref_fit.boot])[:, None] / all_ratios[None, :]
                ).ravel()
                low, high = np.quantile(draws, RANGE_LEVELS)
                line |= {
                    "reference": reference,
                    "reference_km": distance,
                    "ref_months": ref_fit.months.get(year, 0),
                    "ref_raw_points": ref_fit.point[year],
                    "a2_divisor": divisor,
                    "a2_mw": ref_fit.point[year] / divisor * scale,
                    "a2_low_mw": float(low) * scale,
                    "a2_high_mw": float(high) * scale,
                }
            rows.append(line)
        print(site, "done", flush=True)
    pl.DataFrame(rows).write_parquet(common.WORK_DIR / "capacity.parquet")
    pl.DataFrame(check_rows).write_parquet(common.WORK_DIR / "known_answer_reference.parquet")


if __name__ == "__main__":
    main()
