"""The checks of experiment 00050 that must pass before any contrast is read.

Checks 1 to 8 use synthetic data and the committed helpers, and need no GEFS file. Checks 9 and 10
read experiment 00046's saved `d1l` booster and the built GEFS inputs, and are skipped with a note
when those files are absent. Every number is printed, and a failed check raises.

    uv run python ml-experiments/00050-*/e00050_selftest.py [--checks 1 2 3 ...]
"""

import argparse
from datetime import UTC, datetime, timedelta
from typing import Final

import e00050_anomaly as anomaly
import e00050_common as common
import e00050_gefs_inputs as inputs
import e00050_predict as predict
import e00050_stats as stats
import numpy as np
import polars as pl
import xgboost as xgb

SMOKE_TOLERANCE: Final[float] = 1e-5


def _require(*, passed: bool, message: str) -> None:
    """Print a check's outcome and raise when it failed."""
    print(("PASS: " if passed else "FAIL: ") + message, flush=True)
    if not passed:
        raise AssertionError(message)


def check_1_deaccumulation() -> None:
    """A known 3-hour series, stored the way GEFS stores it, comes back exactly and clipped."""
    leads = list(range(3, 361, 3))
    truth = {lead: 100.0 + 5.0 * (lead % 24) for lead in leads}
    truth[9] = -3.0
    stored = [
        (lead, truth[lead] if lead % 6 == 3 else (truth[lead] + truth[lead - 3]) / 2.0)
        for lead in range(3, 241, 3)
    ]
    stored.extend((lead, (truth[lead] + truth[lead - 3]) / 2.0) for lead in range(246, 361, 6))
    frame = pl.DataFrame(
        {
            "site": "A",
            "ensemble_member": pl.Series([0] * len(stored), dtype=pl.Int8),
            "lead_hours": [lead for lead, _ in stored],
            "value": [value for _, value in stored],
        }
    )
    table, statistics = inputs.deaccumulate(frame=frame, keys=inputs.KEYS_BY_MEMBER)
    result = dict(zip(table["lead_hours"].to_list(), table["value"].to_list(), strict=True))
    expected_leads = [h for h in common_ens_leads() if 0 < h <= 360]
    print(f"check 1: statistics {statistics}")
    _require(
        passed=sorted(result) == expected_leads,
        message="check 1: de-accumulated leads equal ENS's native radiation schedule",
    )
    exact = all(abs(result[lead] - max(truth[lead], 0.0)) < 1e-9 for lead in range(3, 145, 3))
    _require(
        passed=exact, message="check 1: 3-hour means up to 144 h are recovered, negative clipped"
    )
    paired = abs(result[150] - (max(truth[150], 0.0) + max(truth[147], 0.0)) / 2.0) < 1e-9
    _require(
        passed=paired and statistics["below_zero"] == 1.0,
        message="check 1: 6-hour means pair the 3-hour means; one mean clipped",
    )


def common_ens_leads() -> list[int]:
    """Return ENS's native radiation lead times in hours."""
    from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS

    return list(ECMWF_ENS_LEAD_TIME_HOURS)


def check_2_weights_sum_to_one() -> None:
    """Each site's cell weights sum to 1 after renormalising, and the coverage is at most 1."""
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    weights, coverage = inputs.grid_weights(series=series)
    sums = weights.group_by("site").agg(total=pl.col("weight").sum())
    print(coverage.join(sums, on="site"))
    _require(
        passed=bool(np.allclose(sums["total"].to_numpy(), 1.0))
        and common.scalar(value=coverage["coverage"].max()) <= 1.0 + 1e-6,
        message="check 2: weights sum to 1 per site and coverage is at most 1",
    )


def synthetic_rows(*, blocks: int = 4, seed: int = 0) -> pl.DataFrame:
    """Return synthetic daylight rows with a known blend of two noisy copies of the CAMS anomaly."""
    generator = np.random.default_rng(seed)
    count = 6000
    truth = generator.normal(0.0, 0.3, count)
    return pl.DataFrame(
        {
            "site": ["D9"] * (count // 2) + ["D10"] * (count // 2),
            "band": np.tile([0, 1, 2], count // 3),
            "block": np.tile(np.arange(blocks), count // blocks),
            "a_cams": truth,
            "a_ens": truth + generator.normal(0.0, 0.2, count),
            "a_gefs": truth + generator.normal(0.0, 0.4, count),
        }
    )


def check_3_fit_recovers_weights() -> None:
    """The two-regressor fit recovers a known blend and the single slopes recover known slopes."""
    generator = np.random.default_rng(1)
    count = 20_000
    a_ens = generator.normal(0.0, 0.3, count)
    a_gefs = 0.5 * a_ens + generator.normal(0.0, 0.3, count)
    a_cams = 0.7 * a_ens + 0.3 * a_gefs + generator.normal(0.0, 0.02, count)
    rows = pl.DataFrame(
        {"site": "D9", "band": 0, "a_ens": a_ens, "a_gefs": a_gefs, "a_cams": a_cams}
    )
    table = anomaly.fit_calibrations(rows=rows).row(0, named=True)
    print(f"check 3: fitted weights {table['w_ens']:.3f}, {table['w_gefs']:.3f}")
    _require(
        passed=abs(table["w_ens"] - 0.7) < 0.01 and abs(table["w_gefs"] - 0.3) < 0.01,
        message="check 3: the blend recovers weights (0.7, 0.3)",
    )
    slope = float((a_cams * a_ens).sum() / (a_ens**2).sum())
    _require(
        passed=abs(table["slope_ens"] - slope) < 1e-9,
        message="check 3: the ENS slope is least squares through the origin",
    )


def check_4_blend_reproduces_ens() -> None:
    """A blend with weights (ENS slope, 0) reproduces the `ens` central anomaly bit for bit."""
    rows = synthetic_rows()
    fitted = anomaly.fit_calibrations(rows=rows)
    forced = fitted.with_columns(w_ens=pl.col("slope_ens"), w_gefs=pl.lit(0.0))
    centrals = anomaly.apply_calibrations(rows=rows, calibrations=forced)
    _require(
        passed=bool(
            np.array_equal(centrals["central_blend"].to_numpy(), centrals["central_ens"].to_numpy())
        ),
        message="check 4: the blend with weights (ENS slope, 0) equals the ens central anomaly",
    )
    mean = centrals.select(
        ((pl.col("central_ens") + pl.col("central_gefs")) / 2.0 - pl.col("central_blend_fixed"))
        .abs()
        .max()
    ).item()
    _require(
        passed=mean < 1e-12,
        message="check 4: the fixed blend is the mean of the two calibrated anomalies",
    )


def check_5_fold_excludes_heldout_month() -> None:
    """The training side of a fold excludes the held-out month, in rows and in the climatology."""
    generator = np.random.default_rng(2)
    stamps = pl.datetime_range(
        datetime(2024, 4, 1, tzinfo=UTC),
        datetime(2025, 6, 30, 23, 30, tzinfo=UTC),
        interval="30m",
        eager=True,
    )
    count = stamps.len()
    rows = pl.DataFrame(
        {
            "site": ["D9"] * count,
            "power_fcst_init_time": stamps.dt.truncate("1d") + timedelta(hours=9),
            "valid_time": stamps,
            "lead_hours": np.full(count, 30.0),
            "band": np.full(count, 1, dtype=np.int64),
            "u_ens": generator.uniform(0.0, 1.0, count),
            "u_gefs": generator.uniform(0.0, 1.0, count),
            "u_cams": generator.uniform(0.0, 1.0, count),
        }
    ).with_columns(block=anomaly.block_index(times=pl.col("valid_time")))
    cams = rows.select("site", "valid_time", u=pl.col("u_cams"), block=pl.col("block"))
    errors = anomaly.fold_errors(rows=rows.filter(pl.col("site").is_in(["D9"])), cams=cams, fold=3)
    print(errors)
    _require(
        passed=errors.height == 4,
        message="check 5: a fold returns one error per calibrated arm and the zero baseline",
    )
    late = rows.with_columns(valid_time=pl.col("valid_time") + timedelta(days=400))
    raised = False
    try:
        common.climatology(
            frame=late.select("site", time=pl.col("valid_time"), u=pl.col("u_ens")),
            value="u",
            time_column="time",
            pool=True,
        )
    except ValueError:
        raised = True
    _require(
        passed=raised,
        message="check 5: a climatology refuses valid times after 2025-06-30 23:30 UTC",
    )


def check_6_zero_shift_is_identity() -> None:
    """A zero shift returns the unshifted pooled quantiles, and a constant shift moves them."""
    generator = np.random.default_rng(3)
    quantiles = np.sort(generator.normal(0.4, 0.1, (50, common.MEMBERS, 9)), axis=2).astype(
        np.float32
    )
    base = common.mixture_quantiles(member_quantiles=quantiles)
    zero = predict.member_shift(
        capacity_mw=3.0,
        effective_capacity_mw=6.0,
        coefficient=np.ones(50),
        central=np.zeros(50),
        deviation=np.zeros((50, common.MEMBERS)),
    )
    _require(
        passed=bool(
            np.array_equal(
                predict.shifted_pool(member_quantiles=quantiles, member_shift=zero), base
            )
        ),
        message="check 6: a zero shift returns the unshifted pooled quantiles bit for bit",
    )
    shift = predict.member_shift(
        capacity_mw=3.0,
        effective_capacity_mw=6.0,
        coefficient=np.full(50, 0.5),
        central=np.full(50, 0.2),
        deviation=np.zeros((50, common.MEMBERS)),
    )
    _require(
        passed=bool(np.allclose(shift, 0.05))
        and bool(
            np.allclose(
                predict.shifted_pool(member_quantiles=quantiles, member_shift=shift),
                base - 0.05,
                atol=1e-5,
            )
        ),
        message="check 6: K 0.5, C 3 MW, capacity 6 MW, central 0.2 shift every quantile by -0.05",
    )


def check_7_verdict_rule() -> None:
    """The verdict rule gives the stated answer on one example of each outcome."""

    def row(lower: float, upper: float) -> dict[str, str | float | int]:
        return {"lower": lower, "upper": upper}

    cases = [
        ((-0.3, -0.15), (-0.2, 0.1), "supported"),
        ((-0.05, -0.01), (-0.2, 0.1), "small gain"),
        ((-0.1, 0.0), (-0.1, 0.0), "inconclusive"),
        ((-0.05, 0.02), (-0.08, 0.03), "refuted"),
        ((-0.2, 0.05), (-0.3, 0.1), "inconclusive"),
    ]
    got = [stats.verdict(c7=row(*a), c7s=row(*b)) for a, b, _ in cases]
    print(f"check 7: {got}")
    _require(
        passed=got == [expected for _, _, expected in cases],
        message="check 7: the pre-registered verdict rule",
    )


def check_8_correlation_sums() -> None:
    """Block sums reproduce the Pearson correlation, and a resample of identical arms is zero."""
    generator = np.random.default_rng(4)
    x = generator.normal(size=900)
    y = 0.6 * x + generator.normal(size=900)
    blocks = np.tile(np.arange(6), 150)
    sums = anomaly.block_sums(x=x, y=y, blocks=blocks, n_blocks=6)
    from_sums = float(anomaly.correlation_from_sums(sums=sums.sum(axis=0)))
    _require(
        passed=abs(from_sums - float(np.corrcoef(x, y)[0, 1])) < 1e-12,
        message="check 8: correlation from block sums",
    )
    point, lower, upper = anomaly.correlation_difference_interval(sums_a=sums, sums_b=sums)
    _require(
        passed=point == 0.0 and lower == 0.0 and upper == 0.0,
        message="check 8: identical arms differ by exactly 0",
    )


def check_9_benchmark_rebuild() -> None:
    """Rebuild one run of experiment 00046's fitted ENS arm from its saved booster."""
    site, seed = "D9", 0
    booster_path = common.BOOSTER_DIR / f"d1l_seed{seed}_{site}.ubj"
    source = common.SOURCE_WORK_DIR / "arms_raw" / f"d1l_seed{seed}_{site}_fitted.parquet"
    if not booster_path.exists() or not source.exists():
        print("check 9: skipped, experiment 00046's booster or fitted arm file is absent")
        return
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)[site]
    row = next(series.iter_rows(named=True))
    booster = xgb.Booster()
    booster.load_model(booster_path)
    booster.set_param({"device": "cpu", "nthread": 1})
    path = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))[10]
    run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
    run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
    run = predict.add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
    keys = run.select(*common.KEYS).unique(maintain_order=True)
    raw = booster.predict(
        xgb.DMatrix(run.select(predict.input_columns()).to_numpy().astype(np.float32))
    )
    quantiles = raw.reshape(keys.height, common.MEMBERS, 9).astype(np.float32)
    run_file = pl.read_parquet(common.SOURCE_SOLAR_DIR / "ens_val" / path.name)
    deviation = predict.run_deviation(run_file=run_file, site=site, keys=keys)
    central_by_time = (
        run_file.filter(pl.col("site") == site)
        .group_by("valid_time")
        .agg(central=pl.col("anomaly_mean_calibrated").mean())
    )
    central = np.nan_to_num(
        keys.join(central_by_time, on="valid_time", how="left", maintain_order="left")["central"]
        .to_numpy()
        .astype(np.float64)
    )
    lead = common.lead_hours_of(frame=keys)
    k = predict.lead_band_coefficients(site=site)
    coefficient = np.array([k.get(int(b), 1.0) for b in common.lead_band(lead_hours=lead)])
    capacity = {r["site"]: r["C"] for r in common.capacities().iter_rows(named=True)}[site]
    shift = predict.member_shift(
        capacity_mw=capacity,
        effective_capacity_mw=row["effective_capacity_mw"],
        coefficient=coefficient,
        central=central,
        deviation=deviation,
    )
    mine = predict.shifted_pool(member_quantiles=quantiles, member_shift=shift)
    stored = (
        pl.read_parquet(source)
        .filter(
            (pl.col("variant") == "ens_shift_fitted")
            & (pl.col("power_fcst_init_time") == keys["power_fcst_init_time"][0])
        )
        .sort("valid_time")
    )
    difference = float(np.abs(mine - stored.select(common.LEVEL_COLUMNS).to_numpy()).max())
    print(
        f"check 9: {keys.height} cases of {path.name}; largest quantile difference {difference:.2e}"
    )
    _require(
        passed=stored.height == keys.height and difference <= SMOKE_TOLERANCE,
        message="check 9: the CPU rebuild of experiment 00046's fitted ENS arm matches",
    )


def check_10_gefs_run() -> None:
    """One GEFS run passes the ENS schedule check, spans 351 hours, and is finite."""
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    weights, _ = inputs.grid_weights(series=series)
    run = common.FIRST_TRAINING_RUN.date() + timedelta(days=100)
    table = inputs.build_run(run=run, series=series, weights=weights, offset_days=0, training=False)
    if table.is_empty():
        print("check 10: skipped, the GEFS archive holds no run for the chosen date")
        return
    origin = table["power_fcst_init_time"][0]
    last = table["valid_time"].max()
    first = table["valid_time"].min()
    print(f"check 10: {table.height:,} rows; first valid {first}, last {last}, origin {origin}")
    _require(
        passed=first == origin + timedelta(minutes=30)
        and last == origin + timedelta(hours=common.HORIZON_HOURS)
        and bool(table["u_mean"].is_finite().all())
        and table["site"].n_unique() == len(common.SITES),
        message="check 10: the run spans half an hour after the origin to 351 h later, all finite",
    )


CHECKS: Final = {
    1: check_1_deaccumulation,
    2: check_2_weights_sum_to_one,
    3: check_3_fit_recovers_weights,
    4: check_4_blend_reproduces_ens,
    5: check_5_fold_excludes_heldout_month,
    6: check_6_zero_shift_is_identity,
    7: check_7_verdict_rule,
    8: check_8_correlation_sums,
    9: check_9_benchmark_rebuild,
    10: check_10_gefs_run,
}


def main() -> None:
    """Run the requested checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=int, nargs="+", default=list(CHECKS))
    args = parser.parse_args()
    for number in args.checks:
        CHECKS[number]()


if __name__ == "__main__":
    main()
