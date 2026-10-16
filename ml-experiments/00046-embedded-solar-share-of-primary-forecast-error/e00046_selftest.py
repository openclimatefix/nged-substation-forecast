"""The seven checks of experiment 00046 that must pass before any contrast is read.

Checks 4 to 6 need only the solar inputs. Checks 1 to 3 need the first `d0` and `d1l` fit files.
Check 7 needs the assembled arms. Every number is printed, and a failed check raises.

    uv run python ml-experiments/00046-*/e00046_selftest.py --checks 4 5 6
    uv run python ml-experiments/00046-*/e00046_selftest.py --checks 1 2 3
    uv run python ml-experiments/00046-*/e00046_selftest.py --checks 7
"""

import argparse
from typing import Final

import e00046_common as common
import e00046_fit as fit_module
import e00046_solar_inputs as solar_inputs
import numpy as np
import polars as pl
import xgboost as xgb

CHECK_1_TOLERANCE_POINTS: Final[float] = 0.2
CHECK_4_RANGE_POINTS: Final[tuple[float, float]] = (5.0, 10.0)
CHECK_5_MINIMUM_CORRELATION: Final[float] = 0.95
PATH_00033_ARMS: Final = (
    common.ML_EXPERIMENTS_DATA_DIR / "00033-best-guess-architecture-vs-production" / "work" / "arms"
)


def _require(*, passed: bool, message: str) -> None:
    """Print a check's outcome and raise when it failed."""
    print(("PASS: " if passed else "FAIL: ") + message, flush=True)
    if not passed:
        raise AssertionError(message)


def check_1_d0_matches_00033(*, rows: pl.DataFrame) -> None:
    """Compare the refit `d0` raw median with experiment 00033's `n1_refit` on the scored rows."""
    files = sorted(common.RAW_ARMS_DIR.glob("d0_seed*_*.parquet"))
    _require(passed=bool(files), message="check 1: at least one d0 file exists")
    scored = rows.filter(pl.col("scored")).select(*common.KEYS)
    for file in files:
        seed = int(file.stem.split("_seed")[1].split("_")[0])
        mine = pl.read_parquet(file).filter(pl.col("variant") == "base")
        stored = pl.read_parquet(PATH_00033_ARMS / f"n1_refit_seed{seed}.parquet").select(
            *common.KEYS, stored_q50=pl.col("q50")
        )
        both = scored.join(mine.select(*common.KEYS, "q50"), on=list(common.KEYS)).join(
            stored, on=list(common.KEYS)
        )
        difference = float((both["q50"] - both["stored_q50"]).abs().to_numpy().mean()) * 100.0
        print(
            f"check 1: {file.name}: {both.height:,} scored rows, mean absolute difference of the "
            f"raw median {difference:.4f} points",
            flush=True,
        )
        _require(
            passed=both.height > 0 and difference < CHECK_1_TOLERANCE_POINTS,
            message=f"check 1: {file.name} matches n1_refit_seed{seed} of experiment 00033",
        )


def check_2_column_counts() -> None:
    """The column lists hold 13 + 1 + 3 for `d0` and two fewer for `d1l`."""
    d0 = fit_module.input_columns(arm="d0")
    d1l = fit_module.input_columns(arm="d1l")
    print(f"check 2: d0 columns ({len(d0)}): {d0}")
    print(f"check 2: d1l columns ({len(d1l)}): {d1l}")
    _require(
        passed=len(d0) == 17 and len(d1l) == 15 and set(d0) - set(d1l) == {"shortwave", "longwave"},
        message="check 2: d0 has 17 columns and d1l the same minus shortwave and longwave",
    )


def check_3_zero_shift_is_identity() -> None:
    """A shift of zero returns the unshifted pooled quantiles bit for bit."""
    boosters = sorted(common.BOOSTER_DIR.glob("d1l_seed0_*.ubj"))
    _require(passed=bool(boosters), message="check 3: a d1l booster exists")
    site = boosters[0].stem.rsplit("_", 1)[1]
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)
    row = next(series.iter_rows(named=True))
    booster = xgb.Booster()
    booster.load_model(boosters[0])
    _file, keys, member_quantiles = next(
        fit_module.member_quantile_runs(
            row=row,
            arrays=arrays[site],
            booster=booster,
            columns=fit_module.input_columns(arm="d1l"),
        )
    )
    cases = member_quantiles.shape[0]
    base = common.mixture_quantiles(member_quantiles=member_quantiles)
    zero_ens = np.zeros((cases, member_quantiles.shape[1]))
    zero_cams = np.zeros(cases)
    context = {
        "effective_capacity_mw": row["effective_capacity_mw"],
        "ens": {"anomaly_calibrated": zero_ens, "anomaly_mean_calibrated": zero_ens},
        "cams": zero_cams,
    }
    tables = fit_module.variant_tables(
        arm="d1l",
        member_quantiles=member_quantiles,
        run_context=context,
        coefficients={"C": 2.0, "headline": 2.0},
        names=fit_module.VARIANTS_D1L,
    )
    for name, quantiles in tables.items():
        identical = np.array_equal(quantiles, base)
        print(f"check 3: {name} equals the unshifted pool bit for bit: {identical}")
    _require(
        passed=all(np.array_equal(q, base) for q in tables.values()) and keys.height == cases,
        message=f"check 3: a zero anomaly returns the unshifted quantiles for {site}",
    )
    shifted = fit_module.shifted_pool(
        member_quantiles=member_quantiles, member_shift=np.full(zero_ens.shape, 0.1)
    )
    _require(
        passed=bool(np.allclose(shifted, base - 0.1, atol=1e-5)),
        message="check 3: a constant shift of 0.1 moves every pooled quantile by -0.1",
    )


def check_4_shift_units(*, rows: pl.DataFrame, series: pl.DataFrame) -> None:
    """The oracle shift has a standard deviation of 5 to 10 points of capacity at every series."""
    capacity = common.capacities()
    scored = rows.filter(pl.col("scored")).join(
        series.select("site", "effective_capacity_mw"), on="site"
    )
    ok = True
    low, high = CHECK_4_RANGE_POINTS
    for row in capacity.iter_rows(named=True):
        own = scored.filter(pl.col("site") == row["site"])
        shift = 100.0 * row["C"] * own["anomaly_cams"].to_numpy() / own["effective_capacity_mw"][0]
        std = float(shift.std())
        print(
            f"check 4: {row['site']}: C {row['C']:.2f} MW, effective capacity "
            f"{own['effective_capacity_mw'][0]:.2f} MW, {own.height:,} scored rows, standard "
            f"deviation of the oracle shift {std:.2f} points (mean {shift.mean():.2f})"
        )
        if row["site"] in common.VERDICT_SITES:
            ok &= low <= std <= high
    _require(passed=ok, message="check 4: oracle shift standard deviation is 5 to 10 points")
    fraction = rows.filter(pl.col("site").is_in(list(common.VERDICT_SITES)))["scored"].mean()
    print(f"check 4: scored rows are {fraction:.3f} of all validation cases of the verdict sites")


def check_5_ens_chain() -> None:
    """The ENS chain agrees with the CAMS route on anomalies, and the control member's skill."""
    cams = solar_inputs.cams_frame().filter(pl.col("time") <= common.LAST_TRAINING_VALID)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    worst = 1.0
    for row in series.iter_rows(named=True):
        own = cams.filter(pl.col("site") == row["site"]).sort("time")
        chain = solar_inputs.u_from_half_hour_ghi(
            stamps=own["time"],
            ghi=own["ghi_u"].to_numpy()[None, :],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )[0]
        frame = own.with_columns(u_chain=pl.Series(chain))
        pooled = {
            name: solar_inputs.climatology(frame=frame, value=name, time_column="time", pool=True)
            for name in ("u", "u_chain")
        }
        anomalies = common.with_cell(frame=frame, time_column="time")
        for name, clim in pooled.items():
            anomalies = anomalies.join(
                clim.select("site", "month", "hh", **{f"clim_{name}": "clim"}),
                on=["site", "month", "hh"],
            ).with_columns(**{f"anomaly_{name}": pl.col(name) - pl.col(f"clim_{name}")})
        daylight = anomalies.filter(pl.col("zenith") < solar_inputs.MAX_ZENITH_DEG)
        correlation = float(daylight.select(pl.corr("anomaly_u", "anomaly_u_chain")).item())
        maximum = float((daylight["u"] - daylight["u_chain"]).abs().to_numpy().max())
        print(
            f"check 5 part 1: {row['site']}: anomaly correlation of the ENS chain on CAMS "
            f"irradiance with the CAMS route {correlation:.5f}, max |u difference| {maximum:.4f}, "
            f"{daylight.height:,} daylight half-hours"
        )
        worst = min(worst, correlation)
    _require(
        passed=worst >= CHECK_5_MINIMUM_CORRELATION,
        message=f"check 5 part 1: smallest anomaly correlation {worst:.5f} is at least 0.95",
    )
    calibration = pl.read_parquet(common.SOLAR_DIR / "calibration.parquet")
    names = [name for name, _low, _high in common.LEAD_BANDS]
    summary = (
        calibration.group_by("band")
        .agg(
            correlation_mean=pl.col("correlation_mean").mean(),
            correlation_control=pl.col("correlation_control").mean(),
            slope=pl.col("slope").mean(),
        )
        .sort("band")
        .with_columns(
            lead_band=pl.col("band").map_elements(names.__getitem__, return_dtype=pl.String)
        )
    )
    with pl.Config(tbl_rows=20):
        print(
            "check 5 part 2: ENS anomaly against CAMS anomaly on the training window, mean of sites"
        )
        print(summary)
        print(calibration)


def check_6_training_window_only() -> None:
    """Every climatology, slope, and the daylight mask use valid times up to 2025-06-30 only."""
    last = common.LAST_TRAINING_VALID
    train_latest = max(
        common.latest_time(times=pl.read_parquet(path, columns=["valid_time"])["valid_time"])
        for path in (common.SOLAR_DIR / "ens_train").glob("*.parquet")
    )
    print(f"check 6: latest valid time behind the ENS climatology and slopes {train_latest}")
    _require(passed=train_latest <= last, message="check 6: ENS training files end by 2025-06-30")
    cams = solar_inputs.cams_frame()
    later = cams.filter(pl.col("time") > last).select("site", "time", "u")
    try:
        solar_inputs.climatology(frame=later, value="u", time_column="time", pool=False)
    except ValueError as error:
        print(f"check 6: the climatology guard refuses later rows: {error}")
    else:
        _require(passed=False, message="check 6: the climatology guard refuses later valid times")
    saved = pl.read_parquet(common.SOLAR_DIR / "climatology.parquet")
    recomputed = solar_inputs.climatology(
        frame=cams.filter(pl.col("time") <= last), value="u", time_column="time", pool=True
    )
    both = saved.join(recomputed, on=["site", "month", "hh"])
    worst = float((both["clim_cams"] - both["clim"]).abs().to_numpy().max())
    print(f"check 6: saved CAMS climatology against a training-window rebuild, max {worst:.2e}")
    _require(passed=worst < 1e-9, message="check 6: the saved CAMS climatology is training-only")


def check_7_same_keys() -> None:
    """All arms hold the same scored keys, which `losses_for_bootstrap` also enforces."""
    arms = sorted(common.ARMS_DIR.glob("*.parquet"))
    _require(passed=bool(arms), message="check 7: assembled arms exist")
    reference = None
    for path in arms:
        keys = pl.read_parquet(path, columns=[*common.KEYS, "seed"]).sort(*common.KEYS, "seed")
        if reference is None:
            reference = keys
        print(f"check 7: {path.stem}: {keys.height:,} keys")
        _require(passed=keys.equals(reference), message=f"check 7: {path.stem} holds the same keys")


def main() -> None:
    """Run the requested checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=int, nargs="+", required=True)
    args = parser.parse_args()
    wanted = set(args.checks)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    rows = None
    if wanted & {1, 4}:
        arrays = common.load_series_arrays(series=series)
        rows = common.scored_rows(series=series, arrays=arrays)
    if 1 in wanted:
        assert rows is not None
        check_1_d0_matches_00033(rows=rows)
    if 2 in wanted:
        check_2_column_counts()
    if 3 in wanted:
        check_3_zero_shift_is_identity()
    if 4 in wanted:
        assert rows is not None
        check_4_shift_units(rows=rows, series=series)
    if 5 in wanted:
        check_5_ens_chain()
    if 6 in wanted:
        check_6_training_window_only()
    if 7 in wanted:
        check_7_same_keys()


if __name__ == "__main__":
    main()
