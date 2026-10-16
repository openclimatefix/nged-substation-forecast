"""Inject scaled output of the solar farms into the directional Primaries and score the recovery.

The truth is `inj(t) = -k * cap_P * farm(t) / P99(farm)`, so its 99th percentile over the farm's
daylight half-hours in all history is `k * cap_P`. For each Primary, farm, season-year, size `k`,
irradiance point (`sub`: CAMS at the Primary; `own`: CAMS at the farm), and model (M1, M1d, M1L at
`sub` only) the script fits the observed series and the observed series plus the injection with the
same rows and design. The increment of the fitted capacity is the recovered capacity. The
known-answer self-tests run first and stop the script on a breach. A timestamp scan then refits the
farm-point injection with `u` shifted by -2 to +2 half-hours.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python \
        ml-experiments/00039-pv-disaggregation-cams-injection/e00039_inject.py
"""

from collections.abc import Iterator

import e00039_common as common
import e00039_model as model
import e00039_selftest as selftest
import numpy as np
import polars as pl

WhichType = str


def distance_km(*, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the great-circle distance between two (latitude, longitude) points, km."""
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(h)))


def farm_tables(*, daylight: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each farm's normalised power and own-point `u`, without dropouts.

    Power is divided by the farm's 99th percentile over its daylight half-hours in all history, the
    same convention as the 99th percentile of the unit PV shape. A half-hour at or below zero
    power while CAMS global horizontal irradiance exceeds 100 W/m² is a meter dropout and is
    removed, and remaining negative readings are set to zero. Curtailed half-hours are kept.

    Args:
        daylight: The daylight rows of every site (zenith under 75°).

    Returns:
        A table of `time`, `farm_norm`, `u_own` per farm label.
    """
    out = {}
    for farm in common.FARMS:
        g = daylight.filter((pl.col("site") == farm) & pl.col("power").is_not_null())
        p99 = float(np.nanpercentile(g["power"].to_numpy(), 99))
        g = g.filter(~((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2)))
        out[farm] = g.select(
            "time", farm_norm=pl.col("power").clip(lower_bound=0.0) / p99, u_own="u"
        )
    return out


def units(
    *, prepared: pl.DataFrame, farms: dict[str, pl.DataFrame]
) -> Iterator[tuple[str, str, int, pl.DataFrame]]:
    """Yield (Primary, farm, season-year, rows) for every unit with enough shared rows."""
    for primary in common.DIRECTIONAL_PRIMARIES:
        base = prepared.filter(
            (pl.col("site") == primary)
            & pl.col("power").is_not_null()
            & pl.col("hdd").is_not_null()
        )
        for farm in common.FARMS:
            joined = base.join(farms[farm], on="time", how="inner")
            for year in sorted(joined["season_year"].unique().to_list()):
                rows = joined.filter(pl.col("season_year") == year)
                if rows.height >= common.MIN_WINDOW_ROWS:
                    yield primary, farm, year, rows


def _score(
    *,
    base: model.Fit,
    injected: model.Fit,
    truth: np.ndarray,
    u: np.ndarray,
    target: float,
) -> dict[str, float]:
    """Score one recovery against the truth, in shares of the injected capacity `target`."""
    increment = injected.c - base.c
    estimate = increment * u
    return {
        "R": increment / target,
        "energy_ratio": float(estimate.sum() / truth.sum()),
        "nmae": float(np.mean(np.abs(estimate - truth)) / target),
        "corr": float(np.corrcoef(estimate, truth)[0, 1]),
        "se_ratio": injected.se / target,
        "binds": float(base.constrained or injected.constrained),
    }


def _fit(*, arm: str, y: np.ndarray, window: model.Window) -> model.Fit:
    if arm == "M1":
        return model.fit_half_hourly(y=y, window=window)
    if arm == "M1d":
        return model.fit_daily(y=y, window=window)
    return model.fit_half_hourly(y=y, window=window, extra=[window.cloud])


def score_unit(
    *,
    primary: str,
    farm: str,
    year: int,
    rows: pl.DataFrame,
    capacity: float,
    km: float,
) -> tuple[list[dict[str, object]], list[pl.DataFrame]]:
    """Score every size, irradiance point, and model for one unit.

    Args:
        primary: Primary label.
        farm: Farm label.
        year: Season-year.
        rows: The shared rows.
        capacity: The Primary's effective capacity, MW.
        km: Distance from the Primary to the farm.

    Returns:
        The score rows, and the half-hourly tables for the headline size and model M1.
    """
    out: list[dict[str, object]] = []
    half_hourly: list[pl.DataFrame] = []
    farm_norm = rows["farm_norm"].to_numpy().astype(float)
    summer_mask = rows["month"].is_in(common.SUMMER_MONTHS).to_numpy()
    for which, column in (("sub", "u"), ("own", "u_own")):
        window = model.window_from(rows=rows, u_column=column)
        arms = ("M1", "M1d", "M1L") if which == "sub" else ("M1", "M1d")
        base_fits = {arm: _fit(arm=arm, y=window.power, window=window) for arm in arms}
        for k in common.INJECTION_SIZES:
            target = k * capacity
            truth = target * farm_norm
            y1 = window.power - truth
            summer = model.fit_half_hourly(
                y=-truth[summer_mask], window=window.take(mask=summer_mask), constrain=False
            ).c
            winter = model.fit_half_hourly(
                y=-truth[~summer_mask], window=window.take(mask=~summer_mask), constrain=False
            ).c
            for arm in arms:
                injected = _fit(arm=arm, y=y1, window=window)
                out.append(
                    {
                        "primary": primary,
                        "farm": farm,
                        "year": year,
                        "km": km,
                        "k": k,
                        "which": which,
                        "arm": arm,
                        "rows": window.u.size,
                        **_score(
                            base=base_fits[arm],
                            injected=injected,
                            truth=truth,
                            u=window.u,
                            target=target,
                        ),
                        "R_summer": summer / target if arm == "M1" else float("nan"),
                        "R_winter": winter / target if arm == "M1" else float("nan"),
                        "truth_p99_ratio": float(np.percentile(truth, 99) / target),
                    }
                )
                if arm == "M1" and k == common.HEADLINE_SIZE:
                    increment = injected.c - base_fits[arm].c
                    half_hourly.append(
                        pl.DataFrame(
                            {
                                "time": rows["time"],
                                "primary": primary,
                                "farm": farm,
                                "year": year,
                                "which": which,
                                "truth": truth.astype(np.float32),
                                "estimate": (increment * window.u).astype(np.float32),
                                "demand": injected.demand.astype(np.float32),
                                "pv": injected.pv.astype(np.float32),
                                "residual": injected.residual.astype(np.float32),
                                "observed_plus_injection": y1.astype(np.float32),
                                "observed": window.power.astype(np.float32),
                                "u": window.u.astype(np.float32),
                                "ghi": rows["ghi_u"].to_numpy().astype(np.float32),
                            }
                        )
                    )
    return out, half_hourly


def timestamp_scan(
    *, prepared: pl.DataFrame, farms: dict[str, pl.DataFrame], capacity: dict[str, float]
) -> pl.DataFrame:
    """Refit the farm-point injection with `u` shifted by half-hours; the ratio must peak at 0.

    The fit is linear in the injection, so the recovered capacity is the capacity fitted to the
    injection alone. Shifting `u` by `s` half-hours pairs the farm's irradiance at `t + 30 s min`
    with the power at `t`.

    Args:
        prepared: Prepared daylight rows.
        farms: `farm_tables`.
        capacity: Effective capacity of each Primary.

    Returns:
        One row per unit and shift with the capacity ratio.
    """
    rows_out = []
    for shift in common.SHIFTS:
        shifted = {
            farm: table.select(pl.col("time").dt.offset_by(f"{-30 * shift}m"), "farm_norm", "u_own")
            if shift
            else table
            for farm, table in farms.items()
        }
        # the farm power must stay at its own time; only the irradiance moves
        for farm, table in farms.items():
            moved = table.select(
                pl.col("time").dt.offset_by(f"{-30 * shift}m").alias("time"), "u_own"
            )
            shifted[farm] = table.drop("u_own").join(moved, on="time", how="inner")
        for primary, farm, year, rows in units(prepared=prepared, farms=shifted):
            window = model.window_from(rows=rows, u_column="u_own")
            target = common.HEADLINE_SIZE * capacity[primary]
            truth = target * rows["farm_norm"].to_numpy().astype(float)
            c = model.fit_half_hourly(y=-truth, window=window, constrain=False).c
            rows_out.append(
                {"primary": primary, "farm": farm, "year": year, "shift": shift, "R": c / target}
            )
    return pl.DataFrame(rows_out)


def main() -> None:
    """Run the self-tests, the injection grid, and the timestamp scan."""
    frame = pl.read_parquet(common.FRAME_PATH)
    sites = pl.read_parquet(common.SITES_PATH)
    prepared = model.prepare(frame=frame)
    selftest.run(prepared=prepared)
    capacity = {r["site"]: float(r["effective_capacity_mw"]) for r in sites.iter_rows(named=True)}
    where = {r["site"]: (r["latitude"], r["longitude"]) for r in sites.iter_rows(named=True)}
    farms = farm_tables(daylight=prepared)
    scan = timestamp_scan(prepared=prepared, farms=farms, capacity=capacity)
    scan.write_parquet(common.SHIFT_PATH)
    by_shift = scan.group_by("shift").agg(pl.col("R").median()).sort("shift")
    print("timestamp scan, median R at CAMS-at-farm by shift (must peak at 0):")
    print(by_shift)
    peak = by_shift.sort("R", descending=True)["shift"][0]
    assert peak == 0, f"timestamp scan peaks at shift {peak}, not 0: stop and report"

    scores: list[dict[str, object]] = []
    tables: list[pl.DataFrame] = []
    for primary, farm, year, rows in units(prepared=prepared, farms=farms):
        s, t = score_unit(
            primary=primary,
            farm=farm,
            year=year,
            rows=rows,
            capacity=capacity[primary],
            km=distance_km(a=where[primary], b=where[farm]),
        )
        scores.extend(s)
        tables.extend(t)
    result = pl.DataFrame(scores)
    result.write_parquet(common.INJECTION_PATH)
    pl.concat(tables).write_parquet(common.INJECTION_HALF_HOURLY_PATH)
    print(f"{result.select(pl.struct('primary', 'farm', 'year').n_unique()).item()} units")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=3):
        print(
            result.filter(pl.col("k") == common.HEADLINE_SIZE)
            .group_by("which", "arm")
            .agg(
                pl.col("R").median().alias("R_median"),
                pl.col("nmae").median(),
                pl.col("binds").sum().alias("n_constrained"),
            )
            .sort("which", "arm")
        )


if __name__ == "__main__":
    main()
