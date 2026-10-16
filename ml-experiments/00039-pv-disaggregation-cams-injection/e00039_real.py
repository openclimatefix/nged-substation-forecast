"""Fit the models to the real (un-injected) series and compute the inputs of checks K1 to K4.

For every directional Primary, MVA Primary, and bulk supply point (the substations), and for each
solar farm (generation, so its sign is flipped and its own irradiance is used), in every
season-year with enough daylight half-hours, the script writes `work/real.parquet` with:

- `c`, `se`, `b`: M1 capacity (MW), its day-clustered standard error, and the HDD coefficient;
- `c_summer`, `c_winter`: M1 on April to September and on October to March;
- `c_weekday`, `c_weekend`: M1 on weekdays and on weekends and bank holidays (K4);
- `c_cloud`: M1L capacity (K3);
- `c_daily`: M1d capacity;
- `c_clear`: M1 with the clear-sky shape, a control that is not a capacity estimate.

and `work/placebo.parquet` with `PLACEBO_DRAWS` permuted-day capacities per directional
Primary-year (K1).

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python \
        ml-experiments/00039-pv-disaggregation-cams-injection/e00039_real.py
"""

import e00039_common as common
import e00039_model as model
import numpy as np
import polars as pl


def _capacity_or_nan(*, y: np.ndarray, window: model.Window, mask: np.ndarray, hdd: bool) -> float:
    """Fit M1 on the masked rows, returning NaN when there are too few rows."""
    if mask.sum() < 500:
        return float("nan")
    return model.fit_half_hourly(
        y=y[mask], window=window.take(mask=mask), use_hdd=hdd, constrain=False
    ).c


def fit_series_year(*, rows: pl.DataFrame, flip: bool) -> dict[str, float]:
    """Return every real-series fit for one series in one season-year.

    Args:
        rows: Prepared daylight rows with non-null power.
        flip: True for a generator, whose output is minus the demand sign.

    Returns:
        The fitted capacities and standard errors.
    """
    hdd = not flip
    window = model.window_from(rows=rows)
    y = -window.power if flip else window.power
    sign = 1.0
    main = model.fit_half_hourly(y=y, window=window, use_hdd=hdd, constrain=False)
    summer = np.isin(window.month, common.SUMMER_MONTHS)
    weekend = window.off == 1
    clear = model.window_from(rows=rows, u_column="u_clear")
    return {
        "rows": float(len(y)),
        "c": sign * main.c,
        "se": main.se,
        "b": main.b,
        "c_summer": _capacity_or_nan(y=y, window=window, mask=summer, hdd=hdd),
        "c_winter": _capacity_or_nan(y=y, window=window, mask=~summer, hdd=hdd),
        "c_weekday": _capacity_or_nan(y=y, window=window, mask=~weekend, hdd=hdd),
        "c_weekend": _capacity_or_nan(y=y, window=window, mask=weekend, hdd=hdd),
        "c_clear": model.fit_half_hourly(y=y, window=clear, use_hdd=hdd, constrain=False).c,
        "c_daily": model.fit_daily(y=y, window=window, constrain=False).c,
        "c_cloud": model.fit_half_hourly(
            y=y, window=window, extra=[window.cloud], use_hdd=hdd, constrain=False
        ).c
        if hdd
        else float("nan"),
        "constrained": float(main.c < 0.0),
    }


def placebo_capacities(*, rows: pl.DataFrame, rng: np.random.Generator) -> list[float]:
    """Return `PLACEBO_DRAWS` capacities fitted with whole days of `u` permuted within the month."""
    window = model.window_from(rows=rows)
    out = []
    for _ in range(common.PLACEBO_DRAWS):
        permuted = model.placebo_u(window=window, rng=rng)
        ok = ~np.isnan(permuted)
        shuffled = model.Window(
            **{
                **{n: getattr(window, n)[ok] for n in window.__dataclass_fields__},
                "u": permuted[ok],
            }
        )
        out.append(model.fit_half_hourly(y=shuffled.power, window=shuffled, constrain=False).c)
    return out


def main() -> None:
    """Fit every series-year and write the real-series tables."""
    frame = pl.read_parquet(common.FRAME_PATH)
    sites = pl.read_parquet(common.SITES_PATH)
    capacity = {r["site"]: float(r["effective_capacity_mw"]) for r in sites.iter_rows(named=True)}
    prepared = model.prepare(frame=frame)
    rng = np.random.default_rng(common.PLACEBO_SEED)
    substations = (*common.DIRECTIONAL_PRIMARIES, *common.MVA_PRIMARIES, *common.BSP_SITES)
    fits: list[dict[str, object]] = []
    placebos: list[dict[str, object]] = []
    for site in (*substations, *common.FARMS):
        flip = site in common.FARMS
        base = prepared.filter((pl.col("site") == site) & pl.col("power").is_not_null())
        if not flip:
            base = base.filter(pl.col("hdd").is_not_null())
        else:
            base = base.with_columns(hdd=pl.lit(0.0))
        for year in sorted(base["season_year"].unique().to_list()):
            rows = base.filter(pl.col("season_year") == year)
            if rows.height < common.MIN_WINDOW_ROWS:
                continue
            fits.append(
                {
                    "site": site,
                    "year": year,
                    "role": "farm"
                    if flip
                    else (
                        "directional"
                        if site in common.DIRECTIONAL_PRIMARIES
                        else "bsp"
                        if site in common.BSP_SITES
                        else "mva"
                    ),
                    "cap": capacity[site],
                    **fit_series_year(rows=rows, flip=flip),
                }
            )
            if site in common.DIRECTIONAL_PRIMARIES or site in common.MVA_PRIMARIES:
                placebos.extend(
                    {"site": site, "year": year, "draw": i, "c": c}
                    for i, c in enumerate(placebo_capacities(rows=rows, rng=rng))
                )
    real = pl.DataFrame(fits)
    real.write_parquet(common.REAL_PATH)
    pl.DataFrame(placebos).write_parquet(common.PLACEBO_PATH)
    with pl.Config(tbl_rows=120, tbl_width_chars=220, float_precision=2):
        print(
            real.filter(pl.col("role") == "directional")
            .select("site", "year", "rows", "c", "se", "c_summer", "c_winter", "c_cloud")
            .sort("site", "year")
        )


if __name__ == "__main__":
    main()
