"""Build each product's wind forecast at the best guess's cases, and turn it into quantiles.

For every wind farm and seed, this refits experiment 33's ERA5-trained wind model (nine quantile
levels, power-curve base margin, trained on valid times before 2025-06-17) and feeds it each
product's 10 m and 100 m wind at the farm's grid point. A case whose lead day is d reads the
product's `previous_day(d+1)` column, so the run it uses was initialised no later than 00 UTC on
the ENS run's date. A valid time at :30 averages the u and v components of the hours either side.

Output: `work/product_quantiles/<product>_seed<s>.parquet`, keyed by the case, with the nine
quantile columns, and `work/product_checks.json`. Existing outputs are kept.

    uv run python ml-experiments/00052-more-nwp-products-for-wind-farms/e00052_fit.py [--smoke]
"""

import argparse
import json
import time

import e00052_common as common
import numpy as np
import polars as pl
from e00052_common import KEY_COLUMNS, LEVEL_COLUMNS, WIND_FEATURES

OUT_DIR = common.WORK_DIR / "product_quantiles"
HEIGHTS = ("10", "100")


def cases_table(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the best guess's wind cases (seed 0) with site label, lead day, and previous-day N."""
    cases = (
        pl.read_parquet(common.BEST_GUESS_PATH)
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
        .select(*KEY_COLUMNS)
        .join(sites.select("time_series_id", "site"), on="time_series_id", validate="m:1")
    )
    return cases.with_columns(
        lead_day=common.lead_day(
            power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
        )
    ).with_columns(previous_day=pl.col("lead_day") + 1)


def hourly_columns(*, product: str, sites: list[str], days: list[int]) -> pl.DataFrame:
    """Return `site`, `time`, `previous_day`, and the u and v components at 10 m and 100 m (m/s)."""
    path = common.NWP_DIR / common.PRODUCTS[product] / "previous_runs" / "combined.parquet"
    pieces = []
    for day in days:
        columns = {
            f"{kind}_{h}m": f"wind_{kind}_{h}m_previous_day{day}"
            for h in HEIGHTS
            for kind in ("speed", "direction")
        }
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("site").is_in(sites))
            .select("site", "time", *[pl.col(src).alias(dst) for dst, src in columns.items()])
            .collect()
        )
        out = frame.select("site", "time", previous_day=pl.lit(day, dtype=pl.Int32))
        for h in HEIGHTS:
            speed = frame[f"speed_{h}m"] / common.KM_H_PER_M_S
            bearing = frame[f"direction_{h}m"] * np.pi / 180.0
            out = out.with_columns(
                (-speed * bearing.sin()).alias(f"u{h}"), (-speed * bearing.cos()).alias(f"v{h}")
            )
        pieces.append(out.drop_nulls())
    return pl.concat(pieces)


def product_features(*, product: str, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the product's `WIND_FEATURES` at every case it covers, with the case keys."""
    usable = cases.filter(pl.col("previous_day") <= common.MAX_DAY[product])
    days = sorted(usable["previous_day"].unique().to_list())
    hourly = hourly_columns(product=product, sites=cases["site"].unique().to_list(), days=days)
    half = (pl.col("valid_time").dt.minute() == 30).alias("half")
    first = usable.with_columns(
        half, hour_start=pl.col("valid_time").dt.truncate("1h")
    ).with_columns(
        hour_end=pl.when(pl.col("half"))
        .then(pl.col("hour_start").dt.offset_by("1h"))
        .otherwise(pl.col("hour_start"))
    )
    joined = first.join(
        hourly.rename({c: f"{c}_a" for c in ("u10", "v10", "u100", "v100")}),
        left_on=["site", "hour_start", "previous_day"],
        right_on=["site", "time", "previous_day"],
        how="inner",
    ).join(
        hourly.rename({c: f"{c}_b" for c in ("u10", "v10", "u100", "v100")}),
        left_on=["site", "hour_end", "previous_day"],
        right_on=["site", "time", "previous_day"],
        how="inner",
    )
    columns = []
    for h in HEIGHTS:
        u = (pl.col(f"u{h}_a") + pl.col(f"u{h}_b")) / 2
        v = (pl.col(f"v{h}_a") + pl.col(f"v{h}_b")) / 2
        bearing = (pl.arctan2(u, v) * 180.0 / np.pi + 180.0) % 360.0 * np.pi / 180.0
        columns += [
            (u**2 + v**2).sqrt().alias(f"wind_speed_{h}m"),
            bearing.sin().alias(f"wind_dir_sin_{h}m"),
            bearing.cos().alias(f"wind_dir_cos_{h}m"),
        ]
    out = joined.with_columns(columns)
    # Availability: the selected run was initialised no later than the ENS run's 00 UTC.
    late = out.filter(
        pl.col("hour_end") - pl.duration(hours=24 * pl.col("previous_day"))
        > pl.col("power_fcst_init_time").dt.truncate("1d")
    )
    if late.height:
        msg = f"{product}: a selected run was initialised after the ENS run's 00 UTC"
        raise ValueError(msg)
    if common.PUBLICATION_DELAY_HOURS[product] >= common.SERVICE_READ_HOUR:
        msg = f"{product}: the publication delay reaches the service's read time"
        raise ValueError(msg)
    return out.select(*KEY_COLUMNS, "lead_day", *WIND_FEATURES)


def training_rows(*, series: int, analysis: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return one farm's ERA5 features joined to its observed target, before the end of training."""
    return (
        analysis.filter(pl.col("time_series_id") == series)
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter(
            (pl.col("valid_time") >= common.ERA5_START) & (pl.col("valid_time") < common.TRAIN_END)
        )
    )


def main() -> None:
    """Fit the wind model for each seed and farm and write each product's quantiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, seed 0 only, no output")
    smoke = parser.parse_args().smoke
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    cases = cases_table(sites=sites)
    features = {p: product_features(product=p, cases=cases) for p in common.PRODUCTS}
    checks = {
        p: {
            "cases": cases.height,
            "covered": f.height,
            "by_lead_day": {
                int(d): int(n) for d, n in f.group_by("lead_day").len().sort("lead_day").iter_rows()
            },
            "mean_wind_speed_100m_m_s": float(f["wind_speed_100m"].mean()),
        }
        for p, f in features.items()
    }
    print(json.dumps(checks, indent=1), flush=True)
    analysis = pl.read_parquet(common.ANALYSIS_PATH)
    observed = common.observed_training_power(sites=sites)
    era5_speed = float(
        analysis.filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list())
            & (pl.col("valid_time") >= pl.datetime(2025, 7, 1, time_zone="UTC"))
            & (pl.col("valid_time") < pl.datetime(2026, 7, 1, time_zone="UTC"))
        )["wind_speed_100m"].mean()
    )
    checks["era5_mean_wind_speed_100m_m_s_validation_year"] = era5_speed
    if not smoke:
        (common.WORK_DIR / "product_checks.json").write_text(json.dumps(checks, indent=1))
    for seed in common.SEEDS[:1] if smoke else common.SEEDS:
        targets = {p: OUT_DIR / f"{p}_seed{seed}.parquet" for p in common.PRODUCTS}
        if not smoke and all(t.exists() for t in targets.values()):
            continue
        started = time.monotonic()
        pieces: dict[str, list[pl.DataFrame]] = {p: [] for p in common.PRODUCTS}
        for row in sites.iter_rows(named=True):
            series = int(row["time_series_id"])
            model = common.WindModel(
                rows=training_rows(series=series, analysis=analysis, observed=observed),
                seed=seed,
                rounds=20 if smoke else common.N_ROUNDS,
            )
            for product, frame in features.items():
                part = frame.filter(pl.col("time_series_id") == series)
                quantiles = model.predict(frame=part)
                pieces[product].append(
                    part.select(*KEY_COLUMNS, "lead_day").hstack(
                        pl.DataFrame(quantiles, schema=list(LEVEL_COLUMNS))
                    )
                )
        if smoke:
            print("smoke run: no output written", flush=True)
            return
        for product, frames in pieces.items():
            target = targets[product]
            pl.concat(frames).sort(*KEY_COLUMNS).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
