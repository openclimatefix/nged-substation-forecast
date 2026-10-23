"""Steps 1 and 2: the wind that each farm's own power implies, against ERA5 and the ensemble.

Experiment 00261's physics core (R1) is refitted on ERA5 before 2024-04-01. Its power curve is then
run backwards on every half-hour whose observed power lies between 5% and 90% of the curve's
plateau, outside experiment 00210's idle mask and outside suspected outage runs, which gives an
implied hub-height wind speed. Dividing by ERA5's own shear and air-density factor expresses the
implied wind as a 100 m speed, so it can be compared with ERA5's and the ensemble's 100 m speeds.

Output, in `work/`: `cores.pt`, `implied.parquet` (one row per farm half-hour), and `implied.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00307-*/e00307_implied.py
"""

import json
import os
import pickle
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00307_common as common  # noqa: E402
import e00307_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from studies.power import scan_power  # noqa: E402

BASE_END: Final = datetime(2024, 4, 1, tzinfo=UTC)
DATA_END: Final = datetime(2026, 7, 1, tzinfo=UTC)
LOW_FRACTION: Final[float] = 0.05
HIGH_FRACTION: Final[float] = 0.90
OUTAGE_RUN: Final[int] = 48
OUTAGE_FRACTION: Final[float] = 0.01
GRID_STEP_M_S: Final[float] = 0.01
SPEED_BINS: Final[np.ndarray] = np.array([0, 3, 5, 7, 9, 11, 14, 18, 40], dtype=float)
SECTOR_WIDTH_DEG: Final[float] = 45.0


def weather(*, frame: pl.DataFrame) -> core_module.WeatherArrays:
    """Return the core's three weather columns from a frame with ERA5 column names."""
    return core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy().astype(np.float32),
        speed_100m=frame["wind_speed_100m"].to_numpy().astype(np.float32),
        temperature_c=frame["temperature_c"].to_numpy().astype(np.float32),
    )


def observed_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return power over capacity from the ERA5 era to `DATA_END`, with new-meter weeks dropped."""
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= common.ERA5_START,
            pl.col("time") < DATA_END,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    run = 48
    live = (pl.col("power").abs() > 0.05 * pl.col("effective_capacity_mw")).fill_null(value=False)
    run_end = live.cast(pl.Int32).rolling_sum(window_size=run).over("time_series_id") == run
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time").min().dt.offset_by(f"-{run * 30 - 30}m").dt.offset_by("6w")
        )
    )
    return (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def outage_flag(*, target: np.ndarray) -> np.ndarray:
    """Return True for half-hours inside a run of `OUTAGE_RUN` or more at under 1% of capacity."""
    low = (target < OUTAGE_FRACTION).astype(np.int8)
    flag = np.zeros(target.size, dtype=bool)
    start = None
    for index in range(target.size + 1):
        if index < target.size and low[index]:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= OUTAGE_RUN:
                flag[start:index] = True
            start = None
    return flag


class Inverter:
    """The core's power curve as hub-height speed to power, and its inverse on a monotone range."""

    def __init__(self, *, core: core_module.WindCore) -> None:
        """Tabulate the curve above the cut-in speed on a fine grid."""
        cut_in = float(core.cut_in_m_s)
        grid = np.arange(cut_in, 40.0, GRID_STEP_M_S, dtype=np.float32)
        with torch.no_grad():
            speed = torch.as_tensor(grid)
            active = core.active_power(speed=speed)
            if core.with_idle_layer:
                floor = torch.nn.functional.softplus(core.raw_idle_floor)
                dip = torch.nn.functional.softplus(core.raw_dip)
                position = speed / core.cut_in_m_s
                power = torch.sqrt(active**2 + floor**2) - dip * position * torch.exp(
                    1.0 - position
                )
            else:
                power = active
        curve = np.maximum.accumulate(power.numpy().astype(np.float64))
        self.plateau = float(curve.max())
        self.low = max(LOW_FRACTION * self.plateau, float(curve[0]) + 1e-4)
        self.high = HIGH_FRACTION * self.plateau
        top = int(np.argmax(curve >= self.high)) + 1
        self.grid = grid[:top].astype(np.float64)
        self.curve = curve[:top]
        self.rated_m_s = float(core.rated_m_s)
        self.cut_in_m_s = cut_in

    def invertible(self, *, power: np.ndarray) -> np.ndarray:
        """Return which observed fractions lie inside the invertible range."""
        return (power >= self.low) & (power <= self.high)

    def hub_speed(self, *, power: np.ndarray) -> np.ndarray:
        """Return the hub-height speed whose curve value is `power` (NaN outside the range)."""
        unique, index = np.unique(self.curve, return_index=True)
        speed = np.interp(power, unique, self.grid[index])
        return np.where(self.invertible(power=power), speed, np.nan)


def hub_factor(*, core: core_module.WindCore, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return hub-height over 100 m speed, and the hub-height speed, for the rows of `frame`."""
    arrays = weather(frame=frame)
    speed_10m, speed_100m, temperature = arrays.as_tensors()
    with torch.no_grad():
        hub = core.hub_height_speed(
            speed_10m=speed_10m, speed_100m=speed_100m, temperature_c=temperature
        ).numpy()
    return hub / np.maximum(arrays.speed_100m, 1e-3), hub


def direction_sector(*, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Return the 45-degree sector (0 to 7) of the direction whose sine and cosine are given."""
    degrees = np.degrees(np.arctan2(sin, cos)) % 360.0
    return np.floor(degrees / SECTOR_WIDTH_DEG).astype(np.int8) % 8


def summarise(*, frame: pl.DataFrame, reference: str) -> dict:
    """Return mean difference and correlation of implied 100 m wind against `reference` columns."""
    inv = frame.filter(pl.col("implied_100m").is_not_null())
    diff = inv["implied_100m"] - inv[reference]
    out = {
        "rows": inv.height,
        "mean_diff": float(diff.to_numpy().mean()),
        "rmse": float((diff**2).to_numpy().mean() ** 0.5),
        "correlation": float(np.corrcoef(inv["implied_100m"], inv[reference])[0, 1]),
    }
    binned = (
        inv.with_columns(
            bin=pl.col(reference).cut(list(SPEED_BINS[1:-1]), left_closed=True).cast(pl.String)
        )
        .group_by("bin")
        .agg(
            n=pl.len(),
            ref_mean=pl.col(reference).mean(),
            implied_mean=pl.col("implied_100m").mean(),
            diff=(pl.col("implied_100m") - pl.col(reference)).mean(),
        )
        .sort("ref_mean")
    )
    out["by_speed_bin"] = binned.to_dicts()
    return out


def main() -> None:
    """Write the implied-wind table and its summary."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    observed = observed_power(sites=sites)
    mask = common.idle_mask().with_columns(masked=pl.lit(value=True))
    pieces, cores = [], {}
    for row in sites.iter_rows(named=True):
        series, farm = int(row["time_series_id"]), row["site"]
        era5_farm = era5.filter(pl.col("time_series_id") == series).join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        fit_rows = era5_farm.filter(pl.col("valid_time") < BASE_END)
        core = core_module.fit_core(
            weather=weather(frame=fit_rows),
            target=fit_rows["target"].to_numpy(),
            with_idle_layer=farm != "W2",
            seed=0,
        )
        residuals = core_module.ResidualQuantiles.fit(
            expected=core_module.predict_expected_power(core=core, weather=weather(frame=fit_rows)),
            observed=fit_rows["target"].to_numpy(),
        )
        cores[farm] = {
            "state": core.state_dict(),
            "with_idle_layer": core.with_idle_layer,
            "idle_floor": core_module.idle_floor_start(
                speed_100m=fit_rows["wind_speed_100m"].to_numpy(),
                target=fit_rows["target"].to_numpy(),
            ),
            "residuals": residuals,
        }
        era5_farm = era5_farm.join(
            mask.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
        ).with_columns(masked=pl.col("masked").fill_null(value=False))
        target = era5_farm["target"].to_numpy()
        inverter = Inverter(core=core)
        factor, hub = hub_factor(core=core, frame=era5_farm)
        outage = outage_flag(target=target)
        usable = inverter.invertible(power=target) & ~era5_farm["masked"].to_numpy() & ~outage
        implied_hub = np.where(usable, inverter.hub_speed(power=target), np.nan)
        implied_100m = implied_hub / factor
        sector = direction_sector(
            sin=era5_farm["wind_dir_sin_100m"].to_numpy(),
            cos=era5_farm["wind_dir_cos_100m"].to_numpy(),
        )
        pieces.append(
            era5_farm.select(
                "time_series_id",
                "valid_time",
                "target",
                "masked",
                era5_100m="wind_speed_100m",
                era5_10m="wind_speed_10m",
                era5_dir_sin="wind_dir_sin_100m",
                era5_dir_cos="wind_dir_cos_100m",
            ).with_columns(
                site=pl.lit(farm),
                outage=pl.Series(outage),
                sector=pl.Series(sector),
                era5_hub=pl.Series(hub.astype(np.float32)),
                shear_density_factor=pl.Series(factor.astype(np.float32)),
                implied_hub=pl.Series(implied_hub.astype(np.float32)).fill_nan(None),
                implied_100m=pl.Series(implied_100m.astype(np.float32)).fill_nan(None),
            )
        )
        print(
            f"{farm}: plateau {inverter.plateau:.3f},"
            f" range {inverter.low:.3f} to {inverter.high:.3f},"
            f" hub speed range {inverter.grid[0]:.1f} to {inverter.grid[-1]:.1f} m/s,"
            f" {int(usable.sum())} of {target.size} rows invertible",
            flush=True,
        )
    table = pl.concat(pieces).with_columns(
        month=pl.col("valid_time").dt.month(),
        period=pl.when(pl.col("valid_time") < BASE_END)
        .then(pl.lit("in sample (before 2024-04)"))
        .otherwise(pl.lit("out of sample (2024-04 to 2026-06)")),
    )
    table.write_parquet(common.WORK_DIR / "implied.parquet")
    with (common.WORK_DIR / "cores.pkl").open("wb") as handle:
        pickle.dump(cores, handle)
    summary: dict = {}
    for period in table["period"].unique().to_list():
        part = table.filter(pl.col("period") == period)
        summary[period] = {"all_farms": summarise(frame=part, reference="era5_100m")}
        summary[period]["all_farms"]["rows"] = part.filter(
            pl.col("implied_100m").is_not_null()
        ).height
        summary[period]["rows_total"] = part.height
        summary[period]["rows_not_masked"] = part.filter(~pl.col("masked")).height
        summary[period]["by_farm"] = {
            farm: summarise(frame=part.filter(pl.col("site") == farm), reference="era5_100m")
            for farm in sorted(part["site"].unique())
        }
        summary[period]["by_sector"] = {
            int(sector): summarise(
                frame=part.filter(pl.col("sector") == sector), reference="era5_100m"
            )
            for sector in range(8)
        }
        summary[period]["by_season"] = {
            name: summarise(frame=part.filter(pl.col("month").is_in(months)), reference="era5_100m")
            for name, months in {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11],
            }.items()
        }
        # Share of rows invertible by ERA5 speed bin.
        shares = (
            part.filter(~pl.col("masked"))
            .with_columns(
                bin=pl.col("era5_100m")
                .cut(list(SPEED_BINS[1:-1]), left_closed=True)
                .cast(pl.String)
            )
            .group_by("bin")
            .agg(
                ref_mean=pl.col("era5_100m").mean(),
                n=pl.len(),
                invertible=pl.col("implied_100m").is_not_null().mean(),
            )
            .sort("ref_mean")
        )
        summary[period]["invertible_share_by_era5_bin"] = shares.to_dicts()
    (common.WORK_DIR / "implied.json").write_text(json.dumps(summary, indent=2, default=float))
    for period, entry in summary.items():
        print(period, json.dumps(entry["all_farms"]["by_speed_bin"], indent=1, default=float))
        print(period, {k: v for k, v in entry["all_farms"].items() if k != "by_speed_bin"})


if __name__ == "__main__":
    main()
