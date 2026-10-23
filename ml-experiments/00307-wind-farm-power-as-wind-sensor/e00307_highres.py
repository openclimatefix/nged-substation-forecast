"""Step 4: the first forecast hours of ICON-D2 and UKV as an analysis-like wind, through the core.

Both archives cover the scored year (ICON-D2 from Open-Meteo's previous-runs archive, 2024-01 to
2026-09; UKV from the AWS open-data bucket, 2024-10 to 2026-10, the live feed). The freshest ICON-D2
value of each hour comes from the newest run (leads of 0 to 2 hours), and the UKV value is lead 0 of
the run started at that hour. Each hour's value stands for the hour and the half-hour after it is
the mean of the two neighbouring hours. The core is the ERA5-fitted core of `e00307_implied.py`.

Output, in `work/`: `highres_rows.parquet` and `highres.json`.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00307-*/e00307_highres.py
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
from pyproj import CRS, Transformer  # noqa: E402

NWP_ROOT: Final = common.STUDIES_DATA_DIR / "downloads" / "NWP"
ICON_PATH: Final = (
    NWP_ROOT / "OPEN-METEO-PREVIOUS-RUNS" / "ICON-D2" / "previous_runs" / "combined.parquet"
)
UKV_RUNS: Final = NWP_ROOT / "UKV-AWS" / "runs"
UKV_PROJECTION: Final[str] = "+proj=laea +lat_0=54.9 +lon_0=-2.5 +x_0=0 +y_0=0 +ellps=GRS80"
UKV_HUB_INDEX: Final[int] = 2
"""Position of 100 m in the file's height axis (50, 75, 100, 150 m)."""
KMH_PER_M_S: Final[float] = 3.6
SCORED_START: Final = datetime(2025, 7, 1, tzinfo=UTC)
SCORED_END: Final = datetime(2026, 7, 1, tzinfo=UTC)
MAX_CELL_DISTANCE_M: Final[float] = 2_000.0
SPEED_BINS: Final[list[float]] = [5.0, 7.0, 9.0, 11.0, 14.0]


def to_half_hours(*, hourly: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Return hourly instants plus the mean of neighbouring hours at each :30."""
    later = hourly.select(
        time=pl.col("time") - pl.duration(hours=1), **{f"_{c}": pl.col(c) for c in columns}
    )
    half = hourly.join(later, on="time", how="left").select(
        valid_time=pl.col("time") + pl.duration(minutes=30),
        **{c: (pl.col(c) + pl.col(f"_{c}")) / 2 for c in columns},
    )
    whole = hourly.select(pl.col("time").alias("valid_time"), *columns)
    return pl.concat([whole, half.drop_nulls()]).sort("valid_time")


def icon_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return ICON-D2's freshest 10 m and 100 m wind and 2 m temperature at each farm."""
    pieces = []
    for row in sites.iter_rows(named=True):
        hourly = (
            pl.scan_parquet(ICON_PATH)
            .filter(pl.col("site") == row["site"])
            .select(
                "time",
                wind_speed_10m=pl.col("wind_speed_10m") / KMH_PER_M_S,
                wind_speed_100m=pl.col("wind_speed_100m") / KMH_PER_M_S,
                temperature_c=pl.col("temperature_2m"),
            )
            .collect()
            .drop_nulls()
        )
        pieces.append(
            to_half_hours(
                hourly=hourly, columns=["wind_speed_10m", "wind_speed_100m", "temperature_c"]
            ).with_columns(time_series_id=pl.lit(row["time_series_id"], dtype=pl.Int32))
        )
    return pl.concat(pieces)


def ukv_rows(*, sites: pl.DataFrame, lead: int = 0) -> pl.DataFrame:
    """Return UKV's wind and screen temperature at the cell nearest each farm, for `lead` hours."""
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_proj4(UKV_PROJECTION), always_xy=True
    )
    first = np.load(UKV_RUNS / "20250701T0000Z.npz")
    x_axis, y_axis = first["x"], first["y"]
    cells = {}
    for row in sites.iter_rows(named=True):
        x, y = transformer.transform(row["longitude"], row["latitude"])
        column, line = int(np.argmin(np.abs(x_axis - x))), int(np.argmin(np.abs(y_axis - y)))
        distance = float(np.hypot(x_axis[column] - x, y_axis[line] - y))
        if distance > MAX_CELL_DISTANCE_M:
            msg = f"{row['site']}: nearest UKV cell is {distance:.0f} m away"
            raise ValueError(msg)
        cells[int(row["time_series_id"])] = (line, column)
    records: dict[int, list] = {series: [] for series in cells}
    for path in sorted(UKV_RUNS.glob("*.npz")):
        stamp = datetime.strptime(path.stem, "%Y%m%dT%H%MZ").replace(tzinfo=UTC)
        if not SCORED_START.replace(day=1, month=6, year=2025) <= stamp <= SCORED_END:
            continue
        with np.load(path) as archive:
            ten = archive["wind_speed_at_10m"][lead]
            hub = archive["wind_speed_on_height_levels"][lead, UKV_HUB_INDEX]
            temperature = archive["temperature_at_screen_level"][lead]
        for series, (line, column) in cells.items():
            records[series].append(
                (
                    stamp,
                    float(ten[line, column]),
                    float(hub[line, column]),
                    float(temperature[line, column]) - 273.15,
                )
            )
    pieces = []
    for series, rows in records.items():
        hourly = pl.DataFrame(
            rows,
            schema=["time", "wind_speed_10m", "wind_speed_100m", "temperature_c"],
            orient="row",
        ).with_columns(
            time=(pl.col("time") + pl.duration(hours=lead))
            .dt.cast_time_unit("us")
            .dt.replace_time_zone("UTC"),
            time_series_id=pl.lit(series, dtype=pl.Int32),
        )
        hourly = hourly.drop_nulls().drop("time_series_id")
        pieces.append(
            to_half_hours(
                hourly=hourly, columns=["wind_speed_10m", "wind_speed_100m", "temperature_c"]
            ).with_columns(time_series_id=pl.lit(series, dtype=pl.Int32))
        )
    return pl.concat(pieces)


def core_for(*, entry: dict) -> core_module.WindCore:
    """Rebuild a stored core."""
    core = core_module.WindCore(
        with_idle_layer=entry["with_idle_layer"], idle_floor=entry["idle_floor"]
    )
    core.load_state_dict(entry["state"])
    return core.eval()


def main() -> None:
    """Score each product's analysis-like wind through the core, and compare it with the implied."""
    sites = common.site_table()
    implied = pl.read_parquet(common.WORK_DIR / "implied.parquet").select(
        "time_series_id",
        "valid_time",
        "site",
        "target",
        "masked",
        "outage",
        era5_10m="era5_10m",
        era5_100m="era5_100m",
        era5_hub="era5_hub",
        implied_100m="implied_100m",
    )
    era5_all = common.era5_rows(sites=sites).select(
        "time_series_id", "valid_time", era5_temperature="temperature_c"
    )
    rows = implied.filter(
        (pl.col("valid_time") >= SCORED_START) & (pl.col("valid_time") < SCORED_END)
    ).join(era5_all, on=["time_series_id", "valid_time"], how="left")
    products = {
        "icon_d2": icon_rows(sites=sites),
        "ukv_lead0": ukv_rows(sites=sites, lead=0),
        "ukv_lead2": ukv_rows(sites=sites, lead=2),
    }
    for name, frame in products.items():
        rows = rows.join(
            frame.select(
                "time_series_id",
                "valid_time",
                pl.col("wind_speed_10m").alias(f"{name}_10m"),
                pl.col("wind_speed_100m").alias(f"{name}_100m"),
                pl.col("temperature_c").alias(f"{name}_temperature"),
            ).fill_nan(None),
            on=["time_series_id", "valid_time"],
            how="left",
        )
    with (common.WORK_DIR / "cores.pkl").open("rb") as handle:
        stored = pickle.load(handle)
    out_rows = []
    for farm, entry in stored.items():
        core = core_for(entry=entry)
        part = rows.filter(pl.col("site") == farm).sort("valid_time")
        for name, columns in {
            "era5": ("era5_10m", "era5_100m", "era5_temperature"),
            **{
                p: (f"{p}_10m", f"{p}_100m", f"{p}_temperature")
                for p in ("icon_d2", "ukv_lead0", "ukv_lead2")
            },
        }.items():
            valid = part.drop_nulls(list(columns))
            arrays = core_module.WeatherArrays(
                speed_10m=valid[columns[0]].to_numpy().astype(np.float32),
                speed_100m=valid[columns[1]].to_numpy().astype(np.float32),
                temperature_c=valid[columns[2]].to_numpy().astype(np.float32),
            )
            expected = core_module.predict_expected_power(core=core, weather=arrays)
            quantiles = entry["residuals"].quantiles(expected=expected)
            part = part.join(
                valid.select(
                    "time_series_id",
                    "valid_time",
                    pl.Series(f"q50_{name}", quantiles[:, common.MEDIAN_INDEX]),
                    pl.Series(
                        f"score_{name}",
                        common.quantile_score(
                            quantiles=quantiles,
                            actual=valid["target"].to_numpy().astype(np.float64),
                        ).astype(np.float32),
                    ),
                ),
                on=["time_series_id", "valid_time"],
                how="left",
            )
        out_rows.append(part)
    table = pl.concat(out_rows)
    table.write_parquet(common.WORK_DIR / "highres_rows.parquet")
    names = ("era5", "icon_d2", "ukv_lead0", "ukv_lead2")
    complete = table.drop_nulls([f"q50_{n}" for n in names])
    result: dict = {"rows_scored_year": table.height}
    for label, frame in {
        "kept_all_products": complete.filter(~pl.col("masked")),
        "all_rows_all_products": complete,
    }.items():
        result[label] = {
            "rows": frame.height,
            "share_below_median": {
                n: float((frame["target"] < frame[f"q50_{n}"]).to_numpy().mean() * 100)
                for n in names
            },
            "score_points": {
                n: float(frame[f"score_{n}"].to_numpy().mean() * common.POINTS_PER_FRACTION)
                for n in names
            },
            "by_farm_share": {
                farm: {
                    n: float(
                        (
                            frame.filter(pl.col("site") == farm)["target"]
                            < frame.filter(pl.col("site") == farm)[f"q50_{n}"]
                        )
                        .to_numpy()
                        .mean()
                        * 100
                    )
                    for n in names
                }
                for farm in sorted(frame["site"].unique())
            },
        }
    # Distance to the implied wind by ERA5 speed bin, same rows for every product.
    inv = complete.filter(pl.col("implied_100m").is_not_null() & ~pl.col("masked"))
    distance: dict = {"rows": inv.height, "bins": {}}
    labels = ["<5", "5-7", "7-9", "9-11", "11-14", ">=14"]
    edges = [-1.0, *SPEED_BINS, 99.0]
    for index, label in enumerate(labels):
        part = inv.filter(
            (pl.col("era5_100m") >= edges[index]) & (pl.col("era5_100m") < edges[index + 1])
        )
        if part.height == 0:
            continue
        distance["bins"][label] = {"rows": part.height}
        for n, column in {
            "era5": "era5_100m",
            "icon_d2": "icon_d2_100m",
            "ukv_lead0": "ukv_lead0_100m",
        }.items():
            diff = part[column] - part["implied_100m"]
            distance["bins"][label][n] = {
                "bias": float(diff.to_numpy().mean()),
                "mae": float(diff.abs().to_numpy().mean()),
                "rmse": float((diff**2).to_numpy().mean() ** 0.5),
            }
    distance["overall"] = {
        n: {
            "bias": float((inv[c] - inv["implied_100m"]).to_numpy().mean()),
            "rmse": float(((inv[c] - inv["implied_100m"]) ** 2).to_numpy().mean() ** 0.5),
            "correlation": float(np.corrcoef(inv[c], inv["implied_100m"])[0, 1]),
        }
        for n, c in {
            "era5": "era5_100m",
            "icon_d2": "icon_d2_100m",
            "ukv_lead0": "ukv_lead0_100m",
        }.items()
    }
    result["distance_to_implied"] = distance
    (common.WORK_DIR / "highres.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
