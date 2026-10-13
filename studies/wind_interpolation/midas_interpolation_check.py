"""Compare interpolated nominal 10 m forecast speed directly with MIDAS observations.

This post-hoc check uses measured SYNOP means over HH-20 to HH-10, matching that window with
numerical quadrature of the interpolated vector magnitude. It fits no models and downloads
nothing. Run after `wind_interpolation_experiment.py`; outputs go in its `midas/` subdirectory.
"""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import h3
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference, bootstrap_difference_at_level
from studies.midas import read_station_metadata
from studies.sources import MIDAS_OPEN_PRODUCT_DIR, REPO_DATA_DIR, study_dir_for
from wind_interpolation_experiment import LEADS, METHODS, TRAIN_START, MethodType, interpolate

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
STOP: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
OBS_HOURS: Final[np.ndarray] = np.concatenate((np.arange(25, 49), np.arange(169, 193)))
OUTPUT: Final[Path] = study_dir_for(study="wind_interpolation")


def mean_speed(*, u: np.ndarray, v: np.ndarray, method: MethodType, points: int) -> np.ndarray:
    """Average instantaneous speed over each SYNOP observation's ten-minute window."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    targets = OBS_HOURS[:, None] - 0.25 + nodes[None, :] / 12
    east = interpolate(values=u, x=LEADS, targets=targets.flatten(), method=method)
    north = interpolate(values=v, x=LEADS, targets=targets.flatten(), method=method)
    speed = np.hypot(east, north).reshape(len(u), len(OBS_HOURS), points)
    return (speed * weights[None, None, :]).sum(axis=2) / 2


def _read(*, output: Path, version: int) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Read measured SYNOP speeds and complete native forecasts at station H3 cells."""
    weather_path = MIDAS_OPEN_PRODUCT_DIR / "uk_hourly_weather_obs.parquet"
    metadata_path = (
        MIDAS_OPEN_PRODUCT_DIR
        / "_station_metadata"
        / "midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv"
    )
    raw = (
        pl.scan_parquet(weather_path)
        .filter(
            (pl.col("time") >= TRAIN_START)
            & (pl.col("time") < STOP)
            & (pl.col("met_domain_name") == "SYNOP")
            & pl.col("wind_speed_unit_id").is_in([1, 4])
        )
        .select("src_id", "time", "wind_speed_m_s")
        .collect()
    )
    observed = raw.filter(pl.col("wind_speed_m_s").is_finite() & (pl.col("wind_speed_m_s") >= 0))
    assert observed.select("src_id", "time").unique().height == observed.height
    identifiers = sorted(observed["src_id"].unique().to_list())
    labels = pl.DataFrame(
        {"src_id": identifiers, "site": [f"S{i + 1:02}" for i in range(len(identifiers))]}
    )
    metadata = read_station_metadata(path=metadata_path).join(labels, on="src_id", how="inner")
    assert metadata.height == len(identifiers)
    cells = [
        h3.str_to_int(h3.latlng_to_cell(lat=lat, lng=lon, res=5))
        for lat, lon in metadata.select("latitude", "longitude").iter_rows()
    ]
    lookup = metadata.select("site").with_columns(h3_index=pl.Series(cells, dtype=pl.Int64))
    native = (
        pl.scan_delta(str(REPO_DATA_DIR / "NWP"), version=version)
        .filter(
            (pl.col("ensemble_member") == 0)
            & pl.col("h3_index").is_in(cells)
            & (pl.col("init_time") >= TRAIN_START)
            & (pl.col("init_time") < STOP)
            & (pl.col("init_time").dt.hour() == 0)
        )
        .select("h3_index", "init_time", "valid_time", "wind_u_10m", "wind_v_10m")
        .join(lookup.lazy(), on="h3_index")
        .with_columns(lead=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60)
        .select("site", "init_time", "lead", "wind_u_10m", "wind_v_10m")
        .collect()
        .sort("site", "init_time", "lead")
    )
    assert native.select("site", "init_time", "lead").unique().height == native.height
    all_runs = native.select("site", "init_time").unique().height
    complete = (
        native.group_by("site", "init_time")
        .agg(pl.len().alias("steps"))
        .filter(pl.col("steps") == len(LEADS))
    )
    native = native.join(complete.select("site", "init_time"), on=["site", "init_time"]).sort(
        "site", "init_time", "lead"
    )
    observations = observed.join(labels, on="src_id").select(
        "site", "time", actual=pl.col("wind_speed_m_s")
    )
    native.write_parquet(output / "native.parquet")
    observations.write_parquet(output / "observations.parquet")
    provenance = {
        "nwp_version": version,
        "candidate_stations": len(identifiers),
        "represented_stations": native["site"].n_unique(),
        "excluded_incomplete_station_runs": all_runs - complete.height,
        "weather_sha256": hashlib.sha256(weather_path.read_bytes()).hexdigest(),
        "station_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "unit_codes": [1, 4],
        "message": "SYNOP",
        "window": "timestamp minus20 to minus10 minutes",
        "start": TRAIN_START.isoformat(),
        "stop_exclusive": STOP.isoformat(),
        "status": "post-hoc exploratory",
    }
    return native, observations, provenance


def _score(*, output: Path, native: pl.DataFrame, observed: pl.DataFrame) -> dict[str, float]:
    """Pair every method on identical station/run/observation keys and score speed MAE."""
    run_keys = native.select("site", "init_time").unique(maintain_order=True)
    u = native["wind_u_10m"].to_numpy().reshape(-1, len(LEADS))
    v = native["wind_v_10m"].to_numpy().reshape(-1, len(LEADS))
    assert np.isfinite(u).all()
    assert np.isfinite(v).all()
    assert np.all(native["lead"].to_numpy().reshape(-1, len(LEADS)) == LEADS)
    base = (
        pl.DataFrame(
            {
                "site": np.repeat(run_keys["site"].to_numpy(), len(OBS_HOURS)),
                "init_time": np.repeat(run_keys["init_time"].to_numpy(), len(OBS_HOURS)),
                "observation_hour": np.tile(OBS_HOURS, run_keys.height),
                "band": np.tile(["day1"] * 24 + ["day7"] * 24, run_keys.height),
            }
        )
        .with_columns(pl.col("init_time").dt.replace_time_zone("UTC"))
        .with_columns(time=pl.col("init_time") + pl.duration(hours=pl.col("observation_hour")))
    )
    common = base.join(observed, on=["site", "time"], how="inner").sort(
        "site", "init_time", "observation_hour"
    )
    errors = {}
    parts = []
    for method in METHODS:
        prediction = mean_speed(u=u, v=v, method=method, points=5)
        refined = mean_speed(u=u, v=v, method=method, points=10)
        errors[method] = float(np.max(np.abs(prediction - refined)))
        assert errors[method] < 1e-3, "Ten-minute quadrature is not accurate enough"
        frame = (
            base.with_columns(prediction=pl.Series(prediction.flatten()))
            .join(
                common.select("site", "init_time", "observation_hour", "actual"),
                on=["site", "init_time", "observation_hour"],
                validate="1:1",
            )
            .with_columns(
                arm=pl.lit(method), seed=pl.lit(0), month=pl.col("time").dt.strftime("%Y-%m")
            )
            .with_columns(mae=(pl.col("prediction") - pl.col("actual")).abs())
            .sort("band", "site", "time")
        )
        assert frame.height == common.height
        parts.append(frame)
    losses = pl.concat(parts)
    losses.write_parquet(output / "losses.parquet")
    absolute = (
        losses.group_by("band", "arm")
        .agg(mae_m_s=pl.col("mae").mean(), rows=pl.len())
        .sort("band", "arm")
    )
    absolute.write_csv(output / "absolute.csv")
    contrasts = []
    for (band,), rows in losses.partition_by("band", as_dict=True).items():
        assert rows.select("site", "time", "arm").unique().height == rows.height
        for method in ("pchip", "makima"):
            interval = bootstrap_difference(
                losses=rows, treatment=method, reference="linear", metric="mae"
            )
            low, high = bootstrap_difference_at_level(
                losses=rows, treatment=method, reference="linear", metric="mae", level=98.75
            )
            contrasts.append(
                {
                    "band": band,
                    "method": method,
                    **interval,
                    "lower_98_75": low,
                    "upper_98_75": high,
                }
            )
    contrasts = pl.DataFrame(contrasts).sort("band", "method")
    contrasts.write_csv(output / "contrasts.csv")
    (output / "report.md").write_text(
        "# Post-hoc MIDAS station check\n\n"
        "Speed MAE in m/s; measured SYNOP means over HH-20 to HH-10.\n\n"
        + absolute.write_csv()
        + "\nSmooth minus linear; whole-month paired intervals, no fitted models.\n\n"
        + contrasts.write_csv()
    )
    _LOG.info("Station observations matched: %d; quadrature errors: %s", common.height, errors)
    return errors


def main() -> None:
    """Run the direct observational check once, retaining all anonymised intermediates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output = args.output / "midas"
    output.mkdir(parents=True, exist_ok=False)
    version = json.loads((args.output / "provenance.json").read_text())["nwp_version"]
    native, observed, provenance = _read(output=output, version=version)
    provenance["max_5_vs_10_point_quadrature_error_m_s"] = _score(
        output=output, native=native, observed=observed
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
