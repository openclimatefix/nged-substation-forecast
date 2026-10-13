"""Audit cached control-member features and probe the historical polar-wind calculations.

Only aggregate tables leave memory. Source identifiers and coordinates are never written.
The private cache and the implementation pinned for this historical audit are separate evidence
sources. Reproduction requires checkout 5ac2e16a7088d13f2f7103f0385a58f54388bfba and its
matching polar archive, engineered caches, and dependencies.
"""

import argparse
import hashlib
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import yaml
from contracts.settings import PROJECT_ROOT
from contracts.weather_schemas import Nwp
from deltalake import DeltaTable
from ml_core.features._lags import _nullify_leaky_lags
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from ml_core.features._parsed_features import STATIC_FEATURE_REGISTRY, LagFeature
from ml_core.features.tabular_feature_engineer import _apply_local_time_features
from studies.power import CLEANED_POWER_DELTA_URI, scan_power
from studies.solar import zenith

GROUPS: Final[list[str]] = ["time_series_id", "nwp_init_time", "ensemble_member"]
KEYS: Final[list[str]] = [*GROUPS, "valid_time"]
RADIATION: Final[str] = "downward_short_wave_radiation_flux_surface"
PRECIPITATION: Final[str] = "categorical_precipitation_type_surface"
LAG: Final[str] = "power_lag_336h"
LEAD_BANDS: Final[tuple[str, ...]] = ("0–72 h", "72–144 h", "144–264 h", "264 h and later")


def missing(column: str) -> pl.Expr:
    """Identify both null and NaN numeric values."""
    return pl.col(column).is_null() | pl.col(column).is_nan().fill_null(False)


def add_leads(frame: pl.DataFrame) -> pl.DataFrame:
    """Add issuance-based lead bands and native weather-grid membership."""
    return frame.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600,
        nwp_lead=(pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_seconds() / 3600,
    ).with_columns(
        band=pl.when(pl.col("lead") < 72)
        .then(pl.lit(LEAD_BANDS[0]))
        .when(pl.col("lead") < 144)
        .then(pl.lit(LEAD_BANDS[1]))
        .when(pl.col("lead") < 264)
        .then(pl.lit(LEAD_BANDS[2]))
        .otherwise(pl.lit(LEAD_BANDS[3])),
        native=pl.when(pl.col("nwp_lead") <= 144)
        .then(pl.col("nwp_lead") % 3 == 0)
        .otherwise(pl.col("nwp_lead") % 6 == 0),
    )


def population(frame: pl.DataFrame, *, eligible: pl.Expr, affected: pl.Expr) -> pl.DataFrame:
    """Count affected and eligible rows and series within each issuance lead band."""
    return (
        frame.filter(eligible.fill_null(False))
        .group_by("band")
        .agg(
            eligible=pl.len(),
            affected=affected.fill_null(False).sum(),
            affected_series=pl.col("time_series_id").filter(affected.fill_null(False)).unique(),
        )
        .with_columns(percent=100 * pl.col("affected") / pl.col("eligible"))
    )


def direction_audit(frame: pl.DataFrame, *, height: int) -> pl.DataFrame:
    """Compare inserted directions with shortest-arc reconstruction of finite native knots."""
    column = f"wind_direction_{height}m"
    native = pl.col("native") & pl.col(column).is_finite()
    knots = frame.select(*KEYS, "band", "native", "nwp_lead", column).with_columns(
        knot=pl.when(native).then(pl.col(column)),
        knot_hour=pl.when(native).then(pl.col("nwp_lead")),
    )
    knots = knots.with_columns(
        left=pl.col("knot").forward_fill().over(GROUPS, order_by="valid_time"),
        right=pl.col("knot").backward_fill().over(GROUPS, order_by="valid_time"),
        left_hour=pl.col("knot_hour").forward_fill().over(GROUPS, order_by="valid_time"),
        right_hour=pl.col("knot_hour").backward_fill().over(GROUPS, order_by="valid_time"),
    ).with_columns(delta=(pl.col("right") - pl.col("left") + 180) % 360 - 180)
    knots = (
        knots.with_columns(
            fraction=(pl.col("nwp_lead") - pl.col("left_hour"))
            / (pl.col("right_hour") - pl.col("left_hour")),
        )
        .with_columns(
            circular=(pl.col("left") + pl.col("fraction") * pl.col("delta")) % 360,
        )
        .with_columns(
            discrepancy=((pl.col(column) - pl.col("circular") + 180) % 360 - 180).abs(),
        )
    )
    eligible = (
        ~pl.col("native") & pl.col("discrepancy").is_finite() & (pl.col("delta").abs() < 179.999)
    )
    return population(knots, eligible=eligible, affected=pl.col("discrepancy") > 1).with_columns(
        height_m=pl.lit(height)
    )


def solar_geometry(frame: pl.DataFrame, *, metadata: pl.DataFrame) -> pl.DataFrame:
    """Classify half-hours and native averaging intervals using deduplicated solar positions."""
    pieces = []
    for (series_id,), part in (
        frame.select("time_series_id", "valid_time")
        .unique()
        .partition_by("time_series_id", as_dict=True)
        .items()
    ):
        location = metadata.filter(pl.col("time_series_id") == series_id)
        if location.height != 1:
            raise ValueError("each audited series needs exactly one metadata location")
        latitude, longitude = location.select("latitude", "longitude").row(0)
        stamps = part["valid_time"]
        angles = [
            zenith(
                stamps=stamps.dt.offset_by(f"-{minutes}m"),
                latitude=float(latitude),
                longitude=float(longitude),
            )
            for minutes in range(0, 361, 15)
        ]
        if not np.isfinite(angles).all():
            raise ValueError("non-finite solar geometry")
        pieces.append(
            part.with_columns(
                dark_half_hour=pl.Series(np.min(angles[:3], axis=0) > 90),
                dark_3h=pl.Series(np.min(angles[:13], axis=0) > 90),
                dark_6h=pl.Series(np.min(angles, axis=0) > 90),
            )
        )
    return pl.concat(pieces)


def temperature_ranges(frame: pl.DataFrame) -> pl.DataFrame:
    """Measure retained-knot daily range after thinning short-lead native 3 h values to 6 h."""
    native = frame.filter(
        pl.col("native"), pl.col("nwp_lead") > 0, pl.col("nwp_lead") <= 144
    ).with_columns(day=pl.col("valid_time").dt.truncate("1d"))
    days = native.group_by(*GROUPS, "day").agg(
        count_3h=pl.col("temperature_2m").is_finite().sum(),
        count_6h=pl.col("temperature_2m").filter(pl.col("nwp_lead") % 6 == 0).is_finite().sum(),
        range_3h=pl.col("temperature_2m").max() - pl.col("temperature_2m").min(),
        range_6h=pl.col("temperature_2m").filter(pl.col("nwp_lead") % 6 == 0).max()
        - pl.col("temperature_2m").filter(pl.col("nwp_lead") % 6 == 0).min(),
    )
    return days.filter(pl.col("count_3h") == 8, pl.col("count_6h") == 4).select(
        days=pl.len(),
        mean_range_3h=pl.col("range_3h").mean(),
        mean_range_6h=pl.col("range_6h").mean(),
        mean_range_loss=(pl.col("range_3h") - pl.col("range_6h")).mean(),
        affected_days=(pl.col("range_6h") < pl.col("range_3h") - 1e-5).sum(),
    )


def source_probes() -> dict[str, pl.DataFrame]:
    """Run known-answer probes of the pinned historical production calculations."""
    polar_columns = {
        f"wind_{quantity}_{height}"
        for height in ("10m", "100m")
        for quantity in ("speed", "direction")
    }
    if not polar_columns <= Nwp.all_weather_var_names():
        raise ValueError(
            "Historical polar-wind audit requires checkout "
            "5ac2e16a7088d13f2f7103f0385a58f54388bfba and matching inputs; "
            "the component pipeline cannot reproduce these historical probes."
        )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    stamps = pl.datetime_range(start, start + timedelta(days=3), interval="30m", eager=True)
    hours = np.arange(len(stamps)) / 2
    half_hour_midpoint = (hours - 0.25) % 24
    # Deliberately discrete half-hour means: these are the oracle, not point observations.
    truth = np.maximum(0, np.sin((half_hour_midpoint - 6) * np.pi / 12)) * 800
    radiation_parts = []
    metrics = []
    coordinates = pl.DataFrame(
        {"time_series_id": [0], "latitude": [52.0], "longitude": [0.0]}
    ).lazy()
    for width in (3, 6):
        indices = np.arange(0, len(stamps), width * 2)
        native_values = [None] + [
            float(truth[i - width * 2 + 1 : i + 1].mean()) for i in indices[1:]
        ]
        native = pl.DataFrame(
            {
                "time_series_id": [0] * len(indices),
                "ensemble_member": [0] * len(indices),
                # Six-hour windows begin after lead 144 in the ECMWF schedule.
                "nwp_init_time": [start - timedelta(hours=144 if width == 6 else 0)] * len(indices),
                "valid_time": stamps.gather(indices),
                RADIATION: native_values,
            }
        )
        reconstructed = (
            _upsample_nwp_to_half_hourly(native.lazy(), solar_coordinates=coordinates)
            .collect()
            .sort("valid_time")
        )
        recovered = reconstructed[RADIATION].to_numpy()
        interior = (hours > 24) & (hours <= 48)
        x = hours[interior] - 24
        original, actual = truth[interior], recovered[interior]
        centroid_shift = float(
            np.sum(x * actual) / actual.sum() - np.sum(x * original) / original.sum()
        )
        metrics.append(
            {
                "step_hours": width,
                "centroid_delay_hours": centroid_shift,
                "truth_energy_kwh_m2": float(original.sum() * 0.5 / 1000),
                "reconstructed_energy_kwh_m2": float(actual.sum() * 0.5 / 1000),
                "dark_positive_rows": int(((original == 0) & (actual > 1)).sum()),
                "eligible_rows": int(interior.sum()),
                "leading_null_rows": int(np.isnan(recovered[:1]).sum()),
            }
        )
        zero = native.with_columns(
            pl.when(pl.col(RADIATION).is_null()).then(None).otherwise(0.0).alias(RADIATION)
        )
        zero_result = (
            _upsample_nwp_to_half_hourly(zero.lazy(), solar_coordinates=coordinates)
            .collect()[RADIATION]
            .drop_nulls()
        )
        if (
            not (zero_result == 0).all()
            or not np.isnan(recovered[0])
            or not np.isfinite(recovered[1:]).all()
        ):
            raise ValueError("radiation dark/leading-null control failed")
        radiation_parts.append(
            pl.DataFrame(
                {"hour": x, "truth": original, "production": actual, "step_hours": width}
            ).unpivot(index=["hour", "step_hours"], variable_name="curve", value_name="irradiance")
        )
    direction_parts = []
    for left, right, case in ((350.0, 10.0, "Crossing north"), (100.0, 120.0, "No wrap control")):
        native = pl.DataFrame(
            {
                "time_series_id": [0, 0],
                "ensemble_member": [0, 0],
                "nwp_init_time": [start, start],
                "valid_time": [start, start + timedelta(hours=3)],
                "wind_direction_10m": [left, right],
            }
        )
        result = _upsample_nwp_to_half_hourly(native.lazy()).collect().sort("valid_time")
        elapsed = np.arange(result.height) / 2
        delta = (right - left + 180) % 360 - 180
        expected = (left + elapsed / 3 * delta) % 360
        discrepancy = np.abs((result["wind_direction_10m"].to_numpy() - expected + 180) % 360 - 180)
        if case == "No wrap control" and discrepancy.max() > 1e-5:
            raise ValueError("non-wrapping direction control failed")
        direction_parts.append(
            pl.DataFrame(
                {
                    "hour": elapsed,
                    "production": result["wind_direction_10m"],
                    "shortest_arc": expected,
                    "case": case,
                    "circular_discrepancy": discrepancy,
                }
            )
        )
    weekdays = (
        _apply_local_time_features(
            pl.DataFrame({"valid_time": [start + timedelta(days=i) for i in range(7)]}).lazy()
        )
        .collect()
        .select(
            weekday=pl.col("local_day_of_week").cast(pl.String),
            ordinal=pl.col("local_day_of_week").to_physical(),
            sine=pl.col("local_day_of_week_sin"),
            cosine=pl.col("local_day_of_week_cos"),
        )
        .sort("ordinal")
    )
    issue_time = start + timedelta(hours=9)
    lag_probe = pl.DataFrame(
        {
            "nwp_init_time": [start] * 3,
            "power_fcst_init_time": [issue_time] * 3,
            "valid_time": [issue_time + timedelta(hours=h) for h in (335.5, 336, 336.5)],
            LAG: [1.0] * 3,
        }
    )
    lag_result = _nullify_leaky_lags(lag_probe.lazy(), [LagFeature.from_str(LAG)]).collect()
    if lag_result[LAG].to_list() != [1.0, None, None]:
        raise ValueError("issuance-based lag boundary control failed")
    lag_result = lag_result.select(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds()
        / 3600,
        value=pl.col(LAG),
    )
    return {
        "radiation_probe": pl.concat(radiation_parts),
        "radiation_probe_metrics": pl.DataFrame(metrics),
        "direction_probe": pl.concat(direction_parts),
        "weekday_probe": weekdays,
        "lag_boundary_probe": lag_result,
    }


def lag_lookup(frame: pl.DataFrame, *, power: pl.DataFrame) -> pl.DataFrame:
    """Compare cached safe control-member lags with the current observation table."""
    checked = (
        frame.filter(pl.col("lead") < 336)
        .select(
            "time_series_id",
            "band",
            LAG,
            lookup_time=pl.col("valid_time") - pl.duration(hours=336),
        )
        .join(
            power.select(
                "time_series_id", pl.col("time").alias("lookup_time"), "power"
            ).with_columns(found=pl.lit(True)),
            on=["time_series_id", "lookup_time"],
            how="left",
        )
    )
    checked = checked.with_columns(
        status=pl.when(pl.col("found").is_null())
        .then(pl.lit("Observation key absent"))
        .when(~pl.col("power").is_finite().fill_null(False))
        .then(pl.lit("Observation value missing"))
        .when(missing(LAG))
        .then(pl.lit("Observation present; cached lag missing"))
        .when((pl.col(LAG) - pl.col("power")).abs() > 1e-4)
        .then(pl.lit("Observation present; value mismatch"))
        .otherwise(pl.lit("Matching value")),
    )
    return checked.group_by("band", "status").agg(
        rows=pl.len(),
        series=pl.col("time_series_id").unique(),
        lookup_keys=pl.struct("time_series_id", "lookup_time").unique(),
    )


def audit_frame(
    frame: pl.DataFrame, *, features: list[str], metadata: pl.DataFrame, power: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Compute the planned finite-population diagnostics for one control-member input chunk."""
    frame = add_leads(frame)
    finite = pl.col("temperature_2m").is_finite() & pl.col("wind_speed_10m").is_finite()
    outside = (pl.col("temperature_2m") > 10) | (pl.col("wind_speed_10m") * 3.6 < 4.8)
    reconstructed = frame.with_columns(STATIC_FEATURE_REGISTRY["windchill"].alias("recomputed"))
    windchill = population(frame, eligible=finite, affected=outside)
    formula = reconstructed.select(
        max_abs_formula_difference=(pl.col("windchill") - pl.col("recomputed")).abs().max(),
        outside_rows=outside.sum(),
        outside_above_air=((pl.col("windchill") > pl.col("temperature_2m")) & outside).sum(),
    )
    numeric = frame.select([pl.col(c).to_physical().cast(pl.Float64).alias(c) for c in features])
    ranges = pl.DataFrame(
        [
            {
                "feature": column,
                "rows": numeric.height,
                "missing": int(numeric.select(missing(column).sum()).item()),
                "nonfinite": int(
                    numeric.select((~pl.col(column).is_finite()).fill_null(False).sum()).item()
                ),
                "minimum": None
                if column.startswith("power_lag_")
                else numeric[column]
                .drop_nulls()
                .filter(numeric[column].drop_nulls().is_finite())
                .min(),
                "maximum": None
                if column.startswith("power_lag_")
                else numeric[column]
                .drop_nulls()
                .filter(numeric[column].drop_nulls().is_finite())
                .max(),
            }
            for column in features
        ]
    )
    precip = frame.group_by("nwp_init_time").agg(
        rows=pl.len(),
        missing=missing(PRECIPITATION).sum(),
        series=pl.col("time_series_id").n_unique(),
    )
    geometry = solar_geometry(frame, metadata=metadata)
    solar = frame.join(geometry, on=["time_series_id", "valid_time"], how="left").with_columns(
        grid=pl.when(pl.col("native"))
        .then(pl.lit("Native interval endpoint"))
        .otherwise(pl.lit("Inserted half-hour")),
        dark_native_interval=pl.when(pl.col("nwp_lead") <= 144)
        .then(pl.col("dark_3h"))
        .otherwise(pl.col("dark_6h")),
    )
    radiation = (
        solar.filter(pl.col(RADIATION).is_finite())
        .group_by("band", "grid")
        .agg(
            rows=pl.len(),
            dark_rows=pl.col("dark_half_hour").sum(),
            dark_positive=((pl.col(RADIATION) > 1) & pl.col("dark_half_hour")).sum(),
            affected_series=pl.col("time_series_id")
            .filter((pl.col(RADIATION) > 1) & pl.col("dark_half_hour"))
            .unique(),
            native_dark_interval_rows=(pl.col("native") & pl.col("dark_native_interval")).sum(),
            native_dark_interval_positive=(
                pl.col("native") & pl.col("dark_native_interval") & (pl.col(RADIATION) > 1)
            ).sum(),
        )
    )
    profile = (
        frame.filter(pl.col(RADIATION).is_finite())
        .with_columns(hour=pl.col("valid_time").dt.hour())
        .group_by("hour", "band")
        .agg(
            rows=pl.col(RADIATION).is_finite().sum(),
            radiation_sum=pl.col(RADIATION).sum(),
        )
    )
    chill_profile = (
        frame.filter(finite & pl.col("windchill").is_finite())
        .with_columns(
            temperature_bin=(pl.col("temperature_2m") / 2).floor() * 2,
            domain=pl.when(outside)
            .then(pl.lit("Outside formula domain"))
            .otherwise(pl.lit("Inside formula domain")),
        )
        .group_by("temperature_bin", "domain")
        .agg(
            rows=pl.len(),
            excess_sum=(pl.col("windchill") - pl.col("temperature_2m")).sum(),
        )
    )
    return {
        "ranges": ranges,
        "precip_runs": precip,
        "precip_population": population(
            frame, eligible=pl.lit(True), affected=missing(PRECIPITATION)
        ),
        "windchill": windchill,
        "formula": formula,
        "direction": pl.concat([direction_audit(frame, height=h) for h in (10, 100)]),
        "radiation": radiation,
        "radiation_profile": profile,
        "windchill_profile": chill_profile,
        "temperature": temperature_ranges(frame),
        "lag_missing": population(frame, eligible=pl.col("lead") < 336, affected=missing(LAG)),
        "lag_leakage": population(frame, eligible=pl.col("lead") >= 336, affected=~missing(LAG)),
        "lag_lookup": lag_lookup(frame, power=power),
    }


def write_report(
    tables: dict[str, pl.DataFrame], *, output: Path, manifest: dict[str, object]
) -> None:
    """Print all publishable diagnostic numbers from saved aggregate tables."""
    lines = [
        "# Production feature audit: generated tables",
        "",
        "Control member only; descriptive counts, no forecast scores.",
        "",
        "## Input manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
    ]
    for name, table in tables.items():
        lines += [
            f"## {name}",
            "",
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
        ]
        for row in table.iter_rows():
            values = [f"{v:.8g}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines))


def file_hash(path: Path) -> str:
    """Hash an input file incrementally without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_keys(frame: pl.DataFrame, *, observation: bool = False) -> None:
    """Reject null keys, duplicate keys, non-UTC timestamps and off-grid or negative leads."""
    keys = ["time_series_id", "time"] if observation else [*KEYS, "power_fcst_init_time"]
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError("source contains null required keys")
    identity = ["time_series_id", "time"] if observation else KEYS
    if frame.select(identity).is_duplicated().any():
        raise ValueError("source contains duplicate keys")
    timestamps = (
        ["time"] if observation else ["valid_time", "nwp_init_time", "power_fcst_init_time"]
    )
    for column in timestamps:
        if frame.schema[column] != pl.Datetime("us", "UTC"):
            raise ValueError("source timestamps must use microseconds and UTC")
        if frame.select(
            (
                (pl.col(column).dt.minute() % 30 != 0)
                | (pl.col(column).dt.second() != 0)
                | (pl.col(column).dt.microsecond() != 0)
            ).any()
        ).item():
            raise ValueError("source timestamp lies off the half-hour grid")
    if not observation:
        leads = add_leads(frame)
        if leads.select(((pl.col("lead") < 0) | (pl.col("nwp_lead") < 0)).any()).item():
            raise ValueError("source contains hindcast rows outside the study population")
        if leads.select(
            (
                (pl.col("nwp_init_time").dt.hour() % 6 != 0)
                | (pl.col("nwp_init_time").dt.minute() != 0)
            ).any()
        ).item():
            raise ValueError("source has unexpected NWP grid phase")


def combine_tables(*, name: str, parts: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine chunk aggregates, collapsing private identifier lists before saving."""
    combined = pl.concat(parts, how="vertical_relaxed")
    keys = [c for c, t in combined.schema.items() if t == pl.String or t.is_temporal()]
    if name == "direction":
        keys += ["height_m"]
    if name in {"radiation_profile", "windchill_profile"}:
        keys += ["hour" if name == "radiation_profile" else "temperature_bin"]
    if name == "ranges":
        table = combined.group_by(keys).agg(
            pl.col("rows", "missing", "nonfinite").sum(),
            pl.col("minimum").min(),
            pl.col("maximum").max(),
        )
    elif name == "formula":
        table = combined.group_by(keys).agg(
            pl.col("max_abs_formula_difference").max(),
            pl.col("outside_rows", "outside_above_air").sum(),
        )
    elif name == "temperature":
        table = combined.group_by(keys).agg(
            pl.col("days", "affected_days").sum(),
            *[
                (pl.col(c) * pl.col("days")).sum().truediv(pl.col("days").sum()).alias(c)
                for c in ("mean_range_3h", "mean_range_6h", "mean_range_loss")
            ],
        )
    elif name == "lag_lookup":
        table = combined.group_by(keys).agg(
            pl.col("rows").sum(),
            pl.col("series").explode().drop_nulls().n_unique(),
            pl.col("lookup_keys").explode().drop_nulls().n_unique().alias("unique_lookup_keys"),
        )
    else:
        numeric = [
            c
            for c, t in combined.schema.items()
            if c not in keys and c != "percent" and not isinstance(t, pl.List)
        ]
        # Private identifier lists are collapsed to counts before any output is written.
        series_cols = [c for c, t in combined.schema.items() if isinstance(t, pl.List)]
        table = combined.group_by(keys).agg(
            pl.col([c for c in numeric if c not in series_cols]).sum(),
            *[pl.col(c).explode().drop_nulls().n_unique().alias(c) for c in series_cols],
        )
        if "eligible" in table.columns:
            table = table.with_columns(percent=100 * pl.col("affected") / pl.col("eligible"))
    return table.sort(keys)


def load_power() -> tuple[pl.DataFrame, int]:
    """Collect one stable observation snapshot and reject duplicate keys."""
    power_version = DeltaTable(CLEANED_POWER_DELTA_URI).version()
    power = pl.DataFrame(scan_power().select("time_series_id", "time", "power").collect())
    if DeltaTable(CLEANED_POWER_DELTA_URI).version() != power_version:
        raise ValueError("power snapshot changed during observation collection; rerun the audit")
    validate_keys(power, observation=True)
    return power, power_version


def main() -> None:
    """Validate the forward-fold cache, run the audit and save aggregate evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = source_probes()
    if args.data_root.resolve() != Path(CLEANED_POWER_DELTA_URI).parents[1].resolve():
        raise ValueError("--data-root must match the power source; set DATA_PATH_INTERNAL")
    metadata = pl.read_parquet(args.data_root / "NGED" / "metadata.parquet")
    power, power_version = load_power()
    features = yaml.safe_load((PROJECT_ROOT / "conf" / "model" / "xgboost.yaml").read_text())[
        "model_params"
    ]["selected_features"]
    paths = [("Training", args.frames / "train.parquet")]
    paths += [("Validation", p) for p in sorted((args.frames / "val").glob("init=*.parquet"))]
    if len(paths) != 54:
        raise ValueError("expected one training cache and 53 validation runs")
    collected: dict[str, list[pl.DataFrame]] = {}
    inputs = []
    seen_runs: set[datetime] = set()
    for split, path in paths:
        columns = [*KEYS, "power_fcst_init_time", *features]
        all_members = pl.scan_parquet(path)
        total_rows = all_members.select(pl.len()).collect().item()
        frame = all_members.filter(pl.col("ensemble_member") == 0).select(columns).collect()
        validate_keys(frame)
        if split == "Validation":
            run = datetime.strptime(path.stem, "init=%Y-%m-%d").replace(tzinfo=UTC)
            if run in seen_runs or frame["nwp_init_time"].unique().to_list() != [run]:
                raise ValueError("validation files overlap runs or contain unexpected run keys")
            seen_runs.add(run)
        first, last = frame["valid_time"].min(), frame["valid_time"].max()
        lower = datetime(
            2024 if split == "Training" else 2025, 4 if split == "Training" else 7, 1, tzinfo=UTC
        )
        upper = datetime(2025 if split == "Training" else 2026, 7, 1, tzinfo=UTC)
        if not isinstance(first, datetime) or not isinstance(last, datetime):
            raise TypeError("cache timestamps must be datetimes")
        if first < lower or last >= upper:
            raise ValueError("cache dates lie outside the planned fold")
        inputs.append(
            {
                "split": split,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "all_member_rows": total_rows,
                "control_rows": frame.height,
                "series": frame["time_series_id"].n_unique(),
                "first_valid_time": str(first),
                "last_valid_time": str(last),
            }
        )
        for name, table in audit_frame(
            frame, features=features, metadata=metadata, power=power
        ).items():
            collected.setdefault(name, []).append(table.with_columns(split=pl.lit(split)))
        print(f"audited {split}: {path.name}, {frame.height:,} control rows", flush=True)
    tables.update(
        {name: combine_tables(name=name, parts=parts) for name, parts in collected.items()}
    )
    tables["precip_months"] = (
        tables["precip_runs"]
        .with_columns(month=pl.col("nwp_init_time").dt.strftime("%Y-%m"))
        .group_by("split", "month")
        .agg(pl.col("rows", "missing").sum())
        .with_columns(percent=100 * pl.col("missing") / pl.col("rows"))
        .sort("split", "month")
    )
    source_files = [
        inspect.getfile(f)
        for f in (
            _upsample_nwp_to_half_hourly,
            _nullify_leaky_lags,
            _apply_local_time_features,
            zenith,
        )
    ]
    source_files += [
        str(PROJECT_ROOT / "packages/ml_core/src/ml_core/features/_parsed_features.py")
    ]
    manifest = {
        "features": features,
        "inputs": inputs,
        "production_source_sha256": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files
        },
        "polars_version": pl.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_sha256": file_hash(args.data_root / "NGED" / "metadata.parquet"),
        "model_config_sha256": file_hash(PROJECT_ROOT / "conf/model/xgboost.yaml"),
        "power_delta_version": power_version,
        "power_source": str(Path(CLEANED_POWER_DELTA_URI).resolve()),
        "scope": (
            "Forward fold, member 0 only. Current source probes do not establish "
            "historical cache provenance."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(args.output / f"{name}.parquet")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(tables, output=args.output, manifest=manifest)
    print(f"wrote {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
