"""Audit immutable feature inputs without fitting models or scoring forecast improvements.

One validation run is materialised at a time. Public outputs contain aggregate feature changes,
domain flags and descriptive timing correlations; source manifests remain private. A shifted
radiation diagnostic tests the timing operator, not the power of a forecast-loss comparison.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl

KEYS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "nwp_init_time",
    "valid_time",
    "ensemble_member",
)
NON_FEATURES: Final[frozenset[str]] = frozenset((*KEYS, "power"))
STUDY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GROUPS: Final[tuple[str, ...]] = ("time_series_type", "lead_band", "source_era")
SHORTWAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
LONGWAVE: Final[str] = "downward_long_wave_radiation_flux_surface"
OFFSETS: Final[tuple[int, ...]] = tuple(range(-4, 4))


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(*, value: object) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError("Expected a nonnull UTC timestamp")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class FrameInput:
    """A verified private frame directory and its immutable declared inputs."""

    root: Path
    manifest: dict[str, Any]
    inventory: tuple[str, ...]


def _load(*, root: Path, label: str) -> FrameInput:
    manifest = json.loads((root / "manifest.json").read_text())
    context = json.loads((root / "context.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    if (
        manifest.get("complete") is not True
        or manifest["context"] != context
        or manifest["input_fingerprint"] != fingerprint
        or context["pipeline_tag"] != label
        or manifest["window"] != context["window"]
    ):
        raise ValueError("Frame completion, context, fingerprint or treatment mismatch")
    declared = manifest["files"]
    expected = {"train.parquet", "metadata.parquet", "capacity.parquet", "bounded_power.parquet"}
    validation = tuple(sorted(name for name in declared if name.startswith("val/")))
    if not validation or set(declared) != expected | set(validation):
        raise ValueError("Unexpected declared frame inventory")
    actual = {str(path.relative_to(root)) for path in root.glob("*.parquet")}
    actual.update(str(path.relative_to(root)) for path in (root / "val").rglob("*.*"))
    if actual != set(declared):
        raise ValueError("Actual frame inventory differs from the declared inputs")
    for name, digest in declared.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256(path=path) != digest:
            raise ValueError("Declared input path or SHA256 mismatch")
    for bound in ("train_start", "train_end", "val_start", "val_end"):
        stamp = datetime.fromisoformat(context["window"][bound])
        if stamp.tzinfo != UTC or stamp >= STUDY_END:
            raise ValueError("Source window is not UTC or exceeds the study outcome boundary")
    if datetime.fromisoformat(context["study_end_exclusive"]) != STUDY_END:
        raise ValueError("Unexpected study outcome boundary")
    return FrameInput(root=root, manifest=manifest, inventory=validation)


def _check_pair(*, old: FrameInput, other: FrameInput) -> None:
    if old.inventory != other.inventory or old.manifest["window"] != other.manifest["window"]:
        raise ValueError("Treatments have different validation runs or folds")
    for name in ("metadata.parquet", "capacity.parquet", "bounded_power.parquet"):
        if old.manifest["files"][name] != other.manifest["files"][name]:
            raise ValueError("Treatments have different metadata, capacity or observed power")


def _read_frame(*, source: FrameInput, name: str) -> pl.DataFrame:
    frame = pl.read_parquet(source.root / name).sort(*KEYS)
    if frame.is_empty() or frame.select(*KEYS).is_duplicated().any():
        raise ValueError("Empty frame or duplicate forecast keys")
    for column in ("power_fcst_init_time", "nwp_init_time", "valid_time"):
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise ValueError("Forecast timestamps must be UTC")
        if frame[column].null_count() or _stamp(value=frame[column].max()) >= STUDY_END:
            raise ValueError("Forecast timestamps are null or exceed the outcome boundary")
    window = source.manifest["window"]
    start, end = (datetime.fromisoformat(window[key]) for key in ("val_start", "val_end"))
    if frame.filter(
        (pl.col("valid_time") < start)
        | (pl.col("valid_time") > end)
        | (pl.col("valid_time") <= pl.col("power_fcst_init_time"))
        | (pl.col("nwp_init_time") > pl.col("power_fcst_init_time"))
    ).height:
        raise ValueError("Frame violates the validation or forecast issuance window")
    members = frame.group_by(*KEYS[:-1]).agg(values=pl.col("ensemble_member").sort())
    if members.filter(pl.col("values") != pl.lit(list(range(51)))).height:
        raise ValueError("Every forecast key must hold exactly members 0 through 50")
    return frame


def _groups(*, frame: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
    return frame.join(
        metadata, on="time_series_id", how="left", validate="m:1", maintain_order="left"
    ).with_columns(
        lead_band=pl.when(lead < 72)
        .then(pl.lit("<72h"))
        .when(lead < 264)
        .then(pl.lit("72..<264h"))
        .otherwise(pl.lit(">=264h")),
        source_era=pl.when(pl.col("nwp_init_time") < pl.datetime(2024, 11, 13, time_zone="UTC"))
        .then(pl.lit("before 2024-11-13"))
        .when(pl.col("nwp_init_time") < pl.datetime(2026, 5, 12, time_zone="UTC"))
        .then(pl.lit("2024-11-13 to 2026-05-11"))
        .otherwise(pl.lit("2026-05-12 onward")),
    )


def _changes(*, old: pl.DataFrame, new: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    summaries = []
    for feature in features:
        left, right = old[feature].cast(pl.Float64), new[feature].cast(pl.Float64)
        compared = old.select(*GROUPS).with_columns(old_value=left, new_value=right)
        a, b = pl.col("old_value"), pl.col("new_value")
        finite = a.is_finite().fill_null(False) & b.is_finite().fill_null(False)
        delta = (
            ((b - a + 180) % 360 - 180).abs()
            if feature in {"wind_direction_10m", "wind_direction_100m"}
            else (b - a).abs()
        )
        summaries.append(
            compared.group_by(*GROUPS)
            .agg(
                rows=pl.len().cast(pl.UInt64),
                old_null=a.is_null().sum().cast(pl.UInt64),
                new_null=b.is_null().sum().cast(pl.UInt64),
                old_nonfinite=(a.is_not_null() & ~a.is_finite()).sum().cast(pl.UInt64),
                new_nonfinite=(b.is_not_null() & ~b.is_finite()).sum().cast(pl.UInt64),
                paired_finite=finite.sum().cast(pl.UInt64),
                changed=(finite & (delta > 0)).sum().cast(pl.UInt64),
                null_changed=(a.is_null() != b.is_null()).sum().cast(pl.UInt64),
                abs_sum=delta.filter(finite).sum(),
                abs_max=delta.filter(finite).max(),
            )
            .with_columns(feature=pl.lit(feature))
        )
    return pl.concat(summaries)


def _domain(*, frame: pl.DataFrame) -> pl.DataFrame:
    flags = {
        "windchill_outside_domain": (pl.col("temperature_2m") > 10)
        | (pl.col("wind_speed_10m") * 3.6 < 4.8),
        "shortwave_negative": pl.col(SHORTWAVE) < 0,
        "shortwave_above_1500_screen": pl.col(SHORTWAVE) > 1500,
        "shortwave_night_midpoint_screen": (pl.col("solar_zenith") > 90) & (pl.col(SHORTWAVE) > 1),
        "longwave_negative": pl.col(LONGWAVE) < 0,
        "longwave_above_1000_screen": pl.col(LONGWAVE) > 1000,
    }
    required = {
        "windchill_outside_domain": ("temperature_2m", "wind_speed_10m"),
        "shortwave_negative": (SHORTWAVE,),
        "shortwave_above_1500_screen": (SHORTWAVE,),
        "shortwave_night_midpoint_screen": (SHORTWAVE, "solar_zenith"),
        "longwave_negative": (LONGWAVE,),
        "longwave_above_1000_screen": (LONGWAVE,),
    }
    flags = {
        name: pl.when(pl.all_horizontal(pl.col(required[name]).is_finite())).then(flag)
        for name, flag in flags.items()
    }
    return pl.concat(
        [
            frame.group_by(*GROUPS)
            .agg(
                rows=pl.len().cast(pl.UInt64),
                supported=flag.is_not_null().sum().cast(pl.UInt64),
                flagged=flag.fill_null(False).sum().cast(pl.UInt64),
            )
            .with_columns(flag=pl.lit(name))
            for name, flag in flags.items()
        ]
    )


def _timing(*, frame: pl.DataFrame, power: pl.DataFrame) -> pl.DataFrame | None:
    control = frame.filter(pl.col("ensemble_member") == 0).select(
        *KEYS[:-1], SHORTWAVE, "wind_speed_100m", "time_series_type"
    )
    base = control.join(
        power.rename({"time": "valid_time", "power": "observed"}),
        on=["time_series_id", "valid_time"],
        how="inner",
        validate="m:1",
    )
    join_keys = ["time_series_id", "power_fcst_init_time", "nwp_init_time", "valid_time"]
    delayed = control.select(*join_keys, SHORTWAVE).with_columns(
        valid_time=pl.col("valid_time").dt.offset_by("1h")
    )
    for offset in OFFSETS:
        shifted = (
            control.select(*join_keys, SHORTWAVE, "wind_speed_100m")
            .with_columns(valid_time=pl.col("valid_time").dt.offset_by(f"{-offset}h"))
            .rename({SHORTWAVE: f"solar_{offset}", "wind_speed_100m": f"wind_{offset}"})
        )
        base = base.join(shifted, on=join_keys, how="inner", validate="1:1")
        delayed_offset = delayed.with_columns(
            valid_time=pl.col("valid_time").dt.offset_by(f"{-offset}h")
        ).rename({SHORTWAVE: f"delayed_{offset}"})
        base = base.join(delayed_offset, on=join_keys, how="inner", validate="1:1")
    columns = [
        "observed",
        *(f"{kind}_{offset}" for kind in ("solar", "wind", "delayed") for offset in OFFSETS),
    ]
    base = base.filter(pl.all_horizontal(pl.col(columns).is_finite())).with_columns(
        day=pl.col("valid_time").dt.date()
    )
    day_keys = ["time_series_id", "power_fcst_init_time", "nwp_init_time", "day"]
    base = base.filter(pl.len().over(day_keys) >= 4)
    if base.is_empty():
        return None
    base = base.with_columns(
        [
            pl.when(pl.col("time_series_type") == "PV")
            .then(pl.col(column))
            .when(pl.col("time_series_type") == "Wind")
            .then(pl.col(column).pow(3))
            .otherwise(None)
            .alias(column)
            for column in columns
            if column != "observed"
        ]
    )
    base = base.with_columns(
        [
            (pl.col(column) - pl.col(column).mean().over(day_keys)).alias(column)
            for column in columns
        ]
    )
    parts = []
    for group, kind in (("PV", "solar"), ("PV", "delayed"), ("Wind", "wind")):
        selected = base.filter(pl.col("time_series_type") == group)
        if selected.is_empty():
            continue
        for offset in OFFSETS:
            x, y = pl.col(f"{kind}_{offset}"), pl.col("observed")
            parts.append(
                selected.group_by("time_series_id")
                .agg(
                    rows=pl.len().cast(pl.UInt64),
                    xy=(x * y).sum(),
                    xx=x.pow(2).sum(),
                    yy=y.pow(2).sum(),
                )
                .with_columns(
                    time_series_type=pl.lit(group),
                    offset_hours=pl.lit(offset),
                    construction=pl.lit("one_hour_delayed" if kind == "delayed" else "original"),
                )
            )
    return pl.concat(parts) if parts else None


def _combine_changes(*, parts: list[pl.DataFrame]) -> pl.DataFrame:
    totals = pl.concat(parts)
    keys = ["treatment", *GROUPS, "feature"]
    sums = [name for name in totals.columns if name not in {*keys, "abs_max"}]
    return (
        totals.group_by(keys)
        .agg(pl.col(sums).sum(), pl.col("abs_max").max())
        .with_columns(
            affected_share=pl.when(pl.col("paired_finite") > 0).then(
                pl.col("changed") / pl.col("paired_finite")
            ),
            mean_abs_change=pl.when(pl.col("paired_finite") > 0).then(
                pl.col("abs_sum") / pl.col("paired_finite")
            ),
            old_null_share=pl.col("old_null") / pl.col("rows"),
            new_null_share=pl.col("new_null") / pl.col("rows"),
            old_nonfinite_share=pl.col("old_nonfinite") / pl.col("rows"),
            new_nonfinite_share=pl.col("new_nonfinite") / pl.col("rows"),
        )
        .sort(keys)
    )


def _timing_summary(*, parts: list[pl.DataFrame]) -> pl.DataFrame | None:
    if not parts:
        return None
    sums = (
        pl.concat(parts)
        .group_by("time_series_id", "time_series_type", "offset_hours", "construction")
        .agg(pl.col("rows", "xy", "xx", "yy").sum())
        .with_columns(
            correlation=pl.when((pl.col("xx") > 0) & (pl.col("yy") > 0)).then(
                pl.col("xy") / (pl.col("xx") * pl.col("yy")).sqrt()
            )
        )
    )
    return (
        sums.group_by("time_series_type", "offset_hours", "construction")
        .agg(
            correlation=pl.col("correlation").mean(),
            series=pl.col("correlation").count(),
            rows=pl.col("rows").sum(),
        )
        .sort("time_series_type", "construction", "offset_hours")
    )


def _table(*, frame: pl.DataFrame) -> str:
    names = frame.columns
    rows = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    rows.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.iter_rows())
    return "\n".join(rows)


def _timing_report(*, timing: pl.DataFrame | None) -> str:
    if timing is None:
        return "Timing detector unresolved: no complete offset-independent support."
    lines = [
        "Offsets compare observed power at t to forecast weather at t+offset.",
        "Correlations de-mean each series within each valid day and run; series have equal weight.",
        _table(frame=timing.filter(pl.col("offset_hours") >= -3)),
    ]
    solar = timing.filter(
        pl.col("time_series_type") == "PV", pl.col("construction") == "original"
    ).sort("offset_hours")
    if solar.height == len(OFFSETS) and solar["correlation"].null_count() == 0:
        original = solar.filter(pl.col("offset_hours") >= -3)
        shifted = timing.filter(
            pl.col("time_series_type") == "PV",
            pl.col("construction") == "one_hour_delayed",
            pl.col("offset_hours") >= -3,
        )
        original_peak = original.filter(pl.col("correlation") == pl.col("correlation").max())
        shifted_peak = shifted.filter(pl.col("correlation") == pl.col("correlation").max())
        detected = (
            original_peak.height == 1
            and shifted_peak.height == 1
            and abs(original_peak["offset_hours"][0]) < 3
            and shifted_peak["offset_hours"][0] == original_peak["offset_hours"][0] + 1
        )
        lines.extend(
            [
                "One-hour delayed radiation is x_shift(t)=x(t-1h).",
                _table(frame=shifted),
                f"Shift detector passed: {detected}. Otherwise its result is unresolved.",
            ]
        )
    else:
        lines.append("Shift detector unresolved: incomplete solar support.")
    lines.append(
        "Autocorrelation, curtailment and cloud regimes confound the observed peak. A zero-lag "
        "peak does not establish causal timing accuracy. This operator check does not "
        "establish that a forecast-loss comparison can detect a timing fault."
    )
    return "\n\n".join(lines)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("old", "combined", "radiation", "wind"):
        parser.add_argument(f"--{label}", type=Path, required=label in {"old", "combined"})
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _observations(*, source: FrameInput, metadata: pl.DataFrame) -> pl.DataFrame:
    power = pl.read_parquet(source.root / "bounded_power.parquet").select(
        "time_series_id", "time", "power"
    )
    dtype = power.schema["time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise ValueError("Observed power timestamps must be UTC")
    if (
        power["time"].null_count()
        or power.filter(pl.col("time") >= STUDY_END).height
        or power.select("time_series_id", "time").is_duplicated().any()
        or power.filter(pl.col("power").is_not_null() & ~pl.col("power").is_finite()).height
    ):
        raise ValueError("Observed power violates uniqueness, finiteness or outcome boundaries")
    capacity = pl.read_parquet(source.root / "capacity.parquet")
    if (
        capacity["effective_capacity_mw"].null_count()
        or capacity.filter(
            ~pl.col("effective_capacity_mw").is_finite() | (pl.col("effective_capacity_mw") <= 0)
        ).height
        or capacity["time_series_id"].n_unique() != capacity.height
        or set(capacity["time_series_id"]) != set(metadata["time_series_id"])
        or metadata["time_series_type"].null_count()
        or metadata["time_series_id"].n_unique() != metadata.height
    ):
        raise ValueError("Invalid capacities or metadata population")
    return power


def main() -> None:
    """Verify inputs and write aggregate diagnostics into a new immutable output directory."""
    args = _arguments()
    if args.output.exists():
        raise ValueError("Output already exists; audit outputs are immutable")
    sources = {
        label: _load(root=getattr(args, label).resolve(), label=label)
        for label in ("old", "combined", "radiation", "wind")
        if getattr(args, label) is not None
    }
    old = sources["old"]
    for source in sources.values():
        _check_pair(old=old, other=source)
    metadata = pl.read_parquet(old.root / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    power = _observations(source=old, metadata=metadata)
    change_parts, domain_parts, timing_parts, bounds = [], [], [], []
    feature_lists: dict[str, list[str]] = {}
    for name in old.inventory:
        reference = _groups(frame=_read_frame(source=old, name=name), metadata=metadata)
        features = sorted(set(reference.columns) - NON_FEATURES - set(GROUPS))
        for label, source in sources.items():
            frame = (
                reference
                if label == "old"
                else _groups(frame=_read_frame(source=source, name=name), metadata=metadata)
            )
            actual_features = sorted(set(frame.columns) - NON_FEATURES - set(GROUPS))
            if actual_features != features or feature_lists.setdefault(label, features) != features:
                raise ValueError("Feature lists differ between treatments or runs")
            if not frame.select(*KEYS, "power").equals(reference.select(*KEYS, "power")):
                raise ValueError("Treatment forecast keys or embedded targets differ")
            change_parts.append(
                _changes(old=reference, new=frame, features=features).with_columns(
                    treatment=pl.lit(label)
                )
            )
            domain_parts.append(_domain(frame=frame).with_columns(treatment=pl.lit(label)))
            bounds.append(
                {
                    "treatment": label,
                    "rows": frame.height,
                    "series": frame["time_series_id"].n_unique(),
                    "members": frame["ensemble_member"].n_unique(),
                    "valid_start": _stamp(value=frame["valid_time"].min()).isoformat(),
                    "valid_end": _stamp(value=frame["valid_time"].max()).isoformat(),
                }
            )
            if label == "combined":
                timing = _timing(frame=frame, power=power)
                if timing is not None:
                    timing_parts.append(timing)
    changes = _combine_changes(parts=change_parts)
    domains = (
        pl.concat(domain_parts)
        .group_by("treatment", *GROUPS, "flag")
        .agg(pl.col("rows", "supported", "flagged").sum())
        .with_columns(flagged_share=pl.col("flagged") / pl.col("supported"))
        .sort("treatment", *GROUPS, "flag")
    )
    timing = _timing_summary(parts=timing_parts)
    args.output.mkdir(parents=True)
    changes.write_parquet(args.output / "feature_changes.parquet")
    domains.write_parquet(args.output / "domain_flags.parquet")
    if timing is not None:
        timing.write_parquet(args.output / "timing_correlations.parquet")
    report = [
        "# Fixed-pipeline input audit",
        (
            "These are input correctness diagnostics, not "
            "measurements of predictive score improvements. Counts use all 51 members; timing "
            "uses the control member only. Wind bearings use shortest angular distances in "
            "degrees. Other changes use exact inequality on paired finite Float32 values. "
            "Null and nonfinite counts are separate; the changed share excludes them."
        ),
        (
            "Radiation threshold and nighttime flags are screening diagnostics, not definitive "
            "physical impossibilities. Midpoint night can overlap a partly sunlit half-hour."
        ),
        (
            "Source eras use NWP initialisation. The November archive availability boundary is "
            "13 November 2024; the May cycle marker is 12 May 2026. These are descriptive markers, "
            "to verify against source provenance before interpreting era differences."
        ),
        "## Features",
        *[
            f"{label}: {len(columns)} columns: " + ", ".join(columns)
            for label, columns in feature_lists.items()
        ],
        "## Per-run aggregate bounds",
        _table(frame=pl.DataFrame(bounds)),
        "## Feature changes",
        _table(frame=changes),
        "## Domain flags",
        _table(frame=domains),
        "## Timing detector",
        _timing_report(timing=timing),
    ]
    (args.output / "report.md").write_text("\n\n".join(report) + "\n")
    for label, source in sources.items():
        if _load(root=source.root, label=label).manifest != source.manifest:
            raise ValueError("Frame manifest changed during audit")
    manifest = {
        "script_sha256": _sha256(path=Path(__file__)),
        "specification": {label: str(source.root) for label, source in sources.items()},
        "inputs": {
            label: {
                "manifest_sha256": _sha256(path=source.root / "manifest.json"),
                "files": source.manifest["files"],
            }
            for label, source in sources.items()
        },
        "files": {path.name: _sha256(path=path) for path in sorted(args.output.iterdir())},
    }
    manifest["specification_sha256"] = hashlib.sha256(
        json.dumps(manifest["specification"], sort_keys=True).encode()
    ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed immutable aggregate input audit; no models fitted")


if __name__ == "__main__":
    main()
