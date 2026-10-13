"""Measure half-hour weather resampling on deterministic synthetic bulk inputs.

Run this script under each checkout's ml_core import path. Each repetition uses a fresh process
and collects all output columns. Input construction is excluded from elapsed time; peak RSS
includes imports, native input, query execution, and collected output. This is an in-memory
resampler benchmark, not an end-to-end CV or archive-I/O benchmark.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Literal

import polars as pl
from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS, NwpModelId
from ml_core.features._nwp import _upsample_nwp_to_half_hourly


def measure(*, sites: int, runs: int, members: int) -> dict[str, object]:
    """Collect one complete resample and record elapsed time, peak RSS, and output checks."""
    coordinates = pl.DataFrame(
        {
            "time_series_id": range(sites),
            "latitude": [50.0 + site / max(sites - 1, 1) * 8 for site in range(sites)],
            "longitude": [-4.0] * sites,
        }
    )
    groups = (
        coordinates.select("time_series_id")
        .join(
            pl.DataFrame(
                {
                    "nwp_init_time": [
                        datetime(2026, 3, 21, tzinfo=UTC) + timedelta(days=7 * run)
                        for run in range(runs)
                    ]
                }
            ),
            how="cross",
        )
        .join(pl.DataFrame({"ensemble_member": range(members)}), how="cross")
        .join(pl.DataFrame({"lead": ECMWF_ENS_LEAD_TIME_HOURS}), how="cross")
    )
    native = (
        groups.with_columns(
            valid_time=pl.col("nwp_init_time") + pl.duration(hours=pl.col("lead")),
            nwp_model_id=pl.lit(NwpModelId.ECMWF_ENS_0_25_degree.name),
            downward_short_wave_radiation_flux_surface=pl.when(pl.col("lead") == 0)
            .then(None)
            .otherwise(
                ((pl.col("lead") % 24 - 6) * 3.141592653589793 / 12).sin().clip(0, 1)
                * (300 + pl.col("ensemble_member"))
            ),
            downward_long_wave_radiation_flux_surface=pl.when(pl.col("lead") == 0)
            .then(None)
            .otherwise(310 + 20 * (pl.col("lead") * 3.141592653589793 / 12).cos()),
            temperature_2m=10 + (pl.col("lead") * 3.141592653589793 / 12).sin() * 5,
        )
        .drop("lead")
        .cast(
            {
                "downward_short_wave_radiation_flux_surface": pl.Float32,
                "downward_long_wave_radiation_flux_surface": pl.Float32,
                "temperature_2m": pl.Float32,
            }
        )
        .sample(fraction=1, shuffle=True, seed=526)
    )
    kwargs: dict[Literal["solar_coordinates"], pl.LazyFrame] = (
        {"solar_coordinates": coordinates.lazy()}
        if "solar_coordinates" in inspect.signature(_upsample_nwp_to_half_hourly).parameters
        else {}
    )
    started = perf_counter()
    query = _upsample_nwp_to_half_hourly(native.lazy(), **kwargs)
    result = query.collect()
    elapsed = perf_counter() - started
    assert result.height == sites * runs * members * 721
    null_half_hours = 1 if kwargs else 6
    for column in (
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
    ):
        missing = result.filter(pl.col(column).is_null())
        assert missing.height == sites * runs * members * null_half_hours
        assert (missing["valid_time"] - missing["nwp_init_time"]).max() == timedelta(
            minutes=30 * (null_half_hours - 1)
        )
        assert result[column].drop_nulls().is_finite().all()
    source = Path(inspect.getfile(_upsample_nwp_to_half_hourly))
    sources = [source, source.with_name("_radiation.py")]
    return {
        "sites": sites,
        "runs": runs,
        "members": members,
        "native_rows": native.height,
        "output_rows": result.height,
        "seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "polars_version": pl.__version__,
        "threads": pl.thread_pool_size(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if path.exists()
        },
        "source_directory": str(source.parent),
    }


def main() -> None:
    """Run fresh-process repetitions and save raw measurements with median summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--members", type=int, default=51)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--label", default="current")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--compare-sources", nargs="+", metavar="LABEL=ML_CORE_SOURCE")
    parser.add_argument("--summarize-directory", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("studies/radiation_resampling/benchmark_report.md")
    )
    args = parser.parse_args()
    if args.summarize_directory is not None:
        summarize(directory=args.summarize_directory, report=args.report)
        return
    if args.sites is None or args.runs is None:
        parser.error("--sites and --runs are required when measuring")
    if args.worker:
        print(json.dumps(measure(sites=args.sites, runs=args.runs, members=args.members)))
        return
    sources = (
        [
            (label, str(Path(source).resolve()))
            for label, source in (entry.split("=", maxsplit=1) for entry in args.compare_sources)
        ]
        if args.compare_sources
        else [(args.label, None)]
    )
    measurements = []
    for repeat in range(args.repeats):
        offset = repeat % len(sources)
        for label, source in sources[offset:] + sources[:offset]:
            environment = {**os.environ, "POLARS_MAX_THREADS": str(args.threads)}
            if source is not None:
                environment["PYTHONPATH"] = source
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--sites",
                    str(args.sites),
                    "--runs",
                    str(args.runs),
                    "--members",
                    str(args.members),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            measurement = json.loads(completed.stdout)
            if source is not None:
                assert Path(measurement["source_directory"]).resolve() == (
                    Path(source) / "ml_core/features"
                )
            measurements.append({"label": label, **measurement})
    report = {
        "label": args.label,
        "measurements": measurements,
        "summaries": {
            label: {
                "median_seconds": statistics.median(
                    row["seconds"] for row in measurements if row["label"] == label
                ),
                "median_peak_rss_mib": statistics.median(
                    row["peak_rss_mib"] for row in measurements if row["label"] == label
                ),
            }
            for label, _ in sources
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


def summarize(*, directory: Path, report: Path) -> None:
    """Write a benchmark table and portable raw measurements from interleaved runs."""
    measurements = []
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        for row in result["measurements"]:
            row.pop("source_directory")
            measurements.append({"label": result["label"], **row})
    cpu = next(
        line.split(":", maxsplit=1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    lines = [
        "# Radiation resampler benchmark",
        "",
        "**Native-window neighbour lookup reduces the cost of interval-aware reconstruction.**",
        "The comparison uses identical shuffled synthetic ECMWF inputs, all 51 members, and",
        "the complete 15-day horizon (85 native steps and 721 output timestamps per trajectory).",
        "Each variant has three fresh-process repetitions, with execution order rotated between",
        "variants. Eight Polars threads are fixed for every worker. No tests ran concurrently.",
        "",
        (
            f"Host CPU: {cpu}. Python {measurements[0]['python']}; "
            f"Polars {measurements[0]['polars_version']}."
        ),
        "",
        "**Timing includes query construction and collection of every output column.** Input",
        "construction is excluded. Peak RSS is the whole process, including imports, input/output",
        "frames, and result assertions; it is not incremental resampler memory. The inputs contain",
        "short-wave, long-wave, and temperature. Archive I/O, forecast joins, lags, rolling",
        "features, and forecasting-model fitting are excluded. No full-CV timing or forecast",
        "skill is claimed.",
        "",
        (
            "| Sites × runs | Output rows | Variant | Median time (s) | Range (s) | "
            "Median peak RSS (MiB) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for sites, runs in sorted({(row["sites"], row["runs"]) for row in measurements}):
        for label in ("point", "before", "after"):
            rows = [
                row
                for row in measurements
                if (row["sites"], row["runs"], row["label"]) == (sites, runs, label)
            ]
            assert len(rows) == 3
            seconds = [row["seconds"] for row in rows]
            memory = [row["peak_rss_mib"] for row in rows]
            lines.append(
                f"| {sites} × {runs} | {rows[0]['output_rows']:,} | {label} | "
                f"{statistics.median(seconds):.3f} | {min(seconds):.3f}–{max(seconds):.3f} | "
                f"{statistics.median(memory):.1f} |"
            )
    lines.extend(
        [
            "",
            "**The variants compare different processing policies on the same native means.**",
            "`point` uses main at `ce7285e4`; `before` uses the radiation PR at `99a40cf5` with",
            "full-target sorts and as-of joins. `after` uses native-window neighbour fills, the",
            "20 W/m² conditioning floor, and lazy schedule validation. All helper hashes and",
            "individual measurements are retained in the accompanying JSON. The comparison does",
            "not isolate the lookup change from the other review fixes. Results at larger scales",
            "must be measured; these two workloads do not establish linear scaling or a",
            "memory cap.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(
        json.dumps({"cpu_model": cpu, "measurements": measurements}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
