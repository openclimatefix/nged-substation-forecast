"""Check what each newly added solar product's per-site frame holds, before any build reads it.

One-off throwaway script for the second round of the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>. It reads the per-site
frames `extract_site_series.py` and `fetch_open_meteo_point.py` wrote for the six products the
second round adds, and writes `product_checks.md` under `sources.UPDATE_OUTPUT_DIR`.

**One check gates the study, and raises after the report is written: the timestamps must name the
hour ending at the label.** Each site's global irradiance is correlated with the sun's height at
offsets from its stamps (`studies.timestamp_checks`), and the peak must sit within 15 minutes of 30
minutes before the label. An hour-beginning label, or a snapshot read as a mean, fails. Read the
check over a whole year: over one summer month, afternoon cloud alone pulls the peak about 10
minutes early.

**Three tables are read by a person, not gated:**

- **Whether the published direct flux carries more than a separation model would**
  (`studies.served_column_checks.check_direct_is_not_a_separation_model`). A product that fails it
  still has a usable global flux; its split arm then measures a derived split, and the page has to
  say so.
- **Where the hour-to-hour jumps fall.** A forecast archive switches runs at fixed hours of the day,
  and the switch shows as a larger change in the clearness index between consecutive hours. The
  hours where the jumps peak give each product's run cycle, and so its served lead.
- **Steps in the served data.** Each product's monthly mean global irradiance against CAMS's, over
  the daylight hours both serve. A month far from the product's usual ratio marks a change of
  source or of model, which needs an era boundary or a shorter record.

Run it with `uv run python studies/beam_diffuse_split/check_new_products.py`, after the extractions
and the fetches. `--input-dir` reads `<dir>/<source>.parquet` instead of each product's own path,
and `--output` writes the report elsewhere, for a smoke test.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_dataset import CAMS_PATH, _pv_sites
from sources import UPDATE_OUTPUT_DIR, SourceType, point_output_path_for
from studies.served_column_checks import check_direct_is_not_a_separation_model
from studies.solar import extraterrestrial_horizontal, zenith
from studies.timestamp_checks import (
    HOUR_ENDING_OFFSET_MINUTES,
    best_offset_minutes,
    check_hour_ending,
    correlation_by_offset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("check_new_products")

NEW_SOURCES: Final[tuple[SourceType, ...]] = (
    "sarah-3",
    "icon-dream-eu",
    "ecmwf-ifs-hres",
    "arpege-europe",
    "dmi-harmonie-arome",
    "knmi-harmonie-arome",
)
"""The six products the second round adds to the sunshine study."""

JUMP_MIN_ELEVATION_DEGREES: Final[float] = 15.0
"""Both hours of a pair must have the sun this high for their change in clearness to count.

Near the horizon the clearness index is a ratio of two small numbers and its hour-to-hour change is
noise, which would swamp the run-switch signal in the early and late hours.
"""

STEP_TOLERANCE: Final[float] = 0.10
"""How far a month's ratio to CAMS may stray from the product's median before it is flagged."""


def _with_geometry(*, frame: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Add the sun's position and the clearness index at each hour's midpoint.

    Args:
        frame: One product's per-site frame, with `site`, `time`, `ghi_w_m2` and `bhi_w_m2`.
        sites: The roster, carrying `site`, `latitude` and `longitude`.

    Returns:
        `frame` with `solar_zenith_deg`, `extraterrestrial_horizontal_w_m2` and `clearness_index`.
    """
    parts: list[pl.DataFrame] = []
    for row in sites.iter_rows(named=True):
        rows = frame.filter(pl.col("site") == row["site"]).sort("time")
        midpoint = zenith(
            stamps=rows["time"].dt.offset_by("-30m"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        extraterrestrial = extraterrestrial_horizontal(stamps=rows["time"], zenith_deg=midpoint)
        parts.append(
            rows.with_columns(
                solar_zenith_deg=pl.Series(midpoint),
                extraterrestrial_horizontal_w_m2=pl.Series(extraterrestrial),
                clearness_index=pl.Series(
                    np.where(
                        extraterrestrial > 0.0,
                        rows["ghi_w_m2"].to_numpy() / np.maximum(extraterrestrial, 1e-9),
                        0.0,
                    )
                ),
            )
        )
    return pl.concat(parts)


def _timing_lines(*, frames: dict[SourceType, pl.DataFrame], sites: pl.DataFrame) -> list[str]:
    """Report where each product's correlation with the sun peaks, per site.

    Args:
        frames: Each product's per-site frame.
        sites: The roster, carrying `site`, `latitude` and `longitude`.

    Returns:
        Markdown lines, the last naming every failure, if any.
    """
    labels = sites["site"].to_list()
    lines = [
        "#### When each product's global irradiance tracks the sun best (minutes from the stamp)",
        "",
        f"A mean over the hour ending at the stamp peaks near {HOUR_ENDING_OFFSET_MINUTES:+d}.",
        "",
        "| Product | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    failures: list[str] = []
    for source, frame in frames.items():
        cells: list[str] = []
        for row in sites.iter_rows(named=True):
            rows = frame.filter(pl.col("site") == row["site"]).sort("time")
            correlations = correlation_by_offset(
                times=rows["time"],
                ghi=rows["ghi_w_m2"].to_numpy(),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            cells.append(f"{best_offset_minutes(correlations=correlations):+d}")
            try:
                check_hour_ending(correlations=correlations, name=f"{source} at site {row['site']}")
            except ValueError as failure:
                failures.append(str(failure))
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    lines += ["", *(f"- FAIL: {failure}" for failure in failures)]
    return lines


def _separation_lines(*, frames: dict[SourceType, pl.DataFrame]) -> list[str]:
    """Report whether each product's direct flux carries more than a separation model would.

    Args:
        frames: Each product's per-site frame, with geometry.

    Returns:
        Markdown lines, one per product.
    """
    lines = ["#### Is the published direct flux more than a separation model?", ""]
    for source, frame in frames.items():
        try:
            check_direct_is_not_a_separation_model(frame=frame)
        except ValueError as failure:
            lines.append(f"- {source}: fails: {failure}")
        else:
            lines.append(f"- {source}: passes")
    return lines


def _jump_lines(*, frames: dict[SourceType, pl.DataFrame]) -> list[str]:
    """Report the mean change in clearness index into each UTC hour, relative to the product's mean.

    Args:
        frames: Each product's per-site frame, with geometry.

    Returns:
        Markdown lines: one row per product, one column per hour.
    """
    by_product: dict[SourceType, pl.DataFrame] = {}
    for source, frame in frames.items():
        paired = frame.join(
            frame.select(
                "site",
                time=pl.col("time").dt.offset_by("1h"),
                previous_clearness=pl.col("clearness_index"),
                previous_zenith=pl.col("solar_zenith_deg"),
            ),
            on=["site", "time"],
        ).filter(
            (pl.col("solar_zenith_deg") < 90.0 - JUMP_MIN_ELEVATION_DEGREES)
            & (pl.col("previous_zenith") < 90.0 - JUMP_MIN_ELEVATION_DEGREES)
        )
        by_product[source] = (
            paired.group_by(hour=pl.col("time").dt.hour())
            .agg(jump=(pl.col("clearness_index") - pl.col("previous_clearness")).abs().mean())
            .with_columns(relative=pl.col("jump") / pl.col("jump").mean())
            .sort("hour")
        )
    hours = sorted({hour for table in by_product.values() for hour in table["hour"].to_list()})
    lines = [
        "#### Mean change in clearness index into each UTC hour, over the product's own mean",
        "",
        "A run switch shows as a value well above 1 at a fixed hour.",
        "",
        "| Product | " + " | ".join(f"{hour:02d}" for hour in hours) + " |",
        "|---" * (len(hours) + 1) + "|",
    ]
    for source, table in by_product.items():
        relative = dict(zip(table["hour"].to_list(), table["relative"].to_list(), strict=True))
        cells = [f"{relative[hour]:.2f}" if hour in relative else "" for hour in hours]
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return lines


def _step_lines(*, frames: dict[SourceType, pl.DataFrame], cams: pl.DataFrame) -> list[str]:
    """Report each month whose mean global irradiance strays from the product's usual ratio to CAMS.

    Args:
        frames: Each product's per-site frame, with geometry.
        cams: CAMS's per-site frame.

    Returns:
        Markdown lines: each product's median ratio and its flagged months.
    """
    lines = [
        "#### Monthly mean global irradiance over CAMS's, on the daylight hours both serve",
        "",
        f"A month is flagged where its ratio strays over {STEP_TOLERANCE:.0%} from the median.",
        "",
    ]
    reference = cams.select("site", "time", cams_ghi=pl.col("ghi_w_m2"))
    for source, frame in frames.items():
        monthly = (
            frame.filter(pl.col("solar_zenith_deg") < 90.0)
            .join(reference, on=["site", "time"])
            .group_by(month=pl.col("time").dt.strftime("%Y-%m"))
            .agg(ratio=pl.col("ghi_w_m2").mean() / pl.col("cams_ghi").mean())
            .sort("month")
        )
        if monthly.height == 0:
            lines.append(f"- {source}: no hours shared with CAMS")
            continue
        median = float(monthly.select(pl.col("ratio").median()).item())
        flagged = monthly.filter((pl.col("ratio") / median - 1.0).abs() > STEP_TOLERANCE)
        months = ", ".join(
            f"{row['month']} ({row['ratio']:.2f})" for row in flagged.iter_rows(named=True)
        )
        lines.append(
            f"- {source}: median ratio {median:.3f} over {monthly.height} months; "
            f"flagged: {months or 'none'}"
        )
    return lines


def main() -> int:
    """Run every check on every new product, write the report, and raise if a timing check failed.

    Returns:
        Zero, the process's exit status, once every timing check has passed.

    Raises:
        ValueError: If any product's timestamps do not name the hour ending at the label.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=NEW_SOURCES, default=list(NEW_SOURCES))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=UPDATE_OUTPUT_DIR / "product_checks.md")
    arguments = parser.parse_args()

    sites = _pv_sites().select("site", "latitude", "longitude").sort("site")
    frames: dict[SourceType, pl.DataFrame] = {}
    for source in arguments.sources:
        path = (
            arguments.input_dir / f"{source}.parquet"
            if arguments.input_dir is not None
            else point_output_path_for(source=source)
        )
        frame = pl.read_parquet(path).select("site", "time", "ghi_w_m2", "bhi_w_m2")
        frames[source] = _with_geometry(frame=frame, sites=sites)
        _LOG.info("%s: %d rows from %s", source, frame.height, path)

    timing = _timing_lines(frames=frames, sites=sites)
    lines = [
        *timing,
        "",
        *_separation_lines(frames=frames),
        "",
        *_jump_lines(frames=frames),
        "",
        *_step_lines(frames=frames, cams=pl.read_parquet(CAMS_PATH)),
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
    if any(line.startswith("- FAIL") for line in timing):
        msg = f"a gating check failed; see {arguments.output}"
        raise ValueError(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
