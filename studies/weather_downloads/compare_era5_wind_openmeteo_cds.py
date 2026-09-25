"""Compare Open-Meteo's ERA5 wind with native ERA5 wind from the Climate Data Store, at three sites.

One-off throwaway script. The past-weather studies use Open-Meteo's mirror of ERA5 wind
(`wind_era5.parquet`, km/h). This script checks that mirror against the native Copernicus fields
(`wind_native_cds.parquet`, m/s) that `fetch_era5_wind.py` downloaded, to find out whether
Open-Meteo applies a scaling factor, interpolates between cells, or shifts the timestamps. The
script is read-only analysis of local files: no request is made and no forecast model is fitted.

**Sites and cells.** The three wind sites, labelled `W1` to `W3`. Each has a 3 x 3 block of native
0.25 degree cells, identified by integer offsets `dy` (north positive) and `dx` (east positive)
from the cell nearest the site. No coordinate is read, printed, or written.

**Units and directions.** Native speed is `sqrt(u^2 + v^2) * 3.6` km/h, the unit Open-Meteo serves.
Native direction is the direction the wind blows from, `(270 - atan2(v, u) in degrees) mod 360`.
Direction differences are wrapped to plus or minus 180 degrees and are scored only on hours when
the native 100 m speed exceeds 5 km/h, because calm-wind direction is noise. Open-Meteo serves a
direction at 100 m only.

**Analyses.** Per site and cell: agreement statistics for 100 m speed, 10 m speed, and 100 m
direction. Per site: the cell that best matches Open-Meteo, a non-negative least-squares blend of
the 9 cells as a diagnostic of Open-Meteo's processing, the effect of shifting Open-Meteo by up to
2 hours, and the ratio of 100 m to 10 m speed on each side.

Run with `uv run python studies/weather_downloads/compare_era5_wind_openmeteo_cds.py`. The
`--site` and `--max-hours` options restrict the run to a small slice, and `--output-dir` redirects
the output away from the final folder, for a trial.
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from lineage import write_lineage_note
from paths import WEATHER_DOWNLOADS_DIR
from scipy.optimize import nnls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("compare_era5_wind_openmeteo_cds")

ERA5_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5"
OPEN_METEO_PATH: Final[Path] = ERA5_DIR / "wind_era5.parquet"
NATIVE_PATH: Final[Path] = ERA5_DIR / "wind_native_cds.parquet"
OUTPUT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR.parent / "era5_wind_compare"
METRES_PER_SECOND_TO_KM_PER_HOUR: Final[float] = 3.6
CALM_SPEED_KM_PER_HOUR: Final[float] = 5.0
"""Direction is scored only where the native 100 m speed exceeds this."""
OPEN_METEO_ROUNDING_KM_PER_HOUR: Final[float] = 0.1
SHIFT_HOURS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)
FACTOR_VISIBLE_THRESHOLD: Final[float] = 0.005
"""A ratio of means further than this from 1 (and beyond 3x the rounding noise floor) is reported
as a visible constant factor."""
HEIGHTS: Final[tuple[str, ...]] = ("100m", "10m")
CELL_KEYS: Final[list[str]] = ["site", "dy", "dx"]


def _load_native(*, site: str | None) -> pl.DataFrame:
    """Return native speed (km/h) and from-direction (degrees) per site, cell, and UTC hour."""
    frame = pl.read_parquet(NATIVE_PATH)
    if site is not None:
        frame = frame.filter(pl.col("site") == site)
    return frame.select(
        *CELL_KEYS,
        pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
        native_speed_100m=(
            pl.col("u100").cast(pl.Float64) ** 2 + pl.col("v100").cast(pl.Float64) ** 2
        ).sqrt()
        * METRES_PER_SECOND_TO_KM_PER_HOUR,
        native_speed_10m=(
            pl.col("u10").cast(pl.Float64) ** 2 + pl.col("v10").cast(pl.Float64) ** 2
        ).sqrt()
        * METRES_PER_SECOND_TO_KM_PER_HOUR,
        native_direction_100m=(270.0 - pl.arctan2("v100", "u100").degrees()).mod(360.0),
    )


def _load_open_meteo(*, site: str | None) -> pl.DataFrame:
    """Return Open-Meteo's ERA5 wind per site and UTC hour, with `open_meteo_` column names."""
    frame = pl.read_parquet(OPEN_METEO_PATH)
    if site is not None:
        frame = frame.filter(pl.col("site") == site)
    return frame.select(
        "site",
        "time",
        open_meteo_speed_100m="wind_speed_100m",
        open_meteo_speed_10m="wind_speed_10m",
        open_meteo_direction_100m="wind_direction_100m",
    )


def _shift_open_meteo(*, open_meteo: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Relabel each Open-Meteo hour as `hours` later, so it pairs with a later native hour."""
    return open_meteo.with_columns(pl.col("time") + pl.duration(hours=hours))


def _pair(*, open_meteo: pl.DataFrame, native: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Return matched hourly pairs, with Open-Meteo shifted by `hours` hours."""
    return native.join(
        _shift_open_meteo(open_meteo=open_meteo, hours=hours), on=["site", "time"], how="inner"
    ).sort(*CELL_KEYS, "time")


def _speed_metrics(*, pairs: pl.DataFrame, height: str, by: list[str]) -> pl.DataFrame:
    """Return agreement statistics of Open-Meteo speed against native speed, in km/h."""
    open_meteo, native = f"open_meteo_speed_{height}", f"native_speed_{height}"
    difference = pl.col(open_meteo) - pl.col(native)
    return pairs.group_by(by).agg(
        pl.len().alias("n"),
        difference.abs().mean().alias("mean_abs_diff"),
        difference.pow(2).mean().sqrt().alias("rms"),
        difference.mean().alias("bias"),
        difference.abs().max().alias("max_abs_diff"),
        pl.corr(open_meteo, native).alias("correlation"),
        (pl.col(open_meteo).mean() / pl.col(native).mean()).alias("ratio_of_means"),
        ((pl.col(open_meteo) * pl.col(native)).sum() / (pl.col(native) ** 2).sum()).alias(
            "slope_through_origin"
        ),
        pl.col(native).mean().alias("native_mean"),
    )


def _direction_metrics(*, pairs: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return circular agreement statistics of the 100 m direction, in degrees."""
    windy = pairs.filter(pl.col("native_speed_100m") > CALM_SPEED_KM_PER_HOUR)
    difference = (
        (pl.col("open_meteo_direction_100m") - pl.col("native_direction_100m") + 180.0) % 360.0
    ) - 180.0
    return windy.group_by(by).agg(
        pl.len().alias("n"),
        difference.abs().mean().alias("mean_abs_diff"),
        difference.mean().alias("bias"),
        difference.abs().max().alias("max_abs_diff"),
    )


def _summary(*, pairs: pl.DataFrame) -> pl.DataFrame:
    """Return one row per site, cell, and measured quantity."""
    speed_rows = [
        _speed_metrics(pairs=pairs, height=height, by=CELL_KEYS).with_columns(
            quantity=pl.lit(f"speed_{height}")
        )
        for height in HEIGHTS
    ]
    direction = _direction_metrics(pairs=pairs, by=CELL_KEYS).with_columns(
        quantity=pl.lit("direction_100m")
    )
    return (
        pl.concat([*speed_rows, direction], how="diagonal")
        .select(
            "site",
            "dy",
            "dx",
            "quantity",
            "n",
            "mean_abs_diff",
            "rms",
            "bias",
            "max_abs_diff",
            "correlation",
            "ratio_of_means",
            "slope_through_origin",
            "native_mean",
        )
        .sort("site", "quantity", "dy", "dx")
    )


def _best_cells(*, summary: pl.DataFrame) -> pl.DataFrame:
    """Return, per site and height, the cell with the lowest RMS and its margin over the second."""
    rows = []
    for height in HEIGHTS:
        speeds = summary.filter(pl.col("quantity") == f"speed_{height}")
        for site, group in speeds.group_by("site", maintain_order=True):
            ranked = group.sort("rms")
            best, second = ranked.row(0, named=True), ranked.row(1, named=True)
            rows.append(
                {
                    "site": site[0],
                    "height": height,
                    "best_dy": best["dy"],
                    "best_dx": best["dx"],
                    "best_rms": best["rms"],
                    "second_dy": second["dy"],
                    "second_dx": second["dx"],
                    "second_rms": second["rms"],
                    "rms_margin_ratio": second["rms"] / best["rms"],
                }
            )
    return pl.DataFrame(rows)


def _blend(*, pairs: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Fit Open-Meteo speed as a non-negative weighted sum of the 9 native cells, per site.

    This diagnoses Open-Meteo's processing (does it interpolate between cells?); it is not a
    forecast model. There is no intercept. The weights are unconstrained in sum, and the sum is
    reported, so a factor such as 0.98 shows as a weight sum below 1.
    """
    rows = []
    for height in HEIGHTS:
        wide_column = f"native_speed_{height}"
        for site in pairs["site"].unique(maintain_order=True):
            wide = (
                pairs.filter(pl.col("site") == site)
                .with_columns(cell=pl.format("d{}_{}", pl.col("dy"), pl.col("dx")))
                .pivot(
                    on="cell",
                    index=["time", f"open_meteo_speed_{height}"],
                    values=wide_column,
                )
                .drop_nulls()
            )
            cells = [c for c in wide.columns if c.startswith("d")]
            design = wide.select(cells).to_numpy()
            target = wide[f"open_meteo_speed_{height}"].to_numpy()
            weights, _ = nnls(design, target)
            fitted_rms = float(np.sqrt(np.mean((design @ weights - target) ** 2)))
            best_rms = best.filter((pl.col("site") == site) & (pl.col("height") == height))[
                "best_rms"
            ][0]
            rows.append(
                {
                    "site": site,
                    "height": height,
                    "n": wide.height,
                    "weight_sum": float(weights.sum()),
                    "blend_rms": fitted_rms,
                    "best_single_cell_rms": best_rms,
                    "weights": " ".join(
                        f"{c}={w:.3f}" for c, w in zip(cells, weights, strict=True)
                    ),
                }
            )
    return pl.DataFrame(rows)


def _timing(*, open_meteo: pl.DataFrame, native: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Return RMS of each height's speed at each Open-Meteo shift, at the best cell per site."""
    rows = []
    for height in HEIGHTS:
        for shift in SHIFT_HOURS:
            shifted = _pair(open_meteo=open_meteo, native=native, hours=shift)
            for row in best.filter(pl.col("height") == height).iter_rows(named=True):
                cell = shifted.filter(
                    (pl.col("site") == row["site"])
                    & (pl.col("dy") == row["best_dy"])
                    & (pl.col("dx") == row["best_dx"])
                )
                squared = (
                    pl.col(f"open_meteo_speed_{height}") - pl.col(f"native_speed_{height}")
                ) ** 2
                rows.append(
                    {
                        "site": row["site"],
                        "height": height,
                        "shift_hours": shift,
                        "n": cell.height,
                        "rms": cell.select(squared.mean().sqrt()).item(),
                    }
                )
    return pl.DataFrame(rows)


def _height_ratio(*, pairs: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Return the ratio of mean 100 m to mean 10 m speed on each side, at the 100 m best cell."""
    rows = []
    for row in best.filter(pl.col("height") == "100m").iter_rows(named=True):
        cell = pairs.filter(
            (pl.col("site") == row["site"])
            & (pl.col("dy") == row["best_dy"])
            & (pl.col("dx") == row["best_dx"])
        )
        open_meteo_ratio = cell.select(
            pl.col("open_meteo_speed_100m").mean() / pl.col("open_meteo_speed_10m").mean()
        ).item()
        native_ratio = cell.select(
            pl.col("native_speed_100m").mean() / pl.col("native_speed_10m").mean()
        ).item()
        rows.append(
            {
                "site": row["site"],
                "open_meteo_100m_over_10m": open_meteo_ratio,
                "native_100m_over_10m": native_ratio,
                "ratio_of_ratios": open_meteo_ratio / native_ratio,
            }
        )
    return pl.DataFrame(rows)


def _factor_verdicts(*, summary: pl.DataFrame, best: pl.DataFrame) -> list[str]:
    """Return one sentence per site and height on whether a constant factor is visible."""
    sentences = []
    for row in best.iter_rows(named=True):
        cell = summary.filter(
            (pl.col("site") == row["site"])
            & (pl.col("quantity") == f"speed_{row['height']}")
            & (pl.col("dy") == row["best_dy"])
            & (pl.col("dx") == row["best_dx"])
        ).row(0, named=True)
        floor = 0.5 * OPEN_METEO_ROUNDING_KM_PER_HOUR / cell["native_mean"]
        deviation = abs(cell["ratio_of_means"] - 1.0)
        visible = deviation > max(FACTOR_VISIBLE_THRESHOLD, 3.0 * floor)
        sentences.append(
            f"{row['site']} {row['height']}: ratio of means {cell['ratio_of_means']:.4f}, "
            f"rounding noise floor {floor:.5f}, "
            f"{'a constant factor IS visible' if visible else 'no constant factor visible'}."
        )
    return sentences


def _markdown_table(*, frame: pl.DataFrame) -> str:
    """Render a small frame as a Markdown table, floats to four significant figures."""
    header = (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join("---" for _ in frame.columns)
        + " |"
    )
    lines = [
        "| "
        + " | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row)
        + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, *lines])


def _write_readme(
    *,
    output_dir: Path,
    span: tuple[str, str],
    best: pl.DataFrame,
    blend: pl.DataFrame,
    timing: pl.DataFrame,
    height_ratio: pl.DataFrame,
    verdicts: list[str],
    summary: pl.DataFrame,
) -> None:
    """Write the README that explains every file and states the findings."""
    nearest = summary.filter(pl.col("quantity") != "direction_100m").filter(
        (pl.col("dy") == 0) & (pl.col("dx") == 0)
    )
    direction = summary.filter(pl.col("quantity") == "direction_100m").filter(
        (pl.col("dy") == 0) & (pl.col("dx") == 0)
    )
    text = f"""# Open-Meteo's ERA5 wind against native ERA5 wind

One-off throwaway comparison written by
`studies/weather_downloads/compare_era5_wind_openmeteo_cds.py`. The results are provisional until
the script has had its review. Sites are the anonymous labels `W1` to `W3`; cells are integer
offsets `dy` (north positive) and `dx` (east positive) from the native ERA5 cell nearest the site.

**`pairs.parquet` holds hourly wind series per anonymised generator and must not be published.**
Nothing in this folder carries a coordinate.

Overlap compared: {span[0]} to {span[1]} (UTC hours). Speeds are km/h. Open-Meteo minus native is
the sign convention for every difference and bias.

## Files

- `pairs.parquet`: matched hourly pairs at zero shift, one row per site, cell, and hour, with
  `open_meteo_*` and `native_*` speeds (km/h) and 100 m from-directions (degrees).
- `summary.csv`: per site, cell, and quantity (`speed_100m`, `speed_10m`, `direction_100m`): count,
  mean absolute difference, RMS, bias, maximum absolute difference, correlation, ratio of means, and
  the slope of a fit through the origin. Direction rows count only hours with native 100 m speed
  above {CALM_SPEED_KM_PER_HOUR:g} km/h, use circular differences wrapped to plus or minus 180
  degrees, and leave the ratio and correlation columns empty.
- `best_cells.csv`, `blend.csv`, `timing.csv`, `height_ratio.csv`: the tables below.
- `lineage.json`: the inputs and the run time.

## Is a constant factor visible?

Open-Meteo rounds speeds to {OPEN_METEO_ROUNDING_KM_PER_HOUR:g} km/h, so a ratio of means carries a
rounding noise floor of at most 0.05 km/h divided by the mean native speed. A ratio further than
{FACTOR_VISIBLE_THRESHOLD:g} from 1 and beyond 3 times that floor is called a visible factor. Ratios
are at the best-matching cell.

{chr(10).join(f"- {sentence}" for sentence in verdicts)}

## Best-matching cell per site and height (lowest RMS)

`rms_margin_ratio` is the second-best RMS divided by the best RMS; near 1 means no clear winner.

{_markdown_table(frame=best)}

## Non-negative least-squares blend of the 9 cells

A diagnostic of Open-Meteo's processing, not a forecast model. Weights are non-negative with no
intercept; `weight_sum` far from 1 would show a scaling factor, and weight spread over several
cells would show interpolation. A bilinear blend of the 4 nearest cells is a special case.

{_markdown_table(frame=blend.drop("weights"))}

Weights (cell `d<dy>_<dx>`):

{chr(10).join(f"- {r['site']} {r['height']}: {r['weights']}" for r in blend.iter_rows(named=True))}

## Timing: RMS by Open-Meteo shift, at the best cell

A shift of `k` hours pairs Open-Meteo's value at hour `t` with the native value at `t + k`. ERA5
winds are instants, so the lowest RMS should be at 0.

{_markdown_table(frame=timing)}

## Ratio of 100 m to 10 m mean speed

A different ratio on the two sides would show a height-scaling difference.

{_markdown_table(frame=height_ratio)}

## Nearest cell (0, 0), speed statistics

{_markdown_table(frame=nearest.drop("dy", "dx"))}

## Nearest cell (0, 0), 100 m direction (native speed above {CALM_SPEED_KM_PER_HOUR:g} km/h)

{_markdown_table(frame=direction.select("site", "n", "mean_abs_diff", "bias", "max_abs_diff"))}
"""
    (output_dir / "README.md").write_text(text)


def main() -> int:
    """Load both sides, run every analysis, and write the output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", help="Restrict to one site label, for a trial.")
    parser.add_argument("--max-hours", type=int, help="Keep only the first N overlap hours.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    open_meteo = _load_open_meteo(site=arguments.site)
    native = _load_native(site=arguments.site)
    first = max(
        open_meteo.select(pl.col("time").min()).item(), native.select(pl.col("time").min()).item()
    )
    last = min(
        open_meteo.select(pl.col("time").max()).item(), native.select(pl.col("time").max()).item()
    )
    open_meteo = open_meteo.filter(pl.col("time").is_between(first, last))
    native = native.filter(pl.col("time").is_between(first, last))
    if arguments.max_hours is not None:
        cutoff = first + timedelta(hours=arguments.max_hours)
        open_meteo = open_meteo.filter(pl.col("time") < cutoff)
    pairs = _pair(open_meteo=open_meteo, native=native, hours=0)
    if pairs.is_empty():
        _LOG.error("no matched hours between the two sources")
        return 1
    _LOG.info("%d matched pairs", pairs.height)

    summary = _summary(pairs=pairs)
    best = _best_cells(summary=summary)
    blend = _blend(pairs=pairs, best=best)
    timing = _timing(open_meteo=open_meteo, native=native, best=best)
    height_ratio = _height_ratio(pairs=pairs, best=best)
    verdicts = _factor_verdicts(summary=summary, best=best)

    pairs.write_parquet(output_dir / "pairs.parquet")
    summary.write_csv(output_dir / "summary.csv")
    best.write_csv(output_dir / "best_cells.csv")
    blend.write_csv(output_dir / "blend.csv")
    timing.write_csv(output_dir / "timing.csv")
    height_ratio.write_csv(output_dir / "height_ratio.csv")
    span = (str(pairs["time"].min()), str(pairs["time"].max()))
    _write_readme(
        output_dir=output_dir,
        span=span,
        best=best,
        blend=blend,
        timing=timing,
        height_ratio=height_ratio,
        verdicts=verdicts,
        summary=summary,
    )
    write_lineage_note(
        product_dir=output_dir,
        source_address="local files: ERA5/wind_era5.parquet and ERA5/wind_native_cds.parquet",
        request_description=(
            "Open-Meteo's ERA5 wind (archive API, model era5, cell_selection=land) compared with "
            "native Copernicus ERA5 wind at a 3 x 3 block of cells per site, on the overlap."
        ),
        variables=["wind_speed_100m", "wind_speed_10m", "wind_direction_100m", "u100", "v100"],
        extra={"span": span, "pairs": pairs.height, "sites": pairs["site"].unique().to_list()},
    )
    _LOG.info("wrote outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
