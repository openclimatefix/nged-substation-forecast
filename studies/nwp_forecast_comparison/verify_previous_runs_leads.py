"""Check Open-Meteo's Previous Runs run-selection rule and each product's run-switch pattern.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>.

**V1 (a gate).** `_previous_dayN` is not documented as coming from a specific run; this checks the
working assumption that it comes from the freshest run initialised at least `24N` hours before the
hour, by comparing Open-Meteo's served GFS `_previous_dayN` values against the raw Dynamical.org GFS
archive on disk in `data/studies/weather/GFS_window_2025-07-01_2025-07-02/`, for a range of
candidate runs `24N + k` hours before the hour, `k` from 0 to 12. The gate is that the mean absolute
difference is lowest at `k = 0`, for `N = 1` and `N = 2`, which is what "the freshest run at least
`24N` hours old" predicts. **A first pass compared each site's served series with the plain average
over the extract's 9 grid cells, and the resulting spatial-sampling noise (about 3 km/h) swamped the
signal at N = 1.** This version instead scores, for each site and each candidate offset, every one
of the extract's 9 grid cells against that site's own series, and keeps the cell with the lowest
mean absolute error — no roster coordinate is read, and every offset gets exactly the same freedom
to pick its best-fitting cell, so a real difference in lead accuracy between offsets survives while
the noise from not knowing which cell a site truly falls in does not. Wind speed at 100 m compares
directly; shortwave radiation additionally checks, at hours divisible by 6 (the only hours where the
two conventions pick a different run), whether Open-Meteo selects the run by the hour's label or by
the hour's start, with the same per-site best-cell scoring.

**V1b.** For every Previous Runs product, per UTC hour of day, the mean absolute second difference
of the `_previous_day1` series (100 m wind and 2 m temperature), divided by that statistic's own
mean over all 24 hours. A run switch is a discontinuity a smooth diurnal signal does not otherwise
have, so the hours where a fresh run cuts in carry a materially higher ratio than their neighbours.
Each product's cycle is read as the `n` in `{1, 3, 6}` whose switch hours (`h mod n == 0`, `n = 1`
excluded as the fallback) have the highest mean ratio relative to the other hours, and reported as
"no clear signature" where the best ratio is below 1.3.

**V3.** Reads each product's timestamp convention (instantaneous snapshot or hour-ending mean) from
the "best fit" line already measured and recorded, at fetch time, in that product's own
`previous_runs/lineage.json`, and states the height convention already established in
`studies.study` project documentation (ICON's served "100 m" wind is its native 120 m wind rescaled
by about 0.98).

No metered generator's name, identifier or coordinate appears anywhere in this script or its output:
every weather value already carries only the anonymised `site` label (`A`-`F`, `W1`-`W3`) that the
fetch scripts wrote.

Run it with `uv run python studies/nwp_forecast_comparison/verify_previous_runs_leads.py
--output-dir <dir>`.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from contracts.settings import PROJECT_ROOT

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

PRODUCT_DIRS: Final[dict[str, str]] = {
    "UKV": "UKV",
    "ICON-D2": "ICON-D2",
    "ICON-EU": "ICON-EU",
    "ICON global": "ICON-GLOBAL",
    "IFS 0.25°": "ECMWF-IFS-025",
    "GFS": "GFS-SEAMLESS",
    "ARPEGE Europe": "ARPEGE-EUROPE",
    "AROME France": "AROME-FRANCE",
    "KNMI HARMONIE-AROME": "KNMI-HARMONIE-AROME",
    "DMI HARMONIE-AROME": "DMI-HARMONIE-AROME",
}
"""Every Previous Runs product this study reads, to its `data/studies/weather/<dir>/` directory."""

GFS_WINDOW_DIR_NAME: Final[str] = "GFS_window_2025-07-01_2025-07-02"
"""The raw Dynamical.org GFS whole-run extract V1 checks Open-Meteo's GFS against."""

V1_OFFSETS_K: Final[tuple[int, ...]] = tuple(range(13))
"""Candidate offsets, in hours, tried on top of `24N` hours before the target hour."""

V1_DAYS_N: Final[tuple[int, ...]] = (1, 2)
"""The `previous_dayN` offsets V1's gate is checked on."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    Every worktree's `data/` would otherwise be empty: the downloads under it run to tens of
    gigabytes and are shared by every branch. A linked worktree marks itself by making `.git` a
    file holding `gitdir: <main>/.git/worktrees/<name>`, which names the main checkout two levels
    up; a plain checkout has no such file and resolves to itself.

    Returns:
        The directory holding `studies/`, `NGED/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def _weather_dir() -> Path:
    """Return `data/studies/weather/`, where every downloaded weather product lives."""
    return _repo_data_dir() / "studies" / "weather"


def _floor6(moment: datetime) -> datetime:
    """Floor a UTC timestamp to the 6-hourly run boundary at or before it (00, 06, 12, 18)."""
    return moment.replace(hour=(moment.hour // 6) * 6, minute=0, second=0, microsecond=0)


def _gfs_hourly_mean_since_reset(*, lead: int, value_at_lead: dict[int, float]) -> float | None:
    """Recover one hour's mean radiation from Dynamical.org GFS's since-reset accumulation.

    GFS resets its radiation accumulation every 6 hours of lead (at lead 0, 6, 12, ...), so the
    value at lead `L` is the mean since the most recent reset strictly before `L`, not the mean over
    the single hour ending at `L`. The hour ending at `L` is recovered by differencing against the
    value one lead hour earlier, weighted by how many hours each has accumulated since the reset.

    Args:
        lead: The lead, in whole hours, of the hour's end.
        value_at_lead: Each lead in the same run to its raw since-reset mean, in W m⁻².

    Returns:
        The mean over the hour ending at `lead`, in W m⁻², or `None` if a lead this needs is
        missing.
    """
    boundary = 6 * ((lead - 1) // 6)
    within = lead - boundary
    at_lead = value_at_lead.get(lead)
    if at_lead is None:
        return None
    if within == 1:
        return at_lead
    before = value_at_lead.get(lead - 1)
    if before is None:
        return None
    return within * at_lead - (within - 1) * before


GridCellByRun = dict[datetime, dict[int, dict[str, dict[int, float]]]]
"""Each run's initialisation time to each grid cell id to `{field: {lead: value}}`."""


def _dynamical_gfs_by_run() -> GridCellByRun:
    """Load the raw Dynamical.org GFS window extract, one grid cell at a time, by run.

    The cell axis is kept rather than averaged away, so V1 can score each of the extract's 9 grid
    cells separately against a site's own series (see `_per_site_best_cell_mad`).

    Returns:
        Each run's initialisation time to each grid cell id (`lat_index * 10 + lon_index`, never a
        real coordinate) to `{"wind_speed_100m_kmh": {lead: value}, "ghi_raw": {lead: value}}`.
    """
    path = _weather_dir() / GFS_WINDOW_DIR_NAME / "GFS.parquet"
    frame = (
        pl.read_parquet(path)
        .with_columns(
            lead_hours=(pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32),
            wind_speed_100m_kmh=(
                (pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2) ** 0.5 * 3.6
            ),
            cell=pl.col("lat_index") * 10 + pl.col("lon_index"),
        )
        .filter(pl.col("lead_hours") <= 120)
        .group_by("init_time", "lead_hours", "cell")
        .agg(
            pl.col("wind_speed_100m_kmh").mean(),
            pl.col("downward_short_wave_radiation_flux_surface").mean().alias("ghi_raw"),
        )
    )
    by_run: GridCellByRun = {}
    for row in frame.iter_rows(named=True):
        run = row["init_time"].replace(tzinfo=UTC)
        cell_entry = by_run.setdefault(run, {}).setdefault(
            row["cell"], {"wind_speed_100m_kmh": {}, "ghi_raw": {}}
        )
        cell_entry["wind_speed_100m_kmh"][row["lead_hours"]] = row["wind_speed_100m_kmh"]
        cell_entry["ghi_raw"][row["lead_hours"]] = row["ghi_raw"]
    return by_run


def _candidate_value(
    *,
    by_run: GridCellByRun,
    target: datetime,
    hours_before: int,
    field: str,
    cell: int,
) -> float | None:
    """Read one candidate run's value, at one grid cell, for a target hour.

    Args:
        by_run: `_dynamical_gfs_by_run`'s result.
        target: The hour being valued, its end.
        hours_before: How many hours before `target` the candidate run is floored from.
        field: `"wind_speed_100m_kmh"` (an instantaneous field, read directly) or `"ghi_raw"` (a
            since-reset accumulation, recovered through `_gfs_hourly_mean_since_reset`).
        cell: The grid cell id to read.

    Returns:
        The candidate's value, or `None` if the run, the cell, or the lead it needs is not in the
        extract.
    """
    run = _floor6(target - timedelta(hours=hours_before))
    cell_values = by_run.get(run, {}).get(cell)
    if cell_values is None:
        return None
    lead = int((target - run).total_seconds() // 3600)
    if field == "wind_speed_100m_kmh":
        return cell_values["wind_speed_100m_kmh"].get(lead)
    return _gfs_hourly_mean_since_reset(lead=lead, value_at_lead=cell_values["ghi_raw"])


def _per_site_best_cell_mad(
    *,
    by_run: GridCellByRun,
    served_rows: pl.DataFrame,
    hours_before: int,
    field: str,
    cells: list[int],
) -> tuple[float | None, dict[str, int]]:
    """Score one candidate offset by the mean, over sites, of each site's lowest-error cell.

    For one candidate offset, this scores every grid cell in the extract against one site's served
    series independently and keeps the cell with the lowest mean absolute error for that site. This
    is what lets V1 test the run-selection rule without knowing which cell a site truly falls in:
    every candidate offset gets exactly the same freedom to pick its own best-fitting cell, so a
    real difference in lead accuracy between offsets survives while the spatial-sampling noise from
    an unknown site-to-cell match does not.

    Args:
        by_run: `_dynamical_gfs_by_run`'s result.
        served_rows: `site`, `time`, `value` for one served column, nulls already dropped.
        hours_before: The candidate offset in hours before each target hour.
        field: Passed to `_candidate_value`.
        cells: Every grid cell id in the extract.

    Returns:
        The mean, over sites with at least one scoreable cell, of each site's best-cell mean
        absolute error (`None` if no site scored), and each scored site's chosen cell id (a raw
        grid-cell id, never printed as such by the caller — see `run_v1`).
    """
    per_site_mad = []
    chosen: dict[str, int] = {}
    for site in sorted(served_rows["site"].unique().to_list()):
        site_rows = served_rows.filter(pl.col("site") == site)
        targets = site_rows["time"].to_list()
        values = site_rows["value"].to_list()
        best_mad: float | None = None
        best_cell: int | None = None
        for cell in cells:
            differences = [
                abs(value - candidate)
                for target, value in zip(targets, values, strict=True)
                if (
                    candidate := _candidate_value(
                        by_run=by_run,
                        target=target,
                        hours_before=hours_before,
                        field=field,
                        cell=cell,
                    )
                )
                is not None
            ]
            if not differences:
                continue
            mad = float(np.mean(differences))
            if best_mad is None or mad < best_mad:
                best_mad, best_cell = mad, cell
        if best_mad is not None and best_cell is not None:
            per_site_mad.append(best_mad)
            chosen[site] = best_cell
    if not per_site_mad:
        return None, {}
    return float(np.mean(per_site_mad)), chosen


def run_v1(*, output_dir: Path) -> bool:
    """Check the Previous Runs run-selection rule against the raw Dynamical.org GFS archive.

    Every candidate offset (and, for radiation, every convention) is scored by
    `_per_site_best_cell_mad`: each site independently picks whichever of the extract's 9 grid
    cells fits its own series best, so a real difference in lead accuracy between candidates
    survives while the spatial-sampling noise from not knowing which cell a site truly falls in
    does not. Only an index rank among the extract's cells is ever printed, never a coordinate.

    Args:
        output_dir: Where `v1_wind.md` and `v1_radiation.md` are written.

    Returns:
        Whether the gate passed: the lowest mean absolute difference is at k = 0, for both N = 1
        and N = 2, on wind speed.
    """
    by_run = _dynamical_gfs_by_run()
    served = pl.read_parquet(_weather_dir() / "GFS-SEAMLESS" / "previous_runs" / "combined.parquet")
    cells = sorted({cell for cell_map in by_run.values() for cell in cell_map})
    cell_rank = {cell: rank for rank, cell in enumerate(cells)}

    wind_lines = [
        "| N | k (h) | mean of each site's best-cell |served − candidate|, km/h | sites |",
        "|---|---|---|---|",
    ]
    gate_pass = True
    for n in V1_DAYS_N:
        served_col = f"wind_speed_100m_previous_day{n}"
        rows = served.select("site", "time", value=pl.col(served_col)).drop_nulls()
        mad_by_k: dict[int, float] = {}
        ranks_by_k: dict[int, dict[str, int]] = {}
        for k in V1_OFFSETS_K:
            mad, chosen = _per_site_best_cell_mad(
                by_run=by_run,
                served_rows=rows,
                hours_before=24 * n + k,
                field="wind_speed_100m_kmh",
                cells=cells,
            )
            if mad is not None:
                mad_by_k[k] = mad
                ranks_by_k[k] = chosen
        for k, mad in sorted(mad_by_k.items()):
            wind_lines.append(f"| {n} | {k} | {mad:.4f} | {len(ranks_by_k[k])} |")
        if mad_by_k:
            best_k = min(mad_by_k, key=lambda k: mad_by_k[k])
            gate_pass = gate_pass and best_k == 0
            chosen_ranks = sorted(cell_rank[cell] for cell in ranks_by_k[best_k].values())
            wind_lines.append(
                f"\nN={n}: lowest mean absolute difference at k={best_k}. Chosen cell ranks "
                f"(of {len(cells)}, one per site): {chosen_ranks}.\n"
            )

    radiation_lines = [
        "| Convention | N | mean of each site's best-cell |served − candidate|, W/m² | sites |",
        "|---|---|---|---|",
    ]
    conventions = (("label", 0), ("start", -1))
    for n in V1_DAYS_N:
        served_col = f"shortwave_radiation_previous_day{n}"
        rows = (
            served.filter(pl.col("time").dt.hour() % 6 == 0)
            .select("site", "time", value=pl.col(served_col))
            .drop_nulls()
        )
        for convention, shift_hours in conventions:
            mad, chosen = _per_site_best_cell_mad(
                by_run=by_run,
                served_rows=rows,
                hours_before=24 * n - shift_hours,
                field="ghi_raw",
                cells=cells,
            )
            if mad is not None:
                radiation_lines.append(f"| {convention} | {n} | {mad:.2f} | {len(chosen)} |")

    (output_dir / "v1_wind.md").write_text("\n".join(wind_lines) + "\n")
    (output_dir / "v1_radiation.md").write_text("\n".join(radiation_lines) + "\n")
    _LOG.info("V1 wind gate: %s", "PASS" if gate_pass else "FAIL")
    return gate_pass


def _switch_ratio_by_hour(*, series: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Return each UTC hour's mean absolute second difference, scaled to its own daily mean.

    A run switch is a discontinuity a smooth diurnal signal does not otherwise have, so the hours
    where a fresh run cuts in carry a second difference well above the statistic's mean over all 24
    hours; scaling by that mean turns the raw statistic into a ratio comparable across products and
    fields regardless of each one's own units and variance.

    Args:
        series: Rows carrying `site`, `time` and `value_col`, one row per (site, time).
        value_col: The column to difference.

    Returns:
        One row per UTC hour of day, with `ratio`. Empty if the series' mean second difference is
        zero (a flat or all-null series).
    """
    diffs = (
        series.sort("site", "time")
        .with_columns(
            second_difference=(
                pl.col(value_col) - 2 * pl.col(value_col).shift(1) + pl.col(value_col).shift(2)
            ).over("site")
        )
        .drop_nulls("second_difference")
        .with_columns(hour=pl.col("time").dt.hour(), abs_diff=pl.col("second_difference").abs())
    )
    overall_mean = diffs["abs_diff"].mean()
    if not overall_mean:
        return pl.DataFrame(schema={"hour": pl.Int8, "ratio": pl.Float64})
    return diffs.group_by("hour").agg(ratio=pl.col("abs_diff").mean() / overall_mean).sort("hour")


V1B_COLUMN_TEMPLATES: Final[dict[str, str]] = {
    "wind": "wind_speed_100m_previous_day{day}",
    "temperature": "temperature_2m_previous_day{day}",
}
"""The two series V1b reads a run-switch pattern from, by field label, templated on the day."""

V1B_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The `previous_dayN` offsets V1b is checked on."""

V1B_CYCLE_CANDIDATES: Final[tuple[int, ...]] = (3, 6)
"""The run cycles, in hours, V1b chooses between. `n = 1` is the fallback when neither fits."""

V1B_SIGNATURE_THRESHOLD: Final[float] = 1.3
"""Below this switch-hour-to-other-hour ratio, V1b reports no clear signature rather than a
cycle."""


def _classify_cycle(*, by_hour: pl.DataFrame) -> tuple[int | None, float]:
    """Pick the run cycle in `V1B_CYCLE_CANDIDATES` whose switch hours stand out most.

    Args:
        by_hour: `_switch_ratio_by_hour`'s result.

    Returns:
        The best-fitting cycle and its score (the switch hours' mean ratio divided by the other
        hours' mean ratio), or `(None, best_score)` if that score is below
        `V1B_SIGNATURE_THRESHOLD` ("no clear signature").
    """
    best_cycle: int | None = None
    best_score = 0.0
    for n in V1B_CYCLE_CANDIDATES:
        switch = by_hour.filter(pl.col("hour") % n == 0)["ratio"]
        other = by_hour.filter(pl.col("hour") % n != 0)["ratio"]
        if switch.is_empty() or other.is_empty() or other.mean() in (0, None):
            continue
        score = float(switch.to_numpy().mean()) / float(other.to_numpy().mean())
        if score > best_score:
            best_score, best_cycle = score, n
    if best_score < V1B_SIGNATURE_THRESHOLD:
        return None, best_score
    return best_cycle, best_score


def run_v1b(*, output_dir: Path) -> None:
    """Locate each product's run cycle from the day-scaled second difference of its series.

    Args:
        output_dir: Where `v1b_run_switches.md` is written.
    """
    hour_headers = [f"h{hour:02d}" for hour in range(24)]
    lines = [
        "| Product | Field | Day | " + " | ".join(hour_headers) + " | Score | Cycle |",
        "|---|---|---|" + "---|" * len(hour_headers) + "---|---|",
    ]
    for name, dir_name in PRODUCT_DIRS.items():
        path = _weather_dir() / dir_name / "previous_runs" / "combined.parquet"
        if not path.exists():
            continue
        for field, template in V1B_COLUMN_TEMPLATES.items():
            for day in V1B_DAYS:
                column = template.format(day=day)
                combined = pl.read_parquet(path)
                if column not in combined.columns:
                    continue
                frame = combined.select("site", "time", value=pl.col(column)).drop_nulls()
                by_hour = _switch_ratio_by_hour(series=frame, value_col="value")
                if by_hour.is_empty():
                    continue
                ratio_by_hour = dict(
                    zip(by_hour["hour"].to_list(), by_hour["ratio"].to_list(), strict=True)
                )
                row = [f"{ratio_by_hour.get(hour, float('nan')):.2f}" for hour in range(24)]
                cycle, score = _classify_cycle(by_hour=by_hour)
                cycle_label = f"{cycle}-hourly" if cycle is not None else "no clear signature"
                lines.append(
                    f"| {name} | {field} | day{day} | "
                    + " | ".join(row)
                    + f" | {score:.2f} | {cycle_label} |"
                )
    (output_dir / "v1b_run_switches.md").write_text("\n".join(lines) + "\n")


def run_v3(*, output_dir: Path) -> None:
    """Read each product's timestamp convention from its lineage note, and state the height rule.

    Args:
        output_dir: Where `v3_conventions.md` is written.
    """
    lines = [
        "| Product | Timestamp convention (from lineage.json) |",
        "|---|---|",
    ]
    for name, dir_name in PRODUCT_DIRS.items():
        lineage_path = _weather_dir() / dir_name / "previous_runs" / "lineage.json"
        if not lineage_path.exists():
            lines.append(f"| {name} | not measured (no lineage.json) |")
            continue
        note = json.loads(lineage_path.read_text()).get("note", "")
        match = re.search(r"best fit: ([^(.]+)", note)
        convention = match.group(1).strip() if match else "unmeasured"
        lines.append(f"| {name} | {convention} |")
    lines.append(
        '\nHeight convention: ICON\'s served "100 m" wind (ICON-D2, ICON-EU, ICON global) is its '
        "native 120 m wind rescaled by about 0.98, not an independent 100 m level; every other "
        "product's 100 m wind is native. Radiation is a mean over the hour before its label; wind "
        "is instantaneous at its label, except where the table above measures a product as a "
        "snapshot instead."
    )
    (output_dir / "v3_conventions.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run V1 (a gate), V1b and V3, and write their tables under `--output-dir`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory the verification tables are written to.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate_pass = run_v1(output_dir=args.output_dir)
    run_v1b(output_dir=args.output_dir)
    run_v3(output_dir=args.output_dir)

    if not gate_pass:
        _LOG.error("V1's gate failed: the lowest mean absolute difference is not at k=0.")
        return 1
    _LOG.info("V1's gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
