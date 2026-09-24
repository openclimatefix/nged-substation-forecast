"""Check Open-Meteo's Previous Runs run-selection rule and each product's run-switch pattern.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>.

**V1 (a gate).** `_previous_dayN` is not documented as coming from a specific run; this checks the
working assumption that it comes from the freshest run initialised at least `24N` hours before the
hour, by comparing Open-Meteo's served GFS `_previous_dayN` values against the raw Dynamical.org GFS
archive on disk in `data/studies/weather/GFS_window_2025-07-01_2025-07-02/`, for a range of
candidate runs `24N + k` hours before the hour, `k` from 0 to 12. The gate is that the mean absolute
difference is lowest at `k = 0`, for `N = 1` and `N = 2`, which is what "the freshest run at least
`24N` hours old" predicts. Wind speed at 100 m compares directly; shortwave radiation additionally
checks, at hours divisible by 6 (the only hours where the two conventions pick a different run),
whether Open-Meteo selects the run by the hour's label or by the hour's start.

**V1b.** For every Previous Runs product, the mean absolute second difference of the
`_previous_day1` series (100 m wind), grouped by UTC hour of day, locates the hours where a fresh
run cuts in: those hours carry a materially larger second difference than their neighbours, because
a run switch is a discontinuity a smooth diurnal signal does not otherwise have. The spacing
between elevated hours is each product's run cycle.

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
import itertools
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


def _dynamical_gfs_by_run() -> dict[datetime, dict[str, dict[int, float]]]:
    """Load the raw Dynamical.org GFS window extract, averaged over its grid cells, by run.

    Returns:
        Each run's initialisation time to `{"wind_speed_100m_kmh": {lead: value}, "ghi_raw": {lead:
        value}}`, the grid-cell mean at each whole-hour lead the extract covers.
    """
    path = _weather_dir() / GFS_WINDOW_DIR_NAME / "GFS.parquet"
    frame = (
        pl.read_parquet(path)
        .with_columns(
            lead_hours=(pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32),
            wind_speed_100m_kmh=(
                (pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2) ** 0.5 * 3.6
            ),
        )
        .filter(pl.col("lead_hours") <= 120)
        .group_by("init_time", "lead_hours")
        .agg(
            pl.col("wind_speed_100m_kmh").mean(),
            pl.col("downward_short_wave_radiation_flux_surface").mean().alias("ghi_raw"),
        )
    )
    by_run: dict[datetime, dict[str, dict[int, float]]] = {}
    for row in frame.iter_rows(named=True):
        run = row["init_time"].replace(tzinfo=UTC)
        entry = by_run.setdefault(run, {"wind_speed_100m_kmh": {}, "ghi_raw": {}})
        entry["wind_speed_100m_kmh"][row["lead_hours"]] = row["wind_speed_100m_kmh"]
        entry["ghi_raw"][row["lead_hours"]] = row["ghi_raw"]
    return by_run


def _candidate_value(
    *,
    by_run: dict[datetime, dict[str, dict[int, float]]],
    target: datetime,
    hours_before: int,
    field: str,
) -> float | None:
    """Read one candidate run's value for a target hour, `hours_before` hours ahead of the run.

    Args:
        by_run: `_dynamical_gfs_by_run`'s result.
        target: The hour being valued, its end.
        hours_before: How many hours before `target` the candidate run is floored from.
        field: `"wind_speed_100m_kmh"` (an instantaneous field, read directly) or `"ghi_raw"` (a
            since-reset accumulation, recovered through `_gfs_hourly_mean_since_reset`).

    Returns:
        The candidate's value, or `None` if the run or the lead it needs is not in the extract.
    """
    run = _floor6(target - timedelta(hours=hours_before))
    run_values = by_run.get(run)
    if run_values is None:
        return None
    lead = int((target - run).total_seconds() // 3600)
    if field == "wind_speed_100m_kmh":
        return run_values["wind_speed_100m_kmh"].get(lead)
    return _gfs_hourly_mean_since_reset(lead=lead, value_at_lead=run_values["ghi_raw"])


def run_v1(*, output_dir: Path) -> bool:
    """Check the Previous Runs run-selection rule against the raw Dynamical.org GFS archive.

    Args:
        output_dir: Where `v1_wind.md` and `v1_radiation.md` are written.

    Returns:
        Whether the gate passed: the lowest mean absolute difference is at k = 0, for both N = 1
        and N = 2, on wind speed.
    """
    by_run = _dynamical_gfs_by_run()
    served = pl.read_parquet(_weather_dir() / "GFS-SEAMLESS" / "previous_runs" / "combined.parquet")

    wind_lines = ["| N | k (h) | mean |served − candidate|, km/h | n |", "|---|---|---|---|"]
    gate_pass = True
    for n in V1_DAYS_N:
        served_col = f"wind_speed_100m_previous_day{n}"
        rows = served.select("time", value=pl.col(served_col)).drop_nulls()
        # Every k is scored on the *same* rows, not on whichever rows happen to have a candidate at
        # that k: the two 8-run Dynamical.org extract windows only cover a candidate run for a
        # narrow band of target hours, and that band shifts with k. Scoring each k on its own
        # (possibly different) available rows would let a difference in *which weather episodes got
        # compared* masquerade as a difference in lead accuracy.
        candidates_by_k: dict[int, list[float | None]] = {
            k: [
                _candidate_value(
                    by_run=by_run,
                    target=target,
                    hours_before=24 * n + k,
                    field="wind_speed_100m_kmh",
                )
                for target in rows["time"].to_list()
            ]
            for k in V1_OFFSETS_K
        }
        complete = [
            index
            for index in range(rows.height)
            if all(candidates_by_k[k][index] is not None for k in V1_OFFSETS_K)
        ]
        values = rows["value"].to_list()
        mad_by_k: dict[int, float] = {}
        for k in V1_OFFSETS_K:
            differences = [abs(values[index] - candidates_by_k[k][index]) for index in complete]
            if differences:
                mad_by_k[k] = float(np.mean(differences))
        for k, mad in sorted(mad_by_k.items()):
            wind_lines.append(f"| {n} | {k} | {mad:.4f} | {len(complete)} |")
        if mad_by_k:
            best_k = min(mad_by_k, key=lambda k: mad_by_k[k])
            gate_pass = gate_pass and best_k == 0
            wind_lines.append(f"\nN={n}: lowest mean absolute difference at k={best_k}.\n")

    radiation_lines = [
        "| Convention | k=0 mean |served − candidate|, W/m² |",
        "|---|---|",
    ]
    conventions = (("label", 0), ("start", -1))
    for n in V1_DAYS_N:
        served_col = f"shortwave_radiation_previous_day{n}"
        rows = (
            served.filter(pl.col("time").dt.hour() % 6 == 0)
            .select("time", value=pl.col(served_col))
            .drop_nulls()
        )
        targets = rows["time"].to_list()
        values = rows["value"].to_list()
        # Score both conventions on the rows where *both* have a candidate, for the same reason
        # the wind loop above fixes its rows across k.
        candidates_by_convention = {
            convention: [
                _candidate_value(
                    by_run=by_run, target=target, hours_before=24 * n - shift_hours, field="ghi_raw"
                )
                for target in targets
            ]
            for convention, shift_hours in conventions
        }
        complete = [
            index
            for index in range(len(targets))
            if all(candidates_by_convention[c][index] is not None for c, _ in conventions)
        ]
        for convention, _ in conventions:
            differences = [
                abs(values[index] - candidates_by_convention[convention][index])
                for index in complete
            ]
            if differences:
                radiation_lines.append(
                    f"| N={n}, by {convention} | {float(np.mean(differences)):.2f} |"
                )

    (output_dir / "v1_wind.md").write_text("\n".join(wind_lines) + "\n")
    (output_dir / "v1_radiation.md").write_text("\n".join(radiation_lines) + "\n")
    _LOG.info("V1 wind gate: %s", "PASS" if gate_pass else "FAIL")
    return gate_pass


def _second_differences(*, series: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Return the mean absolute second difference of a series, grouped by UTC hour of day.

    Args:
        series: Rows carrying `time` and `value_col`, one row per (site, time).
        value_col: The column to difference.

    Returns:
        One row per UTC hour of day, with `mean_abs_second_difference`.
    """
    return (
        series.sort("site", "time")
        .with_columns(
            second_difference=(
                pl.col(value_col) - 2 * pl.col(value_col).shift(1) + pl.col(value_col).shift(2)
            ).over("site")
        )
        .drop_nulls("second_difference")
        .with_columns(hour=pl.col("time").dt.hour())
        .group_by("hour")
        .agg(mean_abs_second_difference=pl.col("second_difference").abs().mean())
        .sort("hour")
    )


V1B_COLUMNS: Final[dict[str, str]] = {
    "wind": "wind_speed_100m_previous_day1",
    "temperature": "temperature_2m_previous_day1",
}
"""The two day-1 series V1b reads a run-switch pattern from, by field label."""


def run_v1b(*, output_dir: Path) -> None:
    """Locate each product's run-switch hours from the second difference of its day-1 series.

    Args:
        output_dir: Where `v1b_run_switches.md` is written.
    """
    lines = [
        "| Product | Field | Hours with elevated second difference | Inferred cycle |",
        "|---|---|---|---|",
    ]
    for name, dir_name in PRODUCT_DIRS.items():
        path = _weather_dir() / dir_name / "previous_runs" / "combined.parquet"
        if not path.exists():
            continue
        for field, column in V1B_COLUMNS.items():
            frame = pl.read_parquet(path).select("site", "time", value=pl.col(column)).drop_nulls()
            by_hour = _second_differences(series=frame, value_col="value")
            if by_hour.is_empty():
                continue
            stats = by_hour["mean_abs_second_difference"].to_numpy()
            threshold = float(stats.mean()) + float(stats.std())
            elevated = sorted(
                by_hour.filter(pl.col("mean_abs_second_difference") > threshold)["hour"].to_list()
            )
            gaps = sorted({b - a for a, b in itertools.pairwise(elevated)})
            cycle = gaps[0] if gaps else None
            lines.append(f"| {name} | {field} | {elevated} | {cycle or 'unclear'}-hourly |")
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
