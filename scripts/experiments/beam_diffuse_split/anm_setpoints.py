"""Turn NGED's raw ANM setpoint history into an export-cap series, and check how it reads.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**NGED's active-network-management setpoint export is a step function of a generator's export cap,
not a log of megawatts curtailed.** Each row of the CSV is the moment the cap changed. The cap is
negative, because generation is negative in NGED's sign convention, and the largest magnitude it
ever reaches is the connection limit — so a generator sitting at that value is unconstrained, not
fully curtailed. Reading the magnitude as a curtailment volume inverts the signal. Eight of
the 2,268 readings carry a positive value where the rest are negative, so the magnitude is
what this script takes: a sign flip would turn those eight into caps below zero and read
them as curtailment.

Two consequences follow for the arithmetic. The cap has to be carried forward between events
rather than sampled at a window's edge, or curtailment inside the window is missed. And curtailment
in megawatts is not the limit minus the cap: it is what the weather would have allowed minus the
cap, and only where the cap is the smaller. This script therefore publishes the cap and leaves the
volume alone.

The checks it prints are the ones that say the cap is read the right way round: on bright hours
where the cap never left the connection limit, the generator's output per unit of irradiance should
match the uncurtailed sites, and where the cap moved it should fall short.

Run it from the repository root with `uv run --with pvlib python
scripts/experiments/beam_diffuse_split/anm_setpoints.py`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_dataset import REPO_DATA_DIR
from run_experiment import dataset_path_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("anm_setpoints")

ANM_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "anm"
"""Where NGED's setpoint exports are filed, one CSV per `time_series_id`."""

BRIGHT_W_M2: Final[float] = 400.0
"""Global irradiance above which a yield ratio is stable enough to compare."""

CSV_TIME_FORMAT: Final[str] = "%d/%m/%Y %H:%M:%S %:z"
"""NGED exports UK local time with the offset attached, so the offset does the work."""


def _events(*, path: Path) -> pl.DataFrame:
    """Read one setpoint export into `(time, cap_mw)`, newest value per instant.

    Args:
        path: The CSV NGED supplied.

    Returns:
        One row per cap change, sorted by time, with the cap made positive.
    """
    return (
        pl.read_csv(path)
        .with_columns(
            time=pl.col("Date Time").str.to_datetime(CSV_TIME_FORMAT).dt.convert_time_zone("UTC"),
            cap_mw=pl.col("Value").abs(),
        )
        .select("time", "cap_mw")
        .sort("time")
        .unique(subset="time", keep="last")
        .sort("time")
    )


def _half_hourly(*, events: pl.DataFrame) -> pl.DataFrame:
    """Average the step function over each half-hour.

    Expanding to one-minute resolution first is what makes the average a time-weighted one. A
    `join_asof` straight onto the half-hourly grid would sample the cap at each window's edge and
    miss every constraint that started and ended inside it.

    Args:
        events: One row per cap change.

    Returns:
        One row per half-hour, with the mean and the lowest cap in force during it.
    """
    limit = float(np.max(events["cap_mw"].to_numpy()))
    # `Series.min()` is declared as a union wide enough that ty will not take it as a datetime,
    # and `to_list()` on a Datetime series gives the values themselves.
    stamps = events["time"].dt.truncate("1m").to_list()
    grid = pl.datetime_range(
        stamps[0], stamps[-1], interval="1m", time_zone="UTC", eager=True
    ).to_frame("time")
    minutes = grid.join_asof(events, on="time", strategy="backward").with_columns(
        cap_mw=pl.col("cap_mw").fill_null(limit)
    )
    return (
        minutes.group_by_dynamic("time", every="30m", label="right", closed="right")
        .agg(cap_mw=pl.col("cap_mw").mean(), lowest_cap_mw=pl.col("cap_mw").min())
        .sort("time")
    )


def _dwell(*, spans: pl.DataFrame) -> pl.DataFrame:
    """Return how much of the elapsed time each distinct cap accounts for.

    Args:
        spans: One row per cap change, carrying the hours until the next change.

    Returns:
        `(cap_mw, hours, share)`, the share as a percentage, longest first.
    """
    return (
        spans.group_by(pl.col("cap_mw").round(2))
        .agg(hours=pl.col("hours").sum())
        .with_columns(share=pl.col("hours") / float(np.sum(spans["hours"].to_numpy())) * 100)
        .sort("hours", descending=True)
    )


def main() -> int:
    """Build the export-cap series for every setpoint export, and report how it reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cds", "open-meteo", "cams"), default="cams")
    parser.add_argument("--alignment", choices=("as-labelled", "shifted"), default="shifted")
    parser.add_argument(
        "--site", default="E", help="The anonymised label of the site the export belongs to."
    )
    arguments = parser.parse_args()

    exports = sorted(ANM_DIR.glob("ANM_Historic_Data_*.csv"))
    if not exports:
        _LOG.error("no setpoint exports under %s", ANM_DIR)
        return 1

    dataset = pl.read_parquet(
        dataset_path_for(source=arguments.source, alignment=arguments.alignment)
    ).with_columns(
        yield_ratio=(pl.col("power_mw") / pl.col("effective_capacity_mw"))
        / (pl.col("ghi_w_m2") / 1000.0)
    )
    others = dataset.filter(
        (pl.col("site") != arguments.site) & (pl.col("ghi_w_m2") >= BRIGHT_W_M2)
    )
    fleet = float(np.median(others["yield_ratio"].to_numpy()))

    for path in exports:
        events = _events(path=path)
        limit = float(np.max(events["cap_mw"].to_numpy()))
        half_hourly = _half_hourly(events=events)
        out = ANM_DIR / f"{path.stem.replace('ANM_Historic_Data', 'export_cap')}.parquet"
        half_hourly.write_parquet(out)
        _LOG.info("wrote %s", out)

        spans = events.with_columns(
            hours=(pl.col("time").shift(-1) - pl.col("time")).dt.total_seconds() / 3600.0
        ).drop_nulls("hours")
        dwell = _dwell(spans=spans)
        at_limit = events.filter(pl.col("cap_mw") >= limit - 0.001)
        went_live = at_limit["time"][0]
        live_dwell = _dwell(spans=spans.filter(pl.col("time") >= went_live))
        sys.stdout.write(
            f"\n{path.name}: {events.height} cap changes,"
            f" {events['time'].min()} to {events['time'].max()}\n"
            f"connection limit taken as the largest cap ever in force: {limit} MW\n"
            f"first cap at that limit, taken as the scheme going live: {went_live}\n"
            "\n| Cap (MW) | Share of the whole record | Share once live |\n|---|---|---|\n"
        )
        live_share = dict(live_dwell.select("cap_mw", "share").iter_rows())
        for row in dwell.head(5).iter_rows(named=True):
            share = live_share.get(row["cap_mw"])
            live = f"{share:.1f}%" if share is not None else "—"
            sys.stdout.write(f"| {row['cap_mw']:.2f} | {row['share']:.1f}% | {live} |\n")

        hourly = (
            half_hourly.group_by_dynamic("time", every="1h", label="right", closed="right")
            .agg(lowest_cap_mw=pl.col("lowest_cap_mw").min())
            .with_columns(site=pl.lit(arguments.site))
        )
        covered = (
            dataset.filter(pl.col("site") == arguments.site)
            .join(hourly, on=["site", "time"], how="inner")
            .filter(pl.col("ghi_w_m2") >= BRIGHT_W_M2)
            .with_columns(constrained=pl.col("lowest_cap_mw") < limit - 0.001)
        )
        before = covered.filter(pl.col("time") < went_live)
        exporting = before.filter(
            pl.col("constrained")
            & (pl.col("power_mw").abs() > 0.05 * pl.col("effective_capacity_mw"))
        )
        share_of_capacity = (exporting["power_mw"] / exporting["effective_capacity_mw"]).abs()
        median_export = 100 * float(np.median(share_of_capacity.to_numpy()))
        sys.stdout.write(
            f"\nBright hours before the scheme went live: {before.height},"
            f" of which the cap forbids export in {int(before['constrained'].sum())}.\n"
            f"Of those, {exporting.height} exported above 5% of capacity anyway, at a median of"
            f" {median_export:.0f}% — which is why the pre-live readings are discarded.\n"
        )
        site = covered.filter(pl.col("time") >= went_live)
        free = site.filter(~pl.col("constrained"))
        held = site.filter(pl.col("constrained"))
        low = site.filter(pl.col("yield_ratio") < 0.5)
        sys.stdout.write(
            f"\nBright hours since the scheme went live: {site.height}.\n"
            f"Median yield ratio where the cap never moved: "
            f"{float(np.median(free['yield_ratio'].to_numpy())):.4f},"
            f" against {fleet:.4f} for the other sites.\n"
            f"Median yield ratio where the cap moved:"
            f" {float(np.median(held['yield_ratio'].to_numpy())):.4f}"
            f" over {held.height} hours.\n"
            f"Bright hours below half the fleet's yield: {low.height},"
            f" of which the cap flags {int(low['constrained'].sum())}.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
