"""Read NGED's ANM curtailment feed and test it against the one curtailed site's output dips.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**One of the six PV sites spends 7.6% of its bright hours producing far less than the irradiance
implies, and NGED's own curtailment feed says why.** NGED's S3 bucket carries a `curtailment/`
prefix alongside the `timeseries/` prefix this project already ingests, on the same six-hour window
convention. Nothing in the repository reads it. This script is the first look.

The feed's payload is a list of half-hourly `{startTime, endTime, value}` records in megawatts,
under a header naming the generator, its substation number, and its curtailment type. The question
this script answers is what `value` means and whether the feed can be trusted: if it is the power
*lost* to a network instruction, then adding it back to the metered output should restore the site
to the yield its neighbours run at. It does.

The script prints:

- which of the six sites the feed belongs to, and how far back it reaches;
- the yield test above, before and after adding the curtailed megawatts back;
- how much of the site's unexplained shortfall the feed accounts for;
- what every arm's error looks like inside a curtailed hour, and what removing those hours does to
  the headline contrast.

Unlike its siblings this one needs the repository's own environment, because it reads NGED's
bucket through `contracts.settings`, which holds the credentials. Run it from the repository root
with `uv run --with pvlib python
scripts/experiments/beam_diffuse_split/anm_curtailment.py`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_dataset import (
    LABEL_PERMUTATION_SEED,
    METADATA_PATH,
    MIN_YEARS_OF_READINGS,
    POWER_DELTA_URI,
    SITE_LABELS,
)
from contracts.settings import Settings
from run_experiment import _bootstrap_difference, dataset_path_for, results_dir_for
from sources import SOURCE_CHOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("anm_curtailment")

CURTAILMENT_PREFIX: Final[str] = "curtailment/"
"""Where NGED publishes the ANM feed, beside the `timeseries/` prefix this project ingests."""

CACHE_PATH: Final[Path] = Path("/home/jack/scratch/bd/anm_curtailment.parquet")
"""Where the downloaded feed is cached, so a re-run does not re-fetch 579 objects."""

BRIGHT_W_M2: Final[float] = 400.0
"""Global irradiance above which a yield ratio is stable enough to read.

Below it the ratio divides a small number by a small number and scatters, which would swamp the
comparison this script rests on.
"""

LOW_YIELD: Final[float] = 0.5
"""Yield ratio below which an hour counts as an unexplained shortfall.

A normally-running site in this fleet sits between 1.23 and 1.30, so half of that is far outside
anything cloud or temperature produces.
"""

CONTRAST: Final[tuple[str, str]] = ("C_era5_split", "B_erbs")
"""The headline contrast, as `(treatment, reference)`."""


def _download() -> pl.DataFrame:
    """Pull every curtailment object into one half-hourly frame.

    Returns:
        One row per half-hourly curtailment record, with the header fields attached.
    """
    store = Settings().get_nged_s3_store()
    records: list[dict[str, object]] = []
    windows = 0
    for chunk in store.list(prefix=CURTAILMENT_PREFIX):
        for meta in chunk:
            if not meta["path"].endswith(".json"):
                continue
            windows += 1
            payload = json.loads(bytes(store.get(meta["path"]).bytes()))
            records.extend(
                {
                    "start_time": row["startTime"],
                    "end_time": row["endTime"],
                    "curtailed_mw": row["value"],
                    "substation_number": payload["SubstationNumber"],
                    "curtailment_type": payload["CurtailmentType"],
                    "units": payload["Units"],
                }
                for row in payload["data"]
            )
    _LOG.info("%d six-hour windows, %d half-hourly records", windows, len(records))
    return pl.DataFrame(records).with_columns(
        start_time=pl.col("start_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S%z")
        .dt.convert_time_zone("UTC"),
        end_time=pl.col("end_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S%z")
        .dt.convert_time_zone("UTC"),
    )


def _site_labels() -> pl.DataFrame:
    """Return the same site relabelling `build_dataset` applies, with substation numbers."""
    metadata = pl.read_parquet(METADATA_PATH).filter(pl.col("time_series_type") == "PV")
    counts = (
        pl.scan_delta(POWER_DELTA_URI)
        .group_by("time_series_id")
        .agg(pl.len().alias("n_rows"))
        .collect()
    )
    eligible = (
        metadata.select("time_series_id", "substation_number")
        .join(counts, on="time_series_id", how="inner")
        .filter(pl.col("n_rows") >= int(MIN_YEARS_OF_READINGS * 365.25 * 48))
        .sort("time_series_id")
    )
    shuffled = np.random.default_rng(LABEL_PERMUTATION_SEED).permutation(list(SITE_LABELS))
    return eligible.with_columns(site=pl.Series(shuffled, dtype=pl.Utf8)).select(
        "site", "substation_number"
    )


def _hourly(*, curtailment: pl.DataFrame, site: str) -> pl.DataFrame:
    """Put the half-hourly feed on the same period-ending hourly grid as the experiment.

    Args:
        curtailment: The downloaded feed.
        site: The anonymised label of the site the feed belongs to.

    Returns:
        One row per curtailed hour, carrying the mean curtailed megawatts across its half-hours.
    """
    return (
        curtailment.with_columns(
            time=pl.col("end_time").dt.offset_by("-1s").dt.truncate("1h").dt.offset_by("1h")
        )
        .group_by("time")
        .agg(curtailed_mw=pl.col("curtailed_mw").mean(), half_hours=pl.len())
        .with_columns(site=pl.lit(site))
    )


def main() -> int:
    """Print what the curtailment feed holds and what it explains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download rather than reading the cache."
    )
    arguments = parser.parse_args()

    if arguments.refresh or not CACHE_PATH.exists():
        curtailment = _download()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        curtailment.write_parquet(CACHE_PATH)
    else:
        curtailment = pl.read_parquet(CACHE_PATH)

    substation = int(curtailment["substation_number"].to_numpy()[0])
    if curtailment["substation_number"].n_unique() != 1:
        msg = "the feed now carries more than one generator; this script assumes exactly one"
        raise ValueError(msg)
    labels = _site_labels()
    matched = labels.filter(pl.col("substation_number") == substation)
    if matched.is_empty():
        _LOG.warning("the curtailed generator is not one of the six PV sites; nothing to test")
        return 0
    site = str(matched["site"][0])

    lines: list[str] = [
        f"The feed carries one generator, substation {substation}, which is site {site}.",
        (
            f"Curtailment type: {curtailment['curtailment_type'][0]}."
            f" Units: {curtailment['units'][0]}."
        ),
        (
            f"{curtailment.height} half-hourly records, from"
            f" {curtailment['start_time'].min()} to {curtailment['end_time'].max()}."
        ),
        "",
    ]

    hourly = _hourly(curtailment=curtailment, site=site)
    dataset = pl.read_parquet(dataset_path_for(source=arguments.source))
    scored = dataset.with_columns(
        yield_ratio=(pl.col("power_mw") / pl.col("effective_capacity_mw"))
        / (pl.col("ghi_w_m2") / 1000.0)
    ).filter(pl.col("ghi_w_m2") >= BRIGHT_W_M2)

    others = scored.filter(pl.col("site") != site)
    fleet = float(np.median(others["yield_ratio"].to_numpy())) if others.height else float("nan")
    lines.append(f"Median yield ratio across the other five sites: {fleet:.3f}.")

    covered = scored.filter(
        (pl.col("site") == site)
        & (pl.col("time") >= curtailment["start_time"].min())
        & (pl.col("time") <= curtailment["end_time"].max())
    ).join(hourly, on=["site", "time"], how="left")
    logged = covered.filter(pl.col("curtailed_mw").is_not_null())
    restored = logged.with_columns(
        restored_yield=(
            (pl.col("power_mw") + pl.col("curtailed_mw")) / pl.col("effective_capacity_mw")
        )
        / (pl.col("ghi_w_m2") / 1000.0)
    )
    measured_median = float(np.median(logged["yield_ratio"].to_numpy()))
    restored_median = float(np.median(restored["restored_yield"].to_numpy()))
    lines.extend(
        [
            (
                f"Site {site} bright hours inside the covered window: {covered.height},"
                f" of which {logged.height} carry a curtailment record."
            ),
            (
                f"Median yield on those hours: {measured_median:.3f}"
                f" measured, {restored_median:.3f} after adding the"
                " curtailed megawatts back."
            ),
            "",
        ]
    )

    shortfall = covered.filter(pl.col("yield_ratio") < LOW_YIELD)
    explained = shortfall.filter(pl.col("curtailed_mw").is_not_null())
    all_bright = scored.filter(pl.col("site") == site)
    lines.extend(
        [
            (
                f"Unexplained shortfall hours (yield below {LOW_YIELD}) inside the window:"
                f" {shortfall.height}, of which {explained.height} have a curtailment record."
            ),
            (
                f"Across the whole record site {site} has {all_bright.height} bright hours, of"
                f" which {all_bright.filter(pl.col('yield_ratio') < LOW_YIELD).height} fall below"
                f" {LOW_YIELD}. The feed reaches only the window above."
            ),
            "",
        ]
    )

    losses = pl.read_parquet(
        results_dir_for(source=arguments.source) / "per_row_losses.parquet"
    ).filter((pl.col("setting") == "primary") & (pl.col("target") == "power_mw"))
    inside = losses.join(hourly.select("site", "time"), on=["site", "time"], how="semi")
    lines.extend(
        [
            f"{inside.height} scored rows fall in a curtailed hour. Each arm's error there:",
            "",
            (
                "| Arm | MAE against the cap (% of P99 output) |"
                " MAE ignoring the cap (% of P99 output) |"
            ),
            "|---|---|---|",
        ]
    )
    # Both readings belong here. The capped column is what the experiment scores; the uncapped
    # column is the size of the penalty the clamp removes.
    lines.extend(
        f"| {row['arm']} | {row['mae_capped']:.2f} | {row['mae_uncapped']:.2f} |"
        for row in inside.group_by("arm")
        .agg(
            mae_capped=pl.col("absolute_error_capped_fraction_of_capacity").mean() * 100,
            mae_uncapped=pl.col("absolute_error_fraction_of_capacity").mean() * 100,
        )
        .sort("arm")
        .iter_rows(named=True)
    )

    without = losses.join(hourly.select("site", "time"), on=["site", "time"], how="anti")
    lines.extend(["", "| Rows | ΔMAE (pp of P99 output) | 95% interval |", "|---|---|---|"])
    for label, frame in (("As published", losses), ("Curtailed hours removed", without)):
        interval = _bootstrap_difference(
            losses=frame,
            treatment=CONTRAST[0],
            reference=CONTRAST[1],
            metric="absolute_error_capped_fraction_of_capacity",
        )
        lines.append(
            f"| {label} | {interval['difference'] * 100:+.4f} |"
            f" [{interval['lower_95'] * 100:+.4f}, {interval['upper_95'] * 100:+.4f}] |"
        )

    report = "\n".join(lines) + "\n"
    (results_dir_for(source=arguments.source) / "anm_curtailment.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
