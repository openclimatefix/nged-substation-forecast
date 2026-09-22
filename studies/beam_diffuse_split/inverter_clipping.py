"""Split the headline contrast by whether the meter was sitting on its inverter ceiling.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**All six sites run a direct-current array larger than their alternating-current ceiling, so the
brightest hours are clipped, and a clipped hour cannot respond to irradiance at all.** That matters
for this experiment because the beam/diffuse split is a statement about irradiance: in an hour whose
output is pinned to the inverter's limit, knowing the split more precisely cannot move the
prediction. If a meaningful share of the record is clipped, the pooled contrast understates what the
split is worth over the hours where it can bite.

The alternative reading is the one worth ruling out: that clipping *manufactures* the contrast
rather than diluting it, because the arms would then differ in how well they predict a flat top
rather than in what they know about irradiance. Splitting the rows settles which it is.

Rows are split on the experiment's own normaliser, `effective_capacity_mw`, which is each site's
P99 over its whole history. The cut is swept over two thresholds, because "on the ceiling" is a
proxy rather than a measurement: a hot afternoon derates below the nameplate limit and a cold bright
one runs above it, so no single threshold separates the two states cleanly.

Run it with `uv run python
studies/beam_diffuse_split/inverter_clipping.py --source cams`.
"""

import argparse
import logging
import sys
from typing import Final

import polars as pl
from run_experiment import dataset_path_for, results_dir_for
from run_physics_experiment import results_dir_for as physics_results_dir_for
from sources import SOURCE_CHOICES
from studies.bootstrap import bootstrap_difference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("inverter_clipping")

CEILING_THRESHOLDS: Final[tuple[float, ...]] = (0.90, 0.95)
"""Fractions of `effective_capacity_mw` above which an hour counts as sitting on the ceiling.

Two thresholds rather than one, so the finding can be read as robust to the cut or not. A single
threshold would hide the fact that the boundary between a clipped hour and a merely bright one is
itself fuzzy.
"""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("C_era5_split", "B_erbs"),
    ("C_era5_split", "B_learned"),
)
"""The contrasts worth splitting, headline first."""

PHYSICS_ARM: Final[str] = "P_C_source_split"
"""The physical-model arm whose fitted parameters give the direct-current to alternating-current
ratio reported here."""


def _with_ceiling_flag(*, dataset: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Label every row with whether the meter sat on its inverter ceiling.

    Args:
        dataset: The built frame, carrying `power_mw` and `effective_capacity_mw`.
        threshold: Fraction of `effective_capacity_mw` above which an hour counts as on the ceiling.

    Returns:
        A frame of `(site, time, share_of_capacity, on_ceiling)`.
    """
    return dataset.with_columns(
        share_of_capacity=pl.col("power_mw") / pl.col("effective_capacity_mw")
    ).select(
        "site",
        "time",
        "share_of_capacity",
        on_ceiling=pl.col("share_of_capacity") >= threshold,
    )


def _top_flatness(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Return how flat each site's output distribution is at the top.

    A site whose inverter clips hard has almost no headroom above its own P99, because the ceiling
    truncates the distribution there. A site that does not clip keeps a longer tail.

    Args:
        dataset: The built frame, already restricted to daylight rows by the caller.

    Returns:
        One row per site, giving the P99.9 and the highest reading as multiples of the P99.
    """
    return (
        dataset.group_by("site")
        .agg(
            p99=pl.col("power_mw").quantile(0.99),
            p999=pl.col("power_mw").quantile(0.999),
            highest=pl.col("power_mw").max(),
        )
        .with_columns(
            p999_over_p99=pl.col("p999") / pl.col("p99"),
            highest_over_p99=pl.col("highest") / pl.col("p99"),
        )
        .select("site", "p999_over_p99", "highest_over_p99")
        .sort("site")
    )


def _fitted_ratios(*, source: str) -> pl.DataFrame:
    """Return the physical model's fitted direct-current to alternating-current ratio per site."""
    path = physics_results_dir_for(source=source) / "fitted_parameters.parquet"
    return (
        pl.read_parquet(path)
        .filter(pl.col("arm") == PHYSICS_ARM)
        .select(
            "site",
            dc_ac_ratio=pl.col("capacity_fraction_of_p99") / pl.col("clip_fraction_of_p99"),
        )
        .sort("site")
    )


def _flatness_table(*, dataset: pl.DataFrame, source: str) -> list[str]:
    """Return the markdown rows describing how clipped each site is."""
    flatness = _top_flatness(dataset=dataset).join(
        _fitted_ratios(source=source), on="site", how="left"
    )
    lines = [
        (
            "How clipped each site is. A hard inverter ceiling truncates the output distribution,"
            " leaving little headroom above the site's own P99."
        ),
        "",
        "| Site | P99.9 / P99 | Highest / P99 | Fitted DC/AC ratio |",
        "|---|---|---|---|",
    ]
    for row in flatness.iter_rows(named=True):
        ratio = row["dc_ac_ratio"]
        printed = "—" if ratio is None else f"{ratio:.2f}"
        lines.append(
            f"| {row['site']} | {row['p999_over_p99']:.3f} | {row['highest_over_p99']:.3f} |"
            f" {printed} |"
        )
    return lines


def main() -> int:
    """Print the headline contrast on and off the inverter ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    parser.add_argument("--suffix", default="", help="Selects a variant build of the same source.")
    arguments = parser.parse_args()
    source = f"{arguments.source}{arguments.suffix}"

    results_dir = results_dir_for(source=source)
    losses = pl.read_parquet(results_dir / "per_row_losses.parquet").filter(
        (pl.col("setting") == "primary") & (pl.col("target") == "power_mw")
    )
    dataset = pl.read_parquet(dataset_path_for(source=source)).filter(
        pl.col("solar_elevation_deg") > 0
    )

    lines = _flatness_table(dataset=dataset, source=source)
    records: list[dict[str, object]] = []

    for threshold in CEILING_THRESHOLDS:
        flagged = _with_ceiling_flag(dataset=dataset, threshold=threshold)
        share = flagged.group_by("on_ceiling").agg(
            rows=pl.len(), output=pl.col("share_of_capacity").sum()
        )
        on_ceiling_share = share.filter(pl.col("on_ceiling"))
        percent_rows = (
            float(on_ceiling_share["rows"].to_numpy().sum())
            / float(share["rows"].to_numpy().sum())
            * 100.0
        )
        percent_output = (
            float(on_ceiling_share["output"].to_numpy().sum())
            / float(share["output"].to_numpy().sum())
            * 100.0
        )

        lines.extend(
            [
                "",
                (
                    f"Ceiling at {threshold:.2f} of each site's P99: {percent_rows:.1f}% of"
                    f" daylight hours, carrying {percent_output:.1f}% of the output."
                    " Negative favours the treatment arm."
                ),
                "",
                "| Rows | Contrast | ΔMAE (pp of P99 output) | 95% interval | Hours |",
                "|---|---|---|---|---|",
            ]
        )
        for on_ceiling in (False, True):
            keys = flagged.filter(pl.col("on_ceiling") == on_ceiling).select("site", "time")
            scoped = losses.join(keys, on=["site", "time"], how="inner")
            hours = scoped.filter(pl.col("arm") == "B_erbs").select("site", "time").n_unique()
            label = "On the ceiling" if on_ceiling else "Off the ceiling"
            for treatment, reference in CONTRASTS:
                if scoped.filter(pl.col("arm") == treatment).is_empty():
                    continue
                interval = bootstrap_difference(
                    losses=scoped,
                    treatment=treatment,
                    reference=reference,
                    metric="absolute_error_capped_fraction_of_capacity",
                )
                records.append(
                    {
                        "threshold": threshold,
                        "on_ceiling": on_ceiling,
                        "treatment": treatment,
                        "reference": reference,
                        "difference": interval["difference"] * 100.0,
                        "lower_95": interval["lower_95"] * 100.0,
                        "upper_95": interval["upper_95"] * 100.0,
                        "hours": hours,
                    }
                )
                lines.append(
                    f"| {label} | {treatment} − {reference} |"
                    f" {interval['difference'] * 100:+.4f} |"
                    f" [{interval['lower_95'] * 100:+.4f}, {interval['upper_95'] * 100:+.4f}] |"
                    f" {hours:,} |"
                )
            _LOG.info("threshold %.2f, on_ceiling=%s done", threshold, on_ceiling)

    report = "\n".join(lines) + "\n"
    (results_dir / "inverter_clipping.md").write_text(report)
    sys.stdout.write(report)
    pl.DataFrame(records).write_parquet(results_dir / "inverter_clipping.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
