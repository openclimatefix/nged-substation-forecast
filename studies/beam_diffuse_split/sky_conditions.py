"""Split the headline contrast by sky condition, to test the mechanism behind it.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**The reason arm C beats arm B is supposed to be the part of the direct fraction that global
irradiance and solar geometry cannot predict, and that part is not spread evenly across the sky.**
Under a clear sky almost all the irradiance is beam and the direct fraction follows from the sun's
position, so a separation model already knows it. Under thick overcast there is almost no beam to
know about. Between those, under broken cloud, two hours with the same global irradiance can carry
very different beam fractions depending on whether the sun's disc happens to be behind a cloud, and
that is the regime where a published beam field can say something no correlation can infer.

So the mechanism predicts a specific shape: a gain concentrated in the middle of the clearness
range and near zero at both ends. A gain spread evenly across the range would falsify it and point
instead at something duller, such as the product's beam and global fields being calibrated slightly
differently.

The bin edges are on the clearness index — global horizontal irradiance divided by the
extraterrestrial horizontal irradiance — which is the standard way of saying how much of what the
top of the atmosphere offered actually arrived. Both quantities are already columns every arm sees,
so binning on them introduces no information the arms lacked.

Run it with `uv run python
studies/beam_diffuse_split/sky_conditions.py --source cams`.
"""

import argparse
import logging
import sys
from typing import Final

import polars as pl
from run_experiment import (
    _add_time_features,
    _assign_folds,
    _bootstrap_difference,
    dataset_path_for,
    results_dir_for,
)
from sources import SOURCE_CHOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("sky_conditions")

CLEARNESS_EDGES: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 1.2)
"""Clearness-index bin edges, from thick overcast to clear sky.

Four bins rather than more, because the interval on each is what the claim rests on and a narrower
bin buys resolution with width. The top edge sits above one because a bright cloud edge can briefly
reflect more onto the horizontal than the clear-sky beam alone would deliver.
"""

CLEARNESS_LABELS: Final[tuple[str, ...]] = (
    "Overcast (below 0.2)",
    "Mostly cloudy (0.2 to 0.4)",
    "Broken cloud (0.4 to 0.6)",
    "Clear (above 0.6)",
)
"""What each bin is called in the write-up, in the same order as `CLEARNESS_EDGES`."""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("C_era5_split", "B_erbs"),
    ("C_era5_split", "B_learned"),
    ("B_learned", "B_erbs"),
)
"""The contrasts worth splitting, headline first."""

MINIMUM_ELEVATION_DEGREES: Final[float] = 5.0
"""Rows below this elevation are dropped from the breakdown.

The clearness index divides by the extraterrestrial horizontal irradiance, which goes to zero at
sunrise, so the ratio is numerically unstable at the horizon and would scatter those rows across
every bin rather than into the one they belong in.
"""


def _binned(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Label every row with its clearness-index bin.

    Args:
        dataset: The built frame, carrying global and extraterrestrial irradiance.

    Returns:
        The frame with a `clearness` ratio and a `sky` label, low-sun rows removed.
    """
    usable = dataset.filter(pl.col("solar_elevation_deg") >= MINIMUM_ELEVATION_DEGREES)
    with_ratio = usable.with_columns(
        clearness=pl.col("ghi_w_m2") / pl.col("extraterrestrial_horizontal_w_m2")
    )
    sky = pl.when(pl.col("clearness") < CLEARNESS_EDGES[1]).then(pl.lit(CLEARNESS_LABELS[0]))
    for index in range(1, len(CLEARNESS_LABELS) - 1):
        sky = sky.when(pl.col("clearness") < CLEARNESS_EDGES[index + 1]).then(
            pl.lit(CLEARNESS_LABELS[index])
        )
    return with_ratio.with_columns(sky=sky.otherwise(pl.lit(CLEARNESS_LABELS[-1])))


def _rows_for(*, losses: pl.DataFrame, binned: pl.DataFrame, label: str) -> pl.DataFrame:
    """Return the loss rows whose (site, time) falls in one sky bin."""
    keys = binned.filter(pl.col("sky") == label).select("site", "time")
    return losses.join(keys, on=["site", "time"], how="inner")


def main() -> int:
    """Print one table of contrasts per sky condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    parser.add_argument("--suffix", default="", help="Selects a variant build of the same source.")
    arguments = parser.parse_args()
    source = f"{arguments.source}{arguments.suffix}"

    results_dir = results_dir_for(source=source)
    losses = pl.read_parquet(results_dir / "per_row_losses.parquet").filter(
        (pl.col("setting") == "primary") & (pl.col("target") == "power_mw")
    )
    dataset = _assign_folds(
        dataset=_add_time_features(dataset=pl.read_parquet(dataset_path_for(source=source)))
    )
    binned = _binned(dataset=dataset)
    _LOG.info(
        "rows per sky bin: %s",
        binned.group_by("sky").len().sort("sky").to_dicts(),
    )

    lines: list[str] = [
        (
            "Rows are split by the clearness index, the share of the extraterrestrial horizontal"
            " irradiance that reached the ground. Negative favours the treatment arm."
        ),
        "",
        "| Sky condition | Contrast | ΔMAE (pp of P99 output) | 95% interval | Relative | Hours |",
        "|---|---|---|---|---|---|",
    ]
    records: list[dict[str, object]] = []
    for label in CLEARNESS_LABELS:
        scoped = _rows_for(losses=losses, binned=binned, label=label)
        if scoped.is_empty():
            continue
        hours = scoped.filter(pl.col("arm") == "B_erbs").select("site", "time").n_unique()
        for treatment, reference in CONTRASTS:
            if scoped.filter(pl.col("arm") == treatment).is_empty():
                continue
            interval = _bootstrap_difference(
                losses=scoped,
                treatment=treatment,
                reference=reference,
                metric="absolute_error_capped_fraction_of_capacity",
            )
            reference_rows = scoped.filter(pl.col("arm") == reference)
            reference_mae = float(
                reference_rows["absolute_error_capped_fraction_of_capacity"].to_numpy().mean()
            )
            relative = interval["difference"] / reference_mae * 100.0
            records.append(
                {
                    "sky": label,
                    "treatment": treatment,
                    "reference": reference,
                    "difference": interval["difference"] * 100.0,
                    "lower_95": interval["lower_95"] * 100.0,
                    "upper_95": interval["upper_95"] * 100.0,
                    "relative_percent": relative,
                    "hours": hours,
                }
            )
            lines.append(
                f"| {label} | {treatment} − {reference} | "
                f"{interval['difference'] * 100:+.4f} | "
                f"[{interval['lower_95'] * 100:+.4f}, {interval['upper_95'] * 100:+.4f}] | "
                f"{relative:+.2f}% | {hours:,} |"
            )
        _LOG.info("%s done", label)

    report = "\n".join(lines) + "\n"
    (results_dir / "sky_conditions.md").write_text(report)
    sys.stdout.write(report)
    pl.DataFrame(records).write_parquet(results_dir / "sky_intervals.parquet")

    breakdown = binned.group_by("sky").agg(
        hours=pl.len(),
        mean_clearness=pl.col("clearness").mean(),
        mean_direct_fraction=pl.col("direct_fraction").mean(),
        direct_fraction_spread=pl.col("direct_fraction").std(),
    )
    breakdown.write_parquet(results_dir / "sky_conditions.parquet")
    sys.stdout.write("\n" + str(breakdown.sort("sky")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
