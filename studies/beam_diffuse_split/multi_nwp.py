"""Measure what a second numerical weather prediction model adds to a solar power forecast.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/800>.

**Giving one model the irradiance from two weather products is a well-worn way to improve a power
forecast, and none of the per-source runs can measure it**, because each fits on one product's
columns alone. This script joins the ICON-D2 and UKV datasets on the hours and sites they share,
fits XGBoost on the shared rows, and reports what the pair achieves against either product on its
own.

**Two negative controls, because the obvious one cannot fail.** An arm holding two weather
products has more columns than an arm holding one, and a gain could in principle be column count
rather than information. Arm `icon_duplicated` shows the ICON-D2 irradiance twice under two names,
but `colsample_bytree` is 1.0, so the duplicate is scored with identical gain at every split and
every tree comes out identical: that arm is guaranteed to tie, and it therefore demonstrates only
that XGBoost is deterministic on duplicate columns.

Arm `icon_plus_shuffled_ukv` is the control that can fail. It shows the tree a second column with
UKV's exact diurnal and seasonal distribution and no information about the weather of the hour it
sits on, built by permuting UKV's irradiance among the rows sharing a site, a calendar month, and
an hour of day. Folds are whole-month blocks, so a permuted value never crosses a fold boundary.
If a second column of pure climatology bought skill, that arm would show it, and the gain from the
real UKV column would have to beat it.

Every arm is shown the same temperature — ICON-D2's — so that the contrasts measure irradiance and
nothing else.

Run it with `uv run python
studies/beam_diffuse_split/multi_nwp.py`.
"""

import argparse
import logging
import sys
from typing import Final

import polars as pl
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from run_experiment import SHARED_FEATURES, _add_time_features, dataset_path_for
from sources import STUDY_DATA_DIR
from studies.blending import climatology_permutation
from studies.bootstrap import bootstrap_difference
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, assign_folds, out_of_fold_losses

_LOG = logging.getLogger(__name__)

PERCENTAGE_POINTS: Final[float] = 100.0

PRIMARY_SOURCE: Final[str] = "icon-d2"
"""The product whose shared columns — power, geometry, temperature — the joined frame keeps."""

SECOND_SOURCE: Final[str] = "ukv"
"""The product contributing only its irradiance columns, under the `_ukv` suffix."""

SHUFFLE_SEED: Final[int] = 20260922
"""Seed for the permutation behind `icon_plus_shuffled_ukv`."""

JOINED_IRRADIANCE: Final[tuple[str, ...]] = ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2")
"""The columns taken from both products rather than from `PRIMARY_SOURCE` alone."""

ARM_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "icon_global": ("ghi_w_m2",),
    "ukv_global": ("ghi_w_m2_ukv",),
    "icon_duplicated": ("ghi_w_m2", "ghi_w_m2_duplicate"),
    "icon_plus_shuffled_ukv": ("ghi_w_m2", "ghi_w_m2_ukv_shuffled"),
    "both_global": ("ghi_w_m2", "ghi_w_m2_ukv"),
    "icon_split": ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2"),
    "both_split": (
        "ghi_w_m2",
        "bhi_w_m2",
        "dhi_w_m2",
        "ghi_w_m2_ukv",
        "bhi_w_m2_ukv",
        "dhi_w_m2_ukv",
    ),
}
"""The irradiance columns each arm is shown, on top of `SHARED_FEATURES`."""

HEADLINE_CONTRAST: Final[tuple[str, str]] = ("both_global", "icon_global")
"""Whether a second weather product improves on the better of the two on its own."""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    HEADLINE_CONTRAST,
    ("icon_duplicated", "icon_global"),
    ("icon_plus_shuffled_ukv", "icon_global"),
    ("both_global", "icon_plus_shuffled_ukv"),
    ("both_global", "ukv_global"),
    ("both_global", "icon_duplicated"),
    ("icon_global", "ukv_global"),
    ("both_split", "icon_split"),
    ("both_split", "both_global"),
    ("icon_split", "icon_global"),
)
"""Every (treatment, reference) pairing an interval is computed for, headline first."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports, with the export-cap clamp applied identically to every arm.

Each row's error is divided by its own generator's capacity before any mean or difference is taken,
so the arm table and the contrast table subtract into each other exactly.
"""


def _joined() -> pl.DataFrame:
    """Join the two products on the site-hours they both cover.

    Returns:
        One row per shared site-hour, carrying `PRIMARY_SOURCE`'s columns unchanged and
        `SECOND_SOURCE`'s irradiance under a `_ukv` suffix.
    """
    primary = pl.read_parquet(dataset_path_for(source=PRIMARY_SOURCE))
    second = pl.read_parquet(dataset_path_for(source=SECOND_SOURCE)).select(
        ["site", "time", *[pl.col(column).alias(f"{column}_ukv") for column in JOINED_IRRADIANCE]]
    )
    joined = primary.join(second, on=["site", "time"], how="inner")
    if joined.is_empty():
        msg = f"{PRIMARY_SOURCE} and {SECOND_SOURCE} share no site-hours"
        raise RuntimeError(msg)
    return joined.with_columns(ghi_w_m2_duplicate=pl.col("ghi_w_m2"))


def _with_shuffled_second_product(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add a column carrying UKV's distribution with the weather of the hour removed.

    The permutation runs among the rows that share a site, a calendar month, and an hour of day,
    so the column keeps UKV's diurnal shape and its month-to-month climatology exactly and loses
    which day each value described. Folds are whole-month blocks, so no value crosses a fold
    boundary.

    Args:
        dataset: The joined frame, already carrying `month` and `hour_of_day`.

    Returns:
        `dataset` with `ghi_w_m2_ukv_shuffled`.
    """
    return climatology_permutation(
        frame=dataset,
        column_groups=[("ghi_w_m2_ukv",)],
        by=("site", "month", "hour_of_day"),
        seed=SHUFFLE_SEED,
    )


def _as_percentage_of_capacity(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean absolute error as a percentage of effective capacity.

    Args:
        losses: Per-row losses holding every arm.
        arm: The arm to score.

    Returns:
        The mean of each row's error over its generator's capacity, in percentage points.
    """
    rows = losses.filter(pl.col("arm") == arm)
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def main() -> int:
    """Fit every arm on the shared rows and report the table and the intervals."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    # This script takes no arguments. Parsing anyway keeps `--help` working and rejects unknown
    # flags.
    parser.parse_args()

    dataset = with_export_cap(
        dataset=assign_folds(
            dataset=_with_shuffled_second_product(
                dataset=_add_time_features(dataset=drop_commissioning_ramp(dataset=_joined()))
            )
        )
    )
    _LOG.info(
        "shared rows: %d, sites: %d, months: %d",
        dataset.height,
        dataset["site"].n_unique(),
        dataset["month"].n_unique(),
    )

    sites = sorted(dataset["site"].unique().to_list())
    frames: list[pl.DataFrame] = []
    for arm in ARM_FEATURES:
        frames.extend(
            out_of_fold_losses(
                site_rows=dataset.filter(pl.col("site") == site),
                features=[*SHARED_FEATURES, *ARM_FEATURES[arm]],
                target="power_mw",
                hyper_parameters=PRIMARY_HYPER_PARAMETERS,
                with_quantiles=False,
            ).with_columns(arm=pl.lit(arm))
            for site in sites
        )
        _LOG.info("fitted %s", arm)
    losses = pl.concat(frames)

    output_dir = STUDY_DATA_DIR / "beam_diffuse_multi_nwp"
    output_dir.mkdir(parents=True, exist_ok=True)
    losses.write_parquet(output_dir / "losses.parquet")

    lines = [
        f"### ICON-D2 and UKV together, on the {dataset.height:,} site-hours they share",
        "",
        "| Arm | MAE (% of capacity) |",
        "|---|---|",
    ]
    lines.extend(
        f"| {arm} | {_as_percentage_of_capacity(losses=losses, arm=arm):.3f} |"
        for arm in ARM_FEATURES
    )
    lines += [
        "",
        "| Contrast | ΔMAE (pp of capacity) | 95% interval | excludes zero? |",
        "|---|---|---|---|",
    ]
    for treatment, reference in CONTRASTS:
        interval = bootstrap_difference(
            losses=losses, treatment=treatment, reference=reference, metric=METRIC
        )
        excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
        difference, lower, upper = (
            interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
        )
        lines.append(
            f"| {treatment} − {reference} | {difference:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] | "
            f"{'**yes**' if excludes else 'no'} |"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
