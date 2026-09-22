"""Score four irradiance products on one common row set, before and after the UKV upgrade.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>.

**Two defects in the earlier per-source comparison are what this script exists to remove.** Each
source was scored on its own rows, so the four pairwise figures did not compose into a ranking. And
every source's folds were cut from that source's own span, which put every post-upgrade UKV row in
the last fold — whose model had therefore trained on pre-upgrade UKV alone. A product that changed
in January 2026 would look worse under that arrangement whether or not it got worse, because the
model was shown an input it had never trained on.

**Here every product is shown to the same booster on the same rows, and the folds are cut inside
each era.** A model scoring a post-upgrade row has trained on post-upgrade rows, so a surviving
difference is a difference between the products rather than a train-against-test mismatch.

**The eras are not seasonally comparable, which is why there are three of them.** The post-upgrade
span runs February to September and holds no autumn or winter, so comparing it with the whole
pre-upgrade record compares different times of year. `pre_matched` restricts the pre-upgrade record
to the same calendar months, so a pre-against-post reading has the season held roughly fixed. The
cross-product gap inside one era is unaffected by any of this, because every product sees the same
rows.

Run it with `uv run --no-project --with polars --with numpy --with xgboost --with pvlib python
scripts/experiments/beam_diffuse_split/era_comparison.py`.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
from commissioning import drop_commissioning_ramp
from export_cap import clamp_to_cap, with_export_cap
from run_experiment import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SHARED_FEATURES,
    _add_time_features,
    _assign_folds,
    _bootstrap_difference,
    _fit_one_fold,
    dataset_path_for,
)
from sources import REPO_DATA_DIR

_LOG = logging.getLogger(__name__)

PERCENTAGE_POINTS: Final[float] = 100.0

UPGRADE_MONTH: Final[str] = "2026-02"
"""The first whole month after the Met Office made PS47 operational on 21 January 2026."""

PRODUCTS: Final[dict[str, str]] = {
    "cams": "cams",
    "era5": "open-meteo",
    "ukv": "ukv",
    "icon_d2": "icon-d2",
}
"""Arm name to the source name `build_dataset.py` wrote, for every product compared."""

BASE_PRODUCT: Final[str] = "era5"
"""The product whose shared columns - power, capacity, geometry, temperature - the frame keeps.

Every arm is shown one product's irradiance and this product's temperature, so that a contrast
measures irradiance alone. ERA5 is chosen because it covers every hour the others do.
"""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ukv", "era5"),
    ("icon_d2", "era5"),
    ("icon_d2", "ukv"),
    ("cams", "era5"),
    ("cams", "icon_d2"),
)
"""Every (treatment, reference) pairing an interval is computed for, within each era."""

METRIC: Final[str] = "absolute_error_capped_mw"
"""The loss every table reports, with the export-cap clamp applied identically to every arm."""


def _joined(*, alignment: str) -> pl.DataFrame:
    """Inner-join every product on the site-hours all of them cover.

    Args:
        alignment: The power-stamp alignment every dataset was built at.

    Returns:
        One row per common site-hour, carrying `ghi_<arm>` for every arm.

    Raises:
        RuntimeError: If the four products share no site-hours.
    """
    frame = pl.read_parquet(dataset_path_for(source=PRODUCTS[BASE_PRODUCT], alignment=alignment))
    frame = frame.rename({"ghi_w_m2": f"ghi_{BASE_PRODUCT}"})
    for arm, source in PRODUCTS.items():
        if arm == BASE_PRODUCT:
            continue
        other = pl.read_parquet(dataset_path_for(source=source, alignment=alignment)).select(
            ["site", "time", pl.col("ghi_w_m2").alias(f"ghi_{arm}")]
        )
        frame = frame.join(other, on=["site", "time"], how="inner")
    if frame.is_empty():
        msg = f"the {len(PRODUCTS)} products share no site-hours at alignment {alignment}"
        raise RuntimeError(msg)
    return frame


def _era(*, frame: pl.DataFrame, era: str) -> pl.DataFrame:
    """Restrict the joined frame to one era.

    Args:
        frame: The joined frame, already carrying the `month` label.
        era: One of `pre_all`, `pre_matched`, or `post`.

    Returns:
        The rows belonging to that era.

    Raises:
        ValueError: If `era` is not one of the three names.
    """
    post = pl.col("month") >= UPGRADE_MONTH
    if era == "post":
        return frame.filter(post)
    if era == "pre_all":
        return frame.filter(~post)
    if era == "pre_matched":
        months_in_post = sorted({label[-2:] for label in frame.filter(post)["month"].unique()})
        return frame.filter(~post & pl.col("month").str.slice(-2).is_in(months_in_post))
    msg = f"unknown era {era}"
    raise ValueError(msg)


def _losses(*, rows: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Produce out-of-fold losses for one arm across every site.

    Args:
        rows: Every row of one era, carrying `fold`, `month`, and each arm's irradiance.
        arm: The key into `PRODUCTS`.

    Returns:
        One row per (site, time, seed) with the clamped absolute error.
    """
    features = [*SHARED_FEATURES, "ghi_w_m2"]
    outputs: list[pl.DataFrame] = []
    for site in sorted(rows["site"].unique().to_list()):
        at_site = rows.filter(pl.col("site") == site).with_columns(ghi_w_m2=pl.col(f"ghi_{arm}"))
        for fold in range(N_FOLDS):
            test = at_site.filter(pl.col("fold") == fold)
            train = at_site.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
            if test.is_empty() or train.is_empty():
                continue
            actual = test["power_mw"].to_numpy()
            for seed in SEEDS:
                point, _ = _fit_one_fold(
                    train=train,
                    test=test,
                    features=features,
                    target="power_mw",
                    hyper_parameters=PRIMARY_HYPER_PARAMETERS,
                    seed=seed,
                    with_quantiles=False,
                )
                capped = clamp_to_cap(prediction=point, cap_mw=test["cap_mw"])
                outputs.append(
                    test.select("site", "time", "month", "fold", "effective_capacity_mw")
                    .cast({"effective_capacity_mw": pl.Float64})
                    .with_columns(
                        arm=pl.lit(arm),
                        seed=pl.lit(seed, dtype=pl.Int32),
                        absolute_error_capped_mw=pl.Series(
                            np.abs(actual - capped), dtype=pl.Float64
                        ),
                    )
                )
    return pl.concat(outputs)


def _percentage_of_capacity(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean absolute error as a percentage of effective capacity.

    Args:
        losses: Per-row losses holding every arm.
        arm: The arm to score.

    Returns:
        The mean of each row's error over its generator's capacity, in percentage points.
    """
    rows = losses.filter(pl.col("arm") == arm)
    ratio = (pl.col(METRIC) / pl.col("effective_capacity_mw")).mean()
    return float(rows.select(ratio).item()) * PERCENTAGE_POINTS


def main() -> int:
    """Score every product in every era, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment", choices=("as-labelled", "shifted", "piecewise"), default="piecewise"
    )
    arguments = parser.parse_args()

    common = _add_time_features(
        dataset=drop_commissioning_ramp(dataset=_joined(alignment=arguments.alignment))
    )
    _LOG.info(
        "common rows: %s, sites: %d, months: %d",
        f"{common.height:,}",
        common["site"].n_unique(),
        common["month"].n_unique(),
    )

    lines = [
        f"### Four products on one common row set, before and after the {UPGRADE_MONTH} boundary",
        "",
        "| Era | Rows | Months | " + " | ".join(PRODUCTS) + " |",
        "|---" * (len(PRODUCTS) + 3) + "|",
    ]
    contrast_lines = [
        "| Era | Contrast | ΔMAE (pp of capacity) | 95% interval | excludes zero? |",
        "|---|---|---|---|---|",
    ]
    output_dir = REPO_DATA_DIR / "ERA5" / f"beam_diffuse_eras_{arguments.alignment}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for era in ("pre_all", "pre_matched", "post"):
        rows = with_export_cap(dataset=_assign_folds(dataset=_era(frame=common, era=era)))
        _LOG.info("%s: %s rows, %d months", era, f"{rows.height:,}", rows["month"].n_unique())
        losses = pl.concat([_losses(rows=rows, arm=arm) for arm in PRODUCTS])
        losses.write_parquet(output_dir / f"losses_{era}.parquet")
        _LOG.info("%s: fitted every arm", era)

        scores = {arm: _percentage_of_capacity(losses=losses, arm=arm) for arm in PRODUCTS}
        lines.append(
            f"| {era} | {rows.height:,} | {rows['month'].n_unique()} | "
            + " | ".join(f"{scores[arm]:.3f}" for arm in PRODUCTS)
            + " |"
        )
        capacity = float(losses.select(pl.col("effective_capacity_mw").mean()).item())
        scale = PERCENTAGE_POINTS / capacity
        for treatment, reference in CONTRASTS:
            interval = _bootstrap_difference(
                losses=losses, treatment=treatment, reference=reference, metric=METRIC
            )
            excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
            contrast_lines.append(
                f"| {era} | {treatment} − {reference} | {interval['difference'] * scale:+.4f} | "
                f"[{interval['lower_95'] * scale:+.4f}, {interval['upper_95'] * scale:+.4f}] | "
                f"{'**yes**' if excludes else 'no'} |"
            )

    report = (
        "\n".join(
            [*lines, "", "Mean absolute error as a percentage of capacity.", "", *contrast_lines]
        )
        + "\n"
    )
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
