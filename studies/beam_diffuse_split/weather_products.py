"""Score six weather products as descriptions of past sunshine, on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-the-past/>.

**Every product is shown to the same booster, on the same rows, with the same temperature.** Each
arm differs from the others only in its irradiance columns, so a contrast between two arms is a
contrast between two products' irradiance. The products are CAMS (a satellite retrieval), ERA5 (a
reanalysis), UKV, ICON-D2, ICON-EU and ICON global (weather models).

**The rows are chosen by nothing any product says.** Each per-source build drops an hour with a
zero half-hour only where its own irradiance reads bright, so a plain inner join would drop an hour
whenever any one product read it as bright. This script instead drops every hour holding a zero
half-hour, recomputed from the power table, and reads CAMS from the build that keeps every hour
rather than only those CAMS itself rates as reliable.

**The folds are cut inside each era of the UKV record, and every arm is told the era.** The Met
Office upgraded UKV on 21 January 2026. Folds cut from each site's whole span would put every
post-upgrade row in the last fold, scored by a model that never saw post-upgrade UKV. Cutting inside
each era puts both eras in every fold, and the `era_code` feature lets one pooled model learn that
a product's mapping may have changed. A separate fit on the post-upgrade rows alone is run as a
sensitivity check.

**Three instruments answer three questions.** The global arms (`<product>_global`) rank the
products on how well their global irradiance predicts power after per-site recalibration. The split
arms (`<product>_split` against `<product>_erbs`) ask whether a product's published beam/diffuse
split adds anything beyond what a separation model derives from its own global irradiance, which
cancels the gain a model gets from re-encoding alone. The leave-one-site-out arms train on five
sites and score the sixth, which is the situation of a generator with no metered history.

**The forecast lead differs between products and is part of what a consumer receives.** UKV's
archive holds the T+0 analysis; ICON-D2 and ICON-EU hold 1-to-3-hour forecasts; ICON global holds
1-to-6-hour forecasts; ERA5's radiation comes from its own forecasts at 1 to 12 hours. The lead
table breaks the ICON contrasts against ERA5 down by each ICON product's served lead.

Run it with `uv run python studies/beam_diffuse_split/weather_products.py`, after
`build_dataset.py` has been run for `open-meteo`, `ukv`, `icon-d2`, `icon-eu`, `icon-global`, and
for `cams` with `--min-cams-reliability 0 --suffix _allhours`.
"""

import argparse
import concurrent.futures
import logging
import sys
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from build_dataset import _hourly_power, _pv_sites
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from run_experiment import (
    MAX_CONCURRENT_FITS,
    SHARED_FEATURES,
    Job,
    _add_time_features,
    _run_all,
    dataset_path_for,
)
from sources import STUDY_DATA_DIR, point_output_path_for
from studies.bootstrap import bootstrap_difference, per_fold_differences
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    assign_folds,
    clamp_to_cap,
    fit_one_fold,
)

_LOG = logging.getLogger(__name__)

PERCENTAGE_POINTS: Final[float] = 100.0

OUTPUT_DIR_NAME: Final[str] = "beam_diffuse_weather_products"
"""The results directory under `STUDY_DATA_DIR`."""

PRODUCTS: Final[dict[str, str]] = {
    "cams": "cams_allhours",
    "era5": "open-meteo",
    "ukv": "ukv",
    "icon_d2": "icon-d2",
    "icon_eu": "icon-eu",
    "icon_global": "icon-global",
}
"""Arm prefix to the build `build_dataset.py` wrote, for every product compared."""

BASE_PRODUCT: Final[str] = "era5"
"""The product whose shared columns - power, capacity, geometry, temperature - the frame keeps.

ERA5 covers every hour the others do, and every per-site build already takes its temperature from
ERA5, so the base supplies nothing a contestant does not already share.
"""

SERVED_LEAD: Final[dict[str, str]] = {
    "cams": "no forecast step (satellite retrieval)",
    "era5": "1 to 12 hours (its own forecasts from 06 and 18 UTC)",
    "ukv": "T+0 (the analysis)",
    "icon_d2": "1 to 3 hours",
    "icon_eu": "1 to 3 hours",
    "icon_global": "1 to 6 hours",
}
"""How far ahead each product's served hourly value was forecast, as the archive holds it."""

RUN_INTERVAL_HOURS: Final[dict[str, int]] = {"icon_d2": 3, "icon_eu": 3, "icon_global": 6}
"""The run cadence of each ICON product, which fixes its served lead at each label hour.

`verify_icon_lineage.py` measured the mapping for ICON-EU against DWD's own files: the freshest run
reproduced Open-Meteo's served value to within 1 W m⁻² at every hour but one. For ICON-D2 the
freshest run was the closest match at 7 of 9 hours but not exactly; the period-3 sawtooth in
ICON-D2's own errors corroborates freshest-run serving across the record. ICON global is published
by DWD only on its icosahedral grid, so its mapping is inferred from the cadence alone.
"""

LEAD_TABLE_HOURS: Final[tuple[int, int]] = (7, 19)
"""The first and last UTC label hours the lead table uses.

UKV's hourly column is its T+0 snapshot rescaled by a ratio of cosines, which runs away at low sun,
and ERA5's lead does not follow a 3-hour cycle, so differencing each ICON product against ERA5 on
the middle of the day isolates the ICON product's own lead.
"""

UPGRADE_MONTH: Final[str] = "2026-02"
"""The first whole month after the Met Office made the PS47 upgrade operational on 2026-01-21."""

UPGRADE_DAY: Final[datetime] = datetime(2026, 1, 21, tzinfo=UTC)
"""The upgrade instant.

The rows from here to the end of January carry the pre-upgrade month label but post-upgrade UKV,
so they are dropped from every arm.
"""

UKV_LIVE_INGEST: Final[datetime] = datetime(2024, 8, 12, tzinfo=UTC)
"""When Open-Meteo's own UKV downloader started.

Earlier UKV in the archive is a backfill from a source Open-Meteo does not name.
"""

ICON_EU_CORRUPT_BLOCK: Final[tuple[datetime, datetime]] = (
    datetime(2023, 6, 21, 1, tzinfo=UTC),
    datetime(2023, 6, 21, 6, tzinfo=UTC),
)
"""ICON-EU's one known corrupt block, dropped from every arm (see `sources.OPEN_METEO_MODELS`)."""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_global", "icon_d2_global"),
    ("icon_eu_global", "icon_d2_global"),
    ("icon_eu_global", "ukv_global"),
    ("icon_global_global", "icon_eu_global"),
)
"""The four contrasts the recommendations rest on, named before the run.

Whether a satellite retrieval beats the best weather model; what the Great-Britain-wide ICON costs
against the regional one; which Great-Britain-wide weather model is better; and what the global
ICON costs against the European one. Every other contrast in the report is exploratory.
"""

UKV_SNAPSHOT_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "ukv_trap_global": ("ghi_trap_ukv",),
    "ukv_pair_global": ("ghi_instant_previous_ukv", "ghi_instant_ukv"),
}
"""Two further UKV arms built from its instantaneous snapshots rather than its served hourly value.

UKV publishes an instantaneous field each hour, and Open-Meteo's served hourly value is the
snapshot at the hour's end rescaled by a ratio of cosines, where every ICON product serves a true
mean over the hour. `ukv_trap_global` is the mean of the snapshots at both ends of the hour, and
`ukv_pair_global` shows the model both snapshots, so a contrast against them separates the weather
model from how its hour is built. Both were added after the first run, on a reviewer's finding, and
are post hoc.
"""

LEAVE_ONE_SITE_OUT_SEED: Final[int] = SEEDS[0]
"""The one seed the leave-one-site-out fits use, which keeps them to one fit per site and fold."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

SCOPES: Final[tuple[str, ...]] = ("all", "pre", "pre_matched", "post", "ukv_live", "cams_reliable")
"""Every scope the pooled losses are bootstrapped over."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
        "| Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|",
)


def _named(column: str, product: str) -> str:
    """Return a product's copy of an irradiance column, such as `bhi_icon_eu`.

    Args:
        column: The build's column name, such as `bhi_w_m2`.
        product: The product prefix.

    Returns:
        The joined frame's column name.
    """
    return f"{column.removesuffix('_w_m2')}_{product}"


def _joined() -> pl.DataFrame:
    """Inner-join every product on the site-hours all of them cover.

    Returns:
        One row per common site-hour, carrying `ghi_<p>`, `bhi_<p>`, `dhi_<p>`, `erbs_bhi_<p>` and
        `erbs_dhi_<p>` for every product `p`, and the base product's power, geometry and
        temperature.
    """
    irradiance = ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2", "erbs_bhi_w_m2", "erbs_dhi_w_m2")
    base = pl.read_parquet(dataset_path_for(source=PRODUCTS[BASE_PRODUCT]))
    frame = base.with_columns(
        pl.col(column).alias(_named(column, BASE_PRODUCT)) for column in irradiance
    ).drop(*irradiance)
    for product, source in PRODUCTS.items():
        if product == BASE_PRODUCT:
            continue
        other = pl.read_parquet(dataset_path_for(source=source)).select(
            "site",
            "time",
            *(pl.col(column).alias(_named(column, product)) for column in irradiance),
        )
        frame = frame.join(other, on=["site", "time"], how="inner")
    return frame.join(_ukv_snapshots(), on=["site", "time"], how="inner").sort("site", "time")


def _ukv_snapshots() -> pl.DataFrame:
    """Return UKV's instantaneous global irradiance at both ends of each hour, and their mean.

    Returns:
        One row per (site, time) with `ghi_instant_previous_ukv`, `ghi_instant_ukv` and
        `ghi_trap_ukv`, for every hour whose both snapshots were served.
    """
    download = pl.read_parquet(point_output_path_for(source="ukv")).select(
        "site", "time", ghi_instant_ukv=pl.col("ghi_instant_w_m2")
    )
    previous = download.select(
        "site",
        time=pl.col("time").dt.offset_by("1h"),
        ghi_instant_previous_ukv=pl.col("ghi_instant_ukv"),
    )
    return download.join(previous, on=["site", "time"], how="inner").with_columns(
        ghi_trap_ukv=(pl.col("ghi_instant_previous_ukv") + pl.col("ghi_instant_ukv")) / 2.0
    )


def _common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows no product should be scored on, by rules no product's values decide.

    Args:
        frame: The joined frame.

    Returns:
        The frame without zero-half-hour hours, ICON-EU's corrupt block, the post-upgrade tail of
        January 2026, and the commissioning ramp.
    """
    zero_hours = (
        _hourly_power(sites=_pv_sites()).filter(pl.col("has_zero_half_hour")).select("site", "time")
    )
    start, end = ICON_EU_CORRUPT_BLOCK
    february = datetime(2026, 2, 1, tzinfo=UTC)
    return drop_commissioning_ramp(
        dataset=frame.join(zero_hours, on=["site", "time"], how="anti")
        .filter(
            ~pl.col("time").is_between(start, end),
            ~pl.col("time").is_between(UPGRADE_DAY, february, closed="left"),
        )
        .sort("site", "time")
    )


def _with_eras(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Label each row's era, add the era feature, and cut folds inside each era.

    Args:
        frame: The common rows, carrying `month`.

    Returns:
        The frame with `era`, `era_code` and `fold`.
    """
    post = pl.col("month") >= UPGRADE_MONTH
    labelled = frame.with_columns(
        era=pl.when(post).then(pl.lit("post")).otherwise(pl.lit("pre")),
        era_code=post.cast(pl.Int8),
    )
    return assign_folds(dataset=labelled, by=("site", "era"))


def _jobs() -> list[Job]:
    """Return the three arms per product: global only, its own split, and Erbs on its own global.

    Returns:
        One job per arm, every arm shown the shared features and the era.
    """
    shared = (*SHARED_FEATURES, "era_code")
    jobs: list[Job] = []
    for product in PRODUCTS:
        ghi = _named("ghi_w_m2", product)
        arms = {
            f"{product}_global": (ghi,),
            f"{product}_split": (ghi, _named("bhi_w_m2", product), _named("dhi_w_m2", product)),
            f"{product}_erbs": (
                ghi,
                _named("erbs_bhi_w_m2", product),
                _named("erbs_dhi_w_m2", product),
            ),
        }
        jobs += [
            (arm, "pooled", "power_mw", (*shared, *columns), PRIMARY_HYPER_PARAMETERS, False)
            for arm, columns in arms.items()
        ]
    jobs += [
        (arm, "pooled", "power_mw", (*shared, *columns), PRIMARY_HYPER_PARAMETERS, False)
        for arm, columns in UKV_SNAPSHOT_ARMS.items()
    ]
    return jobs


def _post_only_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Fit every global arm on the post-upgrade rows alone, as the separate-era sensitivity check.

    Args:
        frame: The common rows with `era`.

    Returns:
        Losses for every global arm, scored on the post-upgrade rows by models trained on them only.
    """
    post = assign_folds(dataset=frame.filter(pl.col("era") == "post").drop("fold"))
    jobs: list[Job] = [
        (
            f"{product}_global",
            "post_only",
            "power_mw",
            (*SHARED_FEATURES, _named("ghi_w_m2", product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for product in PRODUCTS
    ]
    return _run_all(dataset=post, jobs=jobs)


def _leave_one_site_out_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Train on five sites' capacity-normalised power and score the sixth, one fold at a time.

    The scored site is never trained on, and neither are the scored fold's calendar months at any
    other site: the six sites share their weather, so a model trained on a neighbour's power in the
    same hours would learn each day's outcome rather than transfer. This is the withholding
    `run_experiment._add_learned_split` applies for the same reason. The target is a fraction of
    capacity, so five sites can share one model, and the site's own capacity is assumed known.

    Args:
        frame: The common rows, carrying `fold`, `month`, `constrained` and `cap_mw`.

    Returns:
        One row per (site, time, arm) with the capped error as a fraction of capacity.
    """
    frame = frame.with_columns(power_fraction=pl.col("power_mw") / pl.col("effective_capacity_mw"))
    folds = frame.select("site", "fold").unique().sort("site", "fold").rows()

    def _one(product: str, site: str, fold: int) -> pl.DataFrame:
        test = frame.filter((pl.col("site") == site) & (pl.col("fold") == fold))
        train = frame.filter(
            (pl.col("site") != site)
            & ~pl.col("constrained")
            & ~pl.col("month").is_in(test["month"].unique().to_list())
        )
        point, _ = fit_one_fold(
            train=train,
            test=test,
            features=[*SHARED_FEATURES, "era_code", _named("ghi_w_m2", product)],
            target="power_fraction",
            hyper_parameters=PRIMARY_HYPER_PARAMETERS,
            seed=LEAVE_ONE_SITE_OUT_SEED,
            with_quantiles=False,
        )
        capacity = test["effective_capacity_mw"].cast(pl.Float64).to_numpy()
        capped = clamp_to_cap(prediction=point * capacity, cap_mw=test["cap_mw"]) / capacity
        error = np.abs(test["power_fraction"].cast(pl.Float64).to_numpy() - capped)
        return test.select("site", "time", "month", "fold", "era").with_columns(
            pl.Series(METRIC, error),
            arm=pl.lit(f"{product}_global"),
            seed=pl.lit(LEAVE_ONE_SITE_OUT_SEED, dtype=pl.Int32),
        )

    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        futures = [
            pool.submit(_one, product, site, fold) for product in PRODUCTS for site, fold in folds
        ]
        outputs.extend(future.result() for future in concurrent.futures.as_completed(futures))
    return pl.concat(outputs)


def _yield_stability(*, frame: pl.DataFrame) -> dict[str, float]:
    """Measure how steady each product's monthly ratio of power to irradiance is.

    Capacity estimation reads a product's irradiance with no model fitted to the site, so a product
    whose bias drifts from month to month misleads it however well a recalibrated model scores.
    For each site and month the ratio of summed power to summed irradiance is taken, its
    month-to-month changes are divided by its median, and the median absolute change is pooled
    across sites. Season moves every product's ratio alike, so the comparison between products is
    what the number is for, not its level.

    Args:
        frame: The common rows.

    Returns:
        Each product's median absolute month-to-month change in its yield ratio, as a fraction.
    """
    stability: dict[str, float] = {}
    for product in PRODUCTS:
        monthly = (
            frame.group_by("site", "month")
            .agg(ratio=pl.col("power_mw").sum() / pl.col(_named("ghi_w_m2", product)).sum())
            .sort("site", "month")
            .with_columns(
                change=(pl.col("ratio").diff().abs() / pl.col("ratio").median()).over("site")
            )
        )
        stability[product] = float(monthly.select(pl.col("change").drop_nulls().median()).item())
    return stability


def _scope(*, losses: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Restrict the losses to one scope of the pooled run.

    Args:
        losses: Per-row losses carrying `month` and `time`.
        scope: One of `SCOPES`.

    Returns:
        The rows belonging to that scope.

    Raises:
        ValueError: If `scope` is not one of `SCOPES`.
    """
    losses = losses.with_columns(
        era=pl.when(pl.col("month") >= UPGRADE_MONTH).then(pl.lit("post")).otherwise(pl.lit("pre"))
    )
    if scope == "all":
        return losses
    if scope in ("pre", "post"):
        return losses.filter(pl.col("era") == scope)
    if scope == "pre_matched":
        post_months = {label[-2:] for label in losses.filter(pl.col("era") == "post")["month"]}
        return losses.filter(
            (pl.col("era") == "pre") & pl.col("month").str.slice(-2).is_in(sorted(post_months))
        )
    if scope == "ukv_live":
        return losses.filter(pl.col("time") >= UKV_LIVE_INGEST)
    if scope == "cams_reliable":
        reliable = pl.read_parquet(dataset_path_for(source="cams")).select("site", "time")
        return losses.join(reliable, on=["site", "time"], how="semi")
    msg = f"unknown scope {scope}"
    raise ValueError(msg)


def _mae(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean error, in percentage points of capacity.

    Args:
        losses: Per-row losses.
        arm: The arm to score.

    Returns:
        The mean of each row's error over its generator's capacity, in percentage points.
    """
    rows = losses.filter(pl.col("arm") == arm)
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def _contrast_line(*, losses: pl.DataFrame, treatment: str, reference: str, label: str) -> str:
    """Return one markdown row: the difference, its interval, and the folds agreeing in sign.

    Args:
        losses: Per-row losses holding both arms.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.
        label: The scope label for the first column.

    Returns:
        The table row.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    same_sign = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
    return (
        f"| {label} | {treatment} − {reference} | {difference:+.3f} | "
        f"[{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} | "
        f"{same_sign} of {len(folds)} | {interval['n_rows']:,} |"
    )


def _served_lead(*, product: str) -> pl.Expr:
    """Return the served lead in hours of a product's value at each row's label hour.

    A served hourly value is a backward mean over the hour ending at its label, so the run that
    supplies label hour `T` is the latest one at or before `T - 1`, and the lead is `T` minus that
    run: 1, 2 or 3 hours for a 3-hourly model, 1 to 6 for a 6-hourly one.

    Args:
        product: A key of `RUN_INTERVAL_HOURS`.

    Returns:
        The lead, as an integer expression.
    """
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    return ((hour - 1) % RUN_INTERVAL_HOURS[product]) + 1


def _lead_tables(*, losses: pl.DataFrame) -> list[str]:
    """Compare ICON products at matched served leads, and list each hour's contrast against ERA5.

    ICON-D2 and ICON-EU run on the same 3-hourly cycle, so every label hour holds them at the same
    lead and their contrast within a lead bucket is lead-matched. ICON global's lead equals
    ICON-EU's wherever its own lead is 3 hours or less. The per-hour table carries the time of day
    as well as the lead, which is why it is listed rather than bucketed.

    Args:
        losses: The pooled losses.

    Returns:
        Markdown lines.
    """
    first, last = LEAD_TABLE_HOURS
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    daytime = losses.filter(hour.is_between(first, last)).with_columns(
        lead_3h=_served_lead(product="icon_eu"), lead_6h=_served_lead(product="icon_global")
    )
    label = f"{first:02d}–{last:02d} UTC"
    lines = ["#### ICON-D2 against ICON-EU at matched served leads", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=daytime.filter(pl.col("lead_3h") == lead),
            treatment="icon_d2_global",
            reference="icon_eu_global",
            label=f"both at lead {lead} h, {label}",
        )
        for lead in (1, 2, 3)
    ]
    lines += ["", "#### ICON global against ICON-EU, split by ICON global's lead", ""]
    lines += [*CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=daytime.filter(condition),
            treatment="icon_global_global",
            reference="icon_eu_global",
            label=f"{name}, {label}",
        )
        for name, condition in (
            ("ICON global lead 1 to 3 h, equal to ICON-EU's", pl.col("lead_6h") <= 3),
            ("ICON global lead 4 to 6 h", pl.col("lead_6h") > 3),
        )
    ]
    by_hour = (
        daytime.filter(pl.col("arm").is_in(["era5_global", "icon_d2_global", "icon_eu_global"]))
        .group_by("time", "site", "seed", "arm")
        .agg(pl.col(METRIC).first())
        .pivot(on="arm", index=["time", "site", "seed"], values=METRIC)
        .group_by(hour.alias("hour"))
        .agg(
            icon_d2=(pl.col("icon_d2_global") - pl.col("era5_global")).mean(),
            icon_eu=(pl.col("icon_eu_global") - pl.col("era5_global")).mean(),
        )
        .sort("hour")
    )
    lines += [
        "",
        "#### Each hour's contrast against ERA5 (pp of capacity; point estimates)",
        "",
        "| Hour (UTC) | Served lead, ICON-D2 and ICON-EU | ICON-D2 − ERA5 | ICON-EU − ERA5 |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['hour']:02d} | {((row['hour'] - 1) % 3) + 1} h "
        f"| {row['icon_d2'] * PERCENTAGE_POINTS:+.3f} | {row['icon_eu'] * PERCENTAGE_POINTS:+.3f} |"
        for row in by_hour.iter_rows(named=True)
    ]
    return lines


def _report(
    *,
    frame: pl.DataFrame,
    pooled: pl.DataFrame,
    post_only: pl.DataFrame,
    transfer: pl.DataFrame,
    stability: dict[str, float],
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        pooled: The pooled run's losses.
        post_only: The post-upgrade-only run's losses.
        transfer: The leave-one-site-out losses.
        stability: Each product's month-to-month yield-ratio change.

    Returns:
        The report.
    """
    lines = [
        (
            f"### Six weather products on {frame.height:,} common site-hours "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        (
            "| Product | Served lead | Global only | Own split | Erbs on own global "
            "| Leave one site out |"
        ),
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {product} | {SERVED_LEAD[product]} "
        f"| {_mae(losses=pooled, arm=f'{product}_global'):.3f} "
        f"| {_mae(losses=pooled, arm=f'{product}_split'):.3f} "
        f"| {_mae(losses=pooled, arm=f'{product}_erbs'):.3f} "
        f"| {_mae(losses=transfer, arm=f'{product}_global'):.3f} |"
        for product in PRODUCTS
    ]
    lines += ["", "Mean absolute error as a percentage of each site's P99 output.", ""]
    lines += ["#### Deciding contrasts, named before the run", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=pooled, treatment=treatment, reference=reference, label="all")
        for treatment, reference in DECIDING_CONTRASTS
    ]
    lines += ["", "#### Every product against ERA5, by scope (exploratory)", "", *CONTRAST_HEADER]
    for scope in SCOPES:
        scoped = _scope(losses=pooled, scope=scope)
        lines += [
            _contrast_line(
                losses=scoped, treatment=f"{product}_global", reference="era5_global", label=scope
            )
            for product in PRODUCTS
            if product != BASE_PRODUCT
        ]
    lines += [
        "",
        (
            "The post scope holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        "#### The post scope, fitted on post-upgrade rows alone",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=post_only, treatment=treatment, reference=reference, label="post")
        for treatment, reference in (*DECIDING_CONTRASTS, ("ukv_global", "era5_global"))
    ]
    lines += ["", "#### A product's own split against Erbs on its own global", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=pooled, treatment=f"{product}_split", reference=f"{product}_erbs", label="all"
        )
        for product in PRODUCTS
    ]
    lines += [
        "",
        "#### Leave one site out, the scored months withheld everywhere: the deciding contrasts",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=transfer, treatment=treatment, reference=reference, label="all")
        for treatment, reference in DECIDING_CONTRASTS
    ]
    lines += [
        "",
        "#### UKV's served hour against the mean of its own two snapshots (post hoc)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=scoped, treatment=treatment, reference=reference, label=scope)
        for scope in ("all", "ukv_live")
        for scoped in (_scope(losses=pooled, scope=scope),)
        for treatment, reference in (
            ("ukv_trap_global", "ukv_global"),
            ("icon_eu_global", "ukv_trap_global"),
            ("icon_eu_global", "ukv_pair_global"),
        )
    ]
    lines += [
        "",
        (
            f"MAE: ukv_trap_global {_mae(losses=pooled, arm='ukv_trap_global'):.3f}, "
            f"ukv_pair_global {_mae(losses=pooled, arm='ukv_pair_global'):.3f}."
        ),
        "",
        "#### Month-to-month change in each product's ratio of power to irradiance",
        "",
        "| Product | Median absolute monthly change |",
        "|---|---|",
    ]
    lines += [f"| {product} | {value:.3f} |" for product, value in stability.items()]
    lines += ["", *_lead_tables(losses=pooled)]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    frame = with_export_cap(
        dataset=_with_eras(frame=_add_time_features(dataset=_common_rows(frame=_joined())))
    )
    by_site = frame.group_by("site", "era").agg(pl.len(), pl.col("month").n_unique()).sort("site")
    _LOG.info("common rows: %d\n%s", frame.height, by_site)

    output_dir = STUDY_DATA_DIR / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled = _run_all(dataset=frame, jobs=_jobs())
    pooled.write_parquet(output_dir / "losses.parquet")
    post_only = _post_only_losses(frame=frame)
    post_only.write_parquet(output_dir / "post_only_losses.parquet")
    transfer = _leave_one_site_out_losses(frame=frame)
    transfer.write_parquet(output_dir / "leave_one_site_out_losses.parquet")

    report = _report(
        frame=frame,
        pooled=pooled,
        post_only=post_only,
        transfer=transfer,
        stability=_yield_stability(frame=frame),
    )
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
