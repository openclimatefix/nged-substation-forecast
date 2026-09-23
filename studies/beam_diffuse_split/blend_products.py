"""Measure whether blending weather products beats the best single product, for solar and wind.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/836>. It builds on the two
weather-product studies, `weather_products.py` (solar) and `wind_products.py` (wind), and refits
their single-product arms on their own common rows, folds and seeds, so every contrast here is
paired with theirs.

**The single-product arms must reproduce the published losses row for row before any blend runs.**
The refitted `<product>_global` (solar) and `<product>_wind` (wind) arms are compared against the
two studies' `losses.parquet`: the same (site, time, fold, seed) keys and a bit-identical
`signed_error_capped_mw`. If they differ, the rows or the folds have changed, and the script stops.

**Each set of products is blended four ways**, each against the set's best single product, which is
fixed from the published tables:

- `<set>_xgb`: XGBoost shown every product's columns, with column subsampling at 1.
- `<set>_mean`: XGBoost shown the mean of the products' values, as many columns as one product.
- `<set>_stack`: a linear stack of the single-product models' out-of-fold predictions, with
  non-negative weights summing to 1, cross-fitted per generator, seed and fold by
  `studies.blending.stacked_errors`.
- `<set>_equal`: the equal-weight mean of the single-product predictions.

**Every XGBoost blend has a climatology control, `<set>_control`.** The control holds the best
single product's real columns and every other product's columns permuted among the rows sharing a
site, a month and an hour of day, one permutation per product, so it has the blend's column count
and none of the other products' weather.

**Every blend uses ICON global's wind as served, without the wind study's step indicator.** The
indicator acts as a date-regime feature, which the single-product arms and the stack do not get.
The one exploratory arm `everything_xgb_step` adds it back.

The deciding contrasts, named before the run, are built by `_deciding_contrasts` from each domain's
`named_sets` and `STACK_AGAINST_XGB`. The positive control is `POSITIVE_CONTROL`, and every other
contrast in the report is exploratory.

Run it with `uv run python studies/beam_diffuse_split/blend_products.py`, after both weather-product
studies have been run. `--resume` reuses the per-arm fits a previous run left in `fits/`.
"""

import argparse
import logging
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
import polars as pl
import weather_products
import wind_products
from build_dataset import POWER_DELTA_URI, _wind_sites
from deltalake import DeltaTable
from export_cap import with_export_cap
from run_experiment import SHARED_FEATURES as SOLAR_SHARED_FEATURES
from run_experiment import Job, _add_time_features, _run_all
from sources import STUDY_DATA_DIR
from studies.blending import PERMUTED_SUFFIX, climatology_permutation, stacked_errors
from studies.bootstrap import bootstrap_difference, per_fold_differences
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "blend_products"
"""Where every output of this study is written."""

FITS_DIR_NAME: Final[str] = "fits"
"""The per-arm fits under `OUTPUT_DIR`, kept until the report is written so `--resume` can reuse
them after a crash, and deleted once every output is on disk."""

PERMUTATION_SEED: Final[int] = 20260923
"""The first product's climatology permutation seed; product `i` in a domain's order uses `+ i`."""

PERMUTATION_GROUPS: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""The rows a control column's value may move between: one site, one month, one hour of day."""

PERCENTAGE_POINTS: Final[float] = 100.0

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

SettingType = Literal["pooled", "sensitivity"]
"""The hyperparameter setting. `pooled` is the primary one, as the published studies name it."""

SETTINGS: Final[dict[SettingType, HyperParameters]] = {
    "pooled": PRIMARY_HYPER_PARAMETERS,
    "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
}

DomainType = Literal["solar", "wind"]

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time",
    "month",
    "fold",
    "seed",
    "effective_capacity_mw",
    "constrained",
)
"""The columns identifying a scored row, shared by every arm."""

LOSS_COLUMNS: Final[tuple[str, ...]] = (
    *KEY_COLUMNS,
    "signed_error_capped_mw",
    METRIC,
    "arm",
    "setting",
)
"""The columns `losses.parquet` keeps for every arm, fitted or derived."""


@dataclass(frozen=True)
class BlendSet:
    """A set of products blended together, and the single product it has to beat."""

    name: str
    products: tuple[str, ...]
    best_single: str
    consumer: str


@dataclass(frozen=True)
class Domain:
    """Everything that differs between the solar and the wind halves of the study."""

    name: DomainType
    products: tuple[str, ...]
    sets: tuple[BlendSet, ...]
    shared_features: tuple[str, ...]
    columns: Callable[[str], tuple[str, ...]]
    single_suffix: str
    named_sets: tuple[str, ...]
    published_losses: Path
    published_settings: tuple[SettingType, ...]

    def single(self, product: str) -> str:
        """Return a product's single-product arm, named as the published study names it.

        Args:
            product: The product.

        Returns:
            The arm name.
        """
        return f"{product}{self.single_suffix}"

    def blend_set(self, name: str) -> BlendSet:
        """Return the set with this name.

        Args:
            name: The set's name.

        Returns:
            The set.
        """
        return next(blend for blend in self.sets if blend.name == name)


def _solar_columns(product: str) -> tuple[str]:
    """Return a solar product's one feature column: its global horizontal irradiance.

    Args:
        product: A key of `weather_products.PRODUCTS`.

    Returns:
        The column name.
    """
    return (f"ghi_{product}",)


def _wind_columns(product: str) -> tuple[str, str, str, str]:
    """Return a wind product's four feature columns, as the wind study names them.

    Args:
        product: A key of `fetch_wind_point.PRODUCTS`.

    Returns:
        The hub-height speed, that height's direction as sine and cosine, and the 10 m speed.
    """
    return wind_products._wind_columns(product=product)


SOLAR: Final[Domain] = Domain(
    name="solar",
    products=tuple(weather_products.PRODUCTS),
    sets=(
        BlendSet("cams_icon_d2", ("cams", "icon_d2"), "cams", "history, where ICON-D2 covers"),
        BlendSet("cams_icon_eu", ("cams", "icon_eu"), "cams", "history anywhere in Great Britain"),
        BlendSet("cams_era5", ("cams", "era5"), "cams", "history back to 2004"),
        BlendSet("live_gb", ("ukv", "icon_eu"), "icon_eu", "live service, Great-Britain-wide"),
        BlendSet(
            "live_all",
            ("ukv", "icon_d2", "icon_eu", "icon_global"),
            "icon_d2",
            "live service, where ICON-D2 covers (weather models only)",
        ),
        BlendSet("everything", tuple(weather_products.PRODUCTS), "cams", "the upper bound"),
    ),
    shared_features=(*SOLAR_SHARED_FEATURES, "era_code"),
    columns=_solar_columns,
    single_suffix="_global",
    named_sets=("everything", "cams_icon_eu", "live_all"),
    published_losses=STUDY_DATA_DIR / weather_products.OUTPUT_DIR_NAME / "losses.parquet",
    published_settings=("pooled",),
)

WIND: Final[Domain] = Domain(
    name="wind",
    products=("era5", "ukv", "icon_d2", "icon_eu", "icon_global"),
    sets=(
        BlendSet("best_pair", ("icon_d2", "ukv"), "icon_d2", "the two leaders"),
        BlendSet("live_gb", ("ukv", "icon_eu"), "ukv", "live service, Great-Britain-wide"),
        BlendSet(
            "live_all",
            ("ukv", "icon_d2", "icon_eu", "icon_global"),
            "icon_d2",
            "live service, where ICON-D2 covers",
        ),
        BlendSet(
            "everything",
            ("era5", "ukv", "icon_d2", "icon_eu", "icon_global"),
            "icon_d2",
            "the upper bound, inside ICON-D2's domain",
        ),
    ),
    shared_features=wind_products.SHARED_FEATURES,
    columns=_wind_columns,
    single_suffix="_wind",
    named_sets=("everything", "live_gb"),
    published_losses=STUDY_DATA_DIR / wind_products.OUTPUT_DIR_NAME / "losses.parquet",
    published_settings=("pooled", "sensitivity"),
)

DOMAINS: Final[tuple[Domain, ...]] = (SOLAR, WIND)

STACK_AGAINST_XGB: Final[tuple[str, str]] = ("everything_stack", "everything_xgb")
"""The deciding stack contrast in each domain: a linear stack against XGBoost on every column."""

POSITIVE_CONTROL: Final[tuple[str, str]] = ("cams_icon_d2_xgb", "icon_d2_global")
"""Solar, where a large gain must appear: ICON-D2 with CAMS against ICON-D2 alone."""

STEP_ARM: Final[str] = "everything_xgb_step"
"""The one exploratory wind arm shown ICON global's step indicator, `step_period`."""


def _mean_columns(*, domain: Domain, blend: BlendSet) -> tuple[str, ...]:
    """Return a set's mean-of-inputs columns, as many as one product's.

    Args:
        domain: The domain.
        blend: The set.

    Returns:
        One column per column of a single product, named `<stem>_mean_<set>`.
    """
    return tuple(
        f"{column.removesuffix(f'_{blend.best_single}')}_mean_{blend.name}"
        for column in domain.columns(blend.best_single)
    )


def _with_blend_columns(*, frame: pl.DataFrame, domain: Domain) -> pl.DataFrame:
    """Add every set's mean columns and every product's climatology-permuted columns.

    The mean of the wind products' direction sines and of their cosines together give the circular
    mean direction. Neither step reorders the rows, which the reproduction check relies on.

    Args:
        frame: The domain's common rows.
        domain: The domain.

    Returns:
        `frame` with the mean and the `_shuffled` columns added.
    """
    means = [
        pl.mean_horizontal(
            *(domain.columns(product)[position] for product in blend.products)
        ).alias(name)
        for blend in domain.sets
        for position, name in enumerate(_mean_columns(domain=domain, blend=blend))
    ]
    return climatology_permutation(
        frame=frame.with_columns(means),
        column_groups=[domain.columns(product) for product in domain.products],
        by=PERMUTATION_GROUPS,
        seed=PERMUTATION_SEED,
    )


def _arm_features(*, domain: Domain) -> dict[str, tuple[str, ...]]:
    """Return every fitted arm's feature columns, in the order the model sees them.

    Args:
        domain: The domain.

    Returns:
        Arm name to feature columns, singles first.
    """
    shared = domain.shared_features
    arms = {
        domain.single(product): (*shared, *domain.columns(product)) for product in domain.products
    }
    for blend in domain.sets:
        every = tuple(column for product in blend.products for column in domain.columns(product))
        permuted = tuple(
            f"{column}{PERMUTED_SUFFIX}"
            for product in blend.products
            if product != blend.best_single
            for column in domain.columns(product)
        )
        arms[f"{blend.name}_xgb"] = (*shared, *every)
        arms[f"{blend.name}_control"] = (*shared, *domain.columns(blend.best_single), *permuted)
        arms[f"{blend.name}_mean"] = (*shared, *_mean_columns(domain=domain, blend=blend))
    if domain.name == "wind":
        arms[STEP_ARM] = (*arms["everything_xgb"], "step_period")
    return arms


def _sensitivity_arms(*, domain: Domain) -> tuple[str, ...]:
    """Return the arms fitted at the second setting: every single, and each named blend and control.

    Every single is fitted so that every stack can be formed at the second setting too.

    Args:
        domain: The domain.

    Returns:
        The arm names.
    """
    return (
        *(domain.single(product) for product in domain.products),
        *(f"{name}_{kind}" for name in domain.named_sets for kind in ("xgb", "control")),
    )


def _jobs(*, domain: Domain, arms: Iterable[str], setting: SettingType) -> list[Job]:
    """Return one fit job per arm at one setting.

    Args:
        domain: The domain.
        arms: The arms to fit.
        setting: The hyperparameter setting.

    Returns:
        The jobs `run_experiment._run_all` takes.
    """
    features = _arm_features(domain=domain)
    return [(arm, setting, "power_mw", features[arm], SETTINGS[setting], False) for arm in arms]


def _fit_path(*, domain: Domain, arm: str, setting: str) -> Path:
    """Return where one arm's fit is kept until the report is written.

    Args:
        domain: The domain.
        arm: The arm.
        setting: The setting.

    Returns:
        The parquet path.
    """
    return OUTPUT_DIR / FITS_DIR_NAME / f"{domain.name}__{setting}__{arm}.parquet"


def _fitted(*, frame: pl.DataFrame, domain: Domain, jobs: list[Job], resume: bool) -> pl.DataFrame:
    """Fit every job not already on disk, keep each arm's losses, and return them all.

    Args:
        frame: The domain's common rows.
        domain: The domain.
        jobs: The fits wanted.
        resume: Whether to reuse a fit a previous run left on disk.

    Returns:
        Every job's losses, one row per (site, time, seed, arm, setting), sorted.
    """
    missing = [
        job
        for job in jobs
        if not (resume and _fit_path(domain=domain, arm=job[0], setting=job[1]).exists())
    ]
    if missing:
        fresh = _run_all(dataset=frame, jobs=missing)
        for arm, setting, *_ in missing:
            path = _fit_path(domain=domain, arm=arm, setting=setting)
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh.filter(pl.col("arm") == arm, pl.col("setting") == setting).sort(
                "site", "time", "seed"
            ).write_parquet(path)
    return pl.concat(
        pl.read_parquet(_fit_path(domain=domain, arm=arm, setting=setting))
        for arm, setting, *_ in jobs
    )


class ReproductionRow(TypedDict):
    """One single-product arm's comparison against the published losses."""

    domain: str
    setting: str
    arm: str
    rows: int
    published_rows: int
    keys_equal: bool
    bit_identical: bool
    max_abs_difference_mw: float


def _reproduction(*, fitted: pl.DataFrame, domain: Domain) -> list[ReproductionRow]:
    """Compare every refitted single-product arm with the published study's losses.

    Args:
        fitted: The refitted single-product arms' losses.
        domain: The domain.

    Returns:
        One row per arm and published setting.
    """
    published = pl.read_parquet(domain.published_losses)
    rows: list[ReproductionRow] = []
    for setting in domain.published_settings:
        for product in domain.products:
            arm = domain.single(product)
            ours, theirs = (
                losses.filter(pl.col("arm") == arm, pl.col("setting") == setting)
                .sort("site", "time", "fold", "seed")
                .select("site", "time", "fold", "seed", "signed_error_capped_mw")
                for losses in (fitted, published)
            )
            keys = ["site", "time", "fold", "seed"]
            keys_equal = ours.height == theirs.height and ours.select(keys).equals(
                theirs.select(keys)
            )
            identical = keys_equal and ours["signed_error_capped_mw"].equals(
                theirs["signed_error_capped_mw"]
            )
            difference = (
                float(
                    np.max(
                        np.abs(
                            ours["signed_error_capped_mw"].to_numpy()
                            - theirs["signed_error_capped_mw"].to_numpy()
                        )
                    )
                )
                if keys_equal
                else float("nan")
            )
            rows.append(
                {
                    "domain": domain.name,
                    "setting": setting,
                    "arm": arm,
                    "rows": ours.height,
                    "published_rows": theirs.height,
                    "keys_equal": keys_equal,
                    "bit_identical": identical,
                    "max_abs_difference_mw": difference,
                }
            )
    return rows


def _reproduction_lines(*, rows: list[ReproductionRow]) -> list[str]:
    """Render the reproduction check as a markdown table.

    Args:
        rows: The comparisons.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            "| Domain | Setting | Arm | Rows | Published rows | Keys equal | Bit-identical "
            "`signed_error_capped_mw` | Largest difference (MW) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['domain']} | {row['setting']} | {row['arm']} | {row['rows']:,} "
        f"| {row['published_rows']:,} | {'yes' if row['keys_equal'] else '**no**'} "
        f"| {'yes' if row['bit_identical'] else '**no**'} | {row['max_abs_difference_mw']:.3g} |"
        for row in rows
    ]
    return lines


def _wide_errors(
    *, losses: pl.DataFrame, arms: Iterable[str], setting: SettingType
) -> pl.DataFrame:
    """Pivot the named arms' capped signed errors to one column per arm.

    Args:
        losses: Fitted losses.
        arms: The arms to pivot.
        setting: The setting.

    Returns:
        One row per (site, time, seed) with the key columns and one error column per arm.
    """
    arms = list(arms)
    return (
        losses.filter(pl.col("arm").is_in(arms), pl.col("setting") == setting)
        .pivot(on="arm", index=list(KEY_COLUMNS), values="signed_error_capped_mw")
        .sort("site", "time", "seed")
    )


def _derived_losses(
    *, keys: pl.DataFrame, errors: np.ndarray, arm: str, setting: SettingType
) -> pl.DataFrame:
    """Score a derived arm's signed errors in the loss columns every arm shares.

    Args:
        keys: The key columns, one row per error.
        errors: The derived arm's capped signed error on each row, in MW.
        arm: The derived arm's name.
        setting: The setting of the fits it was derived from.

    Returns:
        One row per scored row, in `LOSS_COLUMNS`.
    """
    return keys.select(KEY_COLUMNS).with_columns(
        signed_error_capped_mw=pl.Series(errors, dtype=pl.Float64),
        **{METRIC: pl.Series(np.abs(errors), dtype=pl.Float64) / pl.col("effective_capacity_mw")},
        arm=pl.lit(arm),
        setting=pl.lit(setting),
    )


def _weights_frame(
    *,
    keys: pl.DataFrame,
    weights: np.ndarray,
    models: list[str],
    labels: dict[str, str],
) -> pl.DataFrame:
    """Return each (site, seed, fold)'s weight on each model, one row per model.

    Args:
        keys: The key columns, one row per weight row.
        weights: The weights each row was scored with.
        models: The model each weight column belongs to.
        labels: Constant columns to add, such as the set and the fitting.

    Returns:
        One row per (site, seed, fold, model).
    """
    return (
        keys.select("site", "seed", "fold")
        .hstack(pl.DataFrame(weights, schema=models, orient="row"))
        .unique(["site", "seed", "fold"])
        .unpivot(index=["site", "seed", "fold"], variable_name="model", value_name="weight")
        .with_columns(**{name: pl.lit(value) for name, value in labels.items()})
    )


def _stacks(
    *, losses: pl.DataFrame, domain: Domain, setting: SettingType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack, in-sample stack and equal-weight-average every set's single-product models.

    Args:
        losses: The fitted losses, carrying every single-product arm at `setting`.
        domain: The domain.
        setting: The setting.

    Returns:
        The derived arms' losses, and every stack's weights.
    """
    singles = [domain.single(product) for product in domain.products]
    wide = _wide_errors(losses=losses, arms=singles, setting=setting)
    derived: list[pl.DataFrame] = []
    weights: list[pl.DataFrame] = []
    for blend in domain.sets:
        models = [domain.single(product) for product in blend.products]
        errors = wide.select(models).to_numpy()
        for suffix, cross_fitted in (("stack", True), ("stack_in_sample", False)):
            result = stacked_errors(
                errors=errors,
                sites=wide["site"].to_numpy(),
                folds=wide["fold"].to_numpy(),
                seeds=wide["seed"].to_numpy(),
                fit_rows=~wide["constrained"].to_numpy(),
                cross_fitted=cross_fitted,
            )
            arm = f"{blend.name}_{suffix}"
            derived.append(
                _derived_losses(keys=wide, errors=result.errors, arm=arm, setting=setting)
            )
            weights.append(
                _weights_frame(
                    keys=wide,
                    weights=result.weights,
                    models=models,
                    labels={"domain": domain.name, "setting": setting, "arm": arm},
                )
            )
        derived.append(
            _derived_losses(
                keys=wide, errors=errors.mean(axis=1), arm=f"{blend.name}_equal", setting=setting
            )
        )
    return pl.concat(derived), pl.concat(weights)


def _seed_stacks(*, losses: pl.DataFrame, domain: Domain) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack each best single product's three seeds, which measures what ensembling alone gives.

    The stacked error is one value per row, repeated under each seed so the paired bootstrap can
    draw a seed for the reference arm as it always does.

    Args:
        losses: The fitted losses at the primary setting.
        domain: The domain.

    Returns:
        The `<single>_seed_stack` arms' losses, and their weights.
    """
    derived: list[pl.DataFrame] = []
    weights: list[pl.DataFrame] = []
    for product in sorted({blend.best_single for blend in domain.sets}):
        single = domain.single(product)
        by_seed = (
            losses.filter(pl.col("arm") == single, pl.col("setting") == "pooled")
            .with_columns(seed_name=pl.format("seed_{}", pl.col("seed")))
            .pivot(
                on="seed_name",
                index=[c for c in KEY_COLUMNS if c != "seed"],
                values="signed_error_capped_mw",
            )
            .sort("site", "time")
        )
        models = sorted(c for c in by_seed.columns if c.startswith("seed_"))
        result = stacked_errors(
            errors=by_seed.select(models).to_numpy(),
            sites=by_seed["site"].to_numpy(),
            folds=by_seed["fold"].to_numpy(),
            seeds=np.zeros(by_seed.height, dtype=np.int64),
            fit_rows=~by_seed["constrained"].to_numpy(),
        )
        arm = f"{single}_seed_stack"
        keys = by_seed.with_columns(seed=pl.lit(-1, dtype=pl.Int32))
        weights.append(
            _weights_frame(
                keys=keys,
                weights=result.weights,
                models=models,
                labels={"domain": domain.name, "setting": "pooled", "arm": arm},
            )
        )
        derived.extend(
            _derived_losses(
                keys=by_seed.with_columns(seed=pl.lit(seed, dtype=pl.Int32)),
                errors=result.errors,
                arm=arm,
                setting="pooled",
            )
            for seed in sorted(losses["seed"].unique().to_list())
        )
    return pl.concat(derived), pl.concat(weights)


class IntervalRecord(TypedDict):
    """One contrast's interval, as `intervals.parquet` holds it."""

    domain: str
    setting: str
    section: str
    scope: str
    treatment: str
    reference: str
    difference_pp: float
    lower_95_pp: float
    upper_95_pp: float
    seed_spread_pp: float
    excludes_zero: bool
    folds_agreeing: int
    n_folds: int
    n_rows: int
    n_months: int


def _interval(
    *,
    losses: pl.DataFrame,
    contrast: tuple[str, str],
    domain: Domain,
    setting: str,
    section: str,
    scope: str = "all",
) -> IntervalRecord:
    """Bootstrap one paired contrast and count the folds agreeing with its sign.

    Args:
        losses: Losses holding both arms, restricted to the scope.
        contrast: The treatment and the reference arm.
        domain: The domain.
        setting: The setting.
        section: Which part of the report the contrast belongs to.
        scope: The label of the rows `losses` holds.

    Returns:
        The interval, in percentage points of capacity.
    """
    treatment, reference = contrast
    pair = losses.filter(pl.col("arm").is_in([treatment, reference]))
    interval = bootstrap_difference(
        losses=pair, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=pair, treatment=treatment, reference=reference, metric=METRIC
    )
    return {
        "domain": domain.name,
        "setting": setting,
        "section": section,
        "scope": scope,
        "treatment": treatment,
        "reference": reference,
        "difference_pp": interval["difference"] * PERCENTAGE_POINTS,
        "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
        "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
        "seed_spread_pp": interval["seed_spread"] * PERCENTAGE_POINTS,
        "excludes_zero": interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0,
        "folds_agreeing": sum(np.sign(value) == np.sign(interval["difference"]) for value in folds),
        "n_folds": len(folds),
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }


CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Domain | Setting | Scope | Contrast | ΔMAE (pp of capacity) | 95% interval "
        "| Excludes zero? | Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|---|---|",
)


def _line(record: IntervalRecord) -> str:
    """Render one interval as a markdown row.

    Args:
        record: The interval.

    Returns:
        The row.
    """
    return (
        f"| {record['domain']} | {record['setting']} | {record['scope']} "
        f"| {record['treatment']} − {record['reference']} | {record['difference_pp']:+.3f} "
        f"| [{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}] "
        f"| {'**yes**' if record['excludes_zero'] else 'no'} "
        f"| {record['folds_agreeing']} of {record['n_folds']} | {record['n_rows']:,} |"
    )


def _section_lines(
    *, records: list[IntervalRecord], section: str, setting: str | None = None
) -> list[str]:
    """Render every interval in one section of the report, optionally at one setting only.

    Args:
        records: Every interval.
        section: The section to render.
        setting: The setting to keep, or `None` for both.

    Returns:
        Markdown rows.
    """
    return [
        _line(record)
        for record in records
        if record["section"] == section and setting in (None, record["setting"])
    ]


def _deciding_contrasts(*, domain: Domain) -> list[tuple[str, str]]:
    """Return a domain's deciding contrasts, and blend against single beside each pair.

    Args:
        domain: The domain.

    Returns:
        (treatment, reference) pairs: per named set, blend against control, control against single,
        and blend against single; then the stack against XGBoost on every column.
    """
    contrasts: list[tuple[str, str]] = []
    for name in domain.named_sets:
        single = domain.single(domain.blend_set(name).best_single)
        contrasts += [
            (f"{name}_xgb", f"{name}_control"),
            (f"{name}_control", single),
            (f"{name}_xgb", single),
        ]
    return [*contrasts, STACK_AGAINST_XGB]


def _exploratory_contrasts(*, domain: Domain) -> list[tuple[str, str]]:
    """Return every set's method contrasts not already deciding, and the step arm's.

    Args:
        domain: The domain.

    Returns:
        (treatment, reference) pairs.
    """
    deciding = set(_deciding_contrasts(domain=domain))
    contrasts: list[tuple[str, str]] = []
    for blend in domain.sets:
        single = domain.single(blend.best_single)
        name = blend.name
        contrasts += [
            pair
            for pair in (
                (f"{name}_xgb", f"{name}_control"),
                (f"{name}_control", single),
                (f"{name}_xgb", single),
                (f"{name}_mean", single),
                (f"{name}_stack", single),
                (f"{name}_equal", single),
                (f"{name}_stack", f"{name}_xgb"),
                (f"{name}_stack", f"{single}_seed_stack"),
            )
            if pair not in deciding
        ]
    if domain.name == "wind":
        contrasts += [(STEP_ARM, "everything_xgb"), (STEP_ARM, domain.single("icon_d2"))]
    return contrasts


def _split_scopes(*, losses: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """Split losses by site, by meteorological season, and by UKV era.

    Args:
        losses: Per-row losses carrying `site`, `time` and `month`.

    Returns:
        (scope label, rows) pairs.
    """
    keyed = losses.with_columns(
        season=pl.col("time")
        .dt.month()
        .replace_strict(weather_products.SEASONS, return_dtype=pl.Utf8),
        era=pl.when(pl.col("month") >= weather_products.UPGRADE_MONTH)
        .then(pl.lit("post"))
        .otherwise(pl.lit("pre")),
    )
    scopes: list[tuple[str, pl.DataFrame]] = []
    for column, values in (
        ("site", sorted(keyed["site"].unique().to_list())),
        ("season", ["winter", "spring", "summer", "autumn"]),
        ("era", ["pre", "post"]),
    ):
        scopes += [(f"{column} {value}", keyed.filter(pl.col(column) == value)) for value in values]
    return scopes


def _mae(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean error, in percentage points of capacity.

    Args:
        losses: Per-row losses at one setting.
        arm: The arm.

    Returns:
        The mean, or NaN if the arm is absent.
    """
    rows = losses.filter(pl.col("arm") == arm)
    if rows.is_empty():
        return float("nan")
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def _error_tables(*, losses: pl.DataFrame, domain: Domain) -> list[str]:
    """Render every arm's absolute error, at both settings.

    Args:
        losses: The domain's losses.
        domain: The domain.

    Returns:
        Markdown lines.
    """
    by_setting = {s: losses.filter(pl.col("setting") == s) for s in SETTINGS}
    lines = [
        f"#### {domain.name.capitalize()}: single products",
        "",
        "| Product | Arm | MAE, primary setting | MAE, second setting |",
        "|---|---|---|---|",
    ]
    for product in domain.products:
        arm = domain.single(product)
        lines.append(
            f"| {product} | {arm} | {_mae(losses=by_setting['pooled'], arm=arm):.3f} "
            f"| {_mae(losses=by_setting['sensitivity'], arm=arm):.3f} |"
        )
    methods = ("xgb", "control", "mean", "stack", "equal")
    lines += [
        "",
        f"#### {domain.name.capitalize()}: blends, primary setting (second setting in brackets)",
        "",
        "| Set | Products | Consumer | Best single | "
        + " | ".join(f"`{m}`" for m in methods)
        + " |",
        "|---" * (4 + len(methods)) + "|",
    ]
    for blend in domain.sets:
        best = domain.single(blend.best_single)
        cells = []
        for method in methods:
            arm = f"{blend.name}_{method}"
            second = _mae(losses=by_setting["sensitivity"], arm=arm)
            cell = f"{_mae(losses=by_setting['pooled'], arm=arm):.3f}"
            cells.append(cell if np.isnan(second) else f"{cell} ({second:.3f})")
        lines.append(
            f"| {blend.name} | {', '.join(blend.products)} | {blend.consumer} "
            f"| {best} {_mae(losses=by_setting['pooled'], arm=best):.3f} | "
            + " | ".join(cells)
            + " |"
        )
    seed_stacks = sorted({blend.best_single for blend in domain.sets})
    lines += [
        "",
        "Seed-ensemble stack, primary setting: "
        + ", ".join(
            f"{domain.single(p)}_seed_stack "
            f"{_mae(losses=by_setting['pooled'], arm=f'{domain.single(p)}_seed_stack'):.3f}"
            for p in seed_stacks
        )
        + ".",
    ]
    if domain.name == "wind":
        lines += ["", f"MAE: {STEP_ARM} {_mae(losses=by_setting['pooled'], arm=STEP_ARM):.3f}."]
    return lines


def _feature_lines(*, domain: Domain) -> list[str]:
    """Render every fitted arm's feature columns, and which settings it was fitted at.

    Args:
        domain: The domain.

    Returns:
        Markdown lines.
    """
    sensitivity = set(_sensitivity_arms(domain=domain))
    lines = [
        f"#### {domain.name.capitalize()}: every fitted arm's feature columns",
        "",
        "| Arm | Settings | Columns | Feature columns |",
        "|---|---|---|---|",
    ]
    for arm, features in _arm_features(domain=domain).items():
        settings = "primary, second" if arm in sensitivity else "primary"
        lines.append(
            f"| {arm} | {settings} | {len(features)} | "
            + ", ".join(f"`{f}`" for f in features)
            + " |"
        )
    return lines


def _weight_lines(*, weights: pl.DataFrame) -> list[str]:
    """Summarise each stack's weight on each model across sites, seeds and folds.

    Args:
        weights: Every stack's weights.

    Returns:
        Markdown lines.
    """
    pooled = weights.filter(pl.col("setting") == "pooled")
    across_folds = (
        pooled.group_by("domain", "arm", "model", "site", "seed")
        .agg(fold_std=pl.col("weight").std())
        .group_by("domain", "arm", "model")
        .agg(pl.col("fold_std").mean())
    )
    summary = (
        pooled.group_by("domain", "arm", "model")
        .agg(
            mean=pl.col("weight").mean(),
            low=pl.col("weight").min(),
            high=pl.col("weight").max(),
        )
        .join(across_folds, on=["domain", "arm", "model"])
        .sort("domain", "arm", "model")
    )
    lines = [
        (
            "| Domain | Stack | Model | Mean weight | Lowest | Highest "
            "| Mean across-fold standard deviation |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['domain']} | {row['arm']} | {row['model']} | {row['mean']:.3f} "
        f"| {row['low']:.3f} | {row['high']:.3f} | {row['fold_std']:.3f} |"
        for row in summary.iter_rows(named=True)
    ]
    return lines


def _gap_lines(*, losses: pl.DataFrame, domain: Domain) -> list[str]:
    """Report each set's in-sample stack against its cross-fitted stack, point estimates.

    Args:
        losses: The domain's losses.
        domain: The domain.

    Returns:
        Markdown rows.
    """
    lines: list[str] = []
    for setting in SETTINGS:
        at_setting = losses.filter(pl.col("setting") == setting)
        for blend in domain.sets:
            cross = _mae(losses=at_setting, arm=f"{blend.name}_stack")
            in_sample = _mae(losses=at_setting, arm=f"{blend.name}_stack_in_sample")
            lines.append(
                f"| {domain.name} | {setting} | {blend.name} | {cross:.3f} | {in_sample:.3f} "
                f"| {in_sample - cross:+.4f} |"
            )
    return lines


def _power_version() -> int:
    """Return the power Delta table's current version, which every power read here sees.

    Returns:
        The version.
    """
    return DeltaTable(POWER_DELTA_URI).version()


def _solar_frame() -> pl.DataFrame:
    """Build the solar study's common rows exactly as `weather_products.main` builds them.

    Returns:
        The rows, with every blend column added.
    """
    frame = with_export_cap(
        dataset=weather_products._with_eras(
            frame=_add_time_features(
                dataset=weather_products._common_rows(frame=weather_products._joined())
            )
        )
    )
    return _with_blend_columns(frame=frame, domain=SOLAR)


def _wind_frame() -> pl.DataFrame:
    """Build the wind study's common rows exactly as `wind_products.main` builds them.

    Returns:
        The rows, with every blend column added.
    """
    frame = weather_products._with_eras(
        frame=_add_time_features(
            dataset=wind_products._common_rows(frame=wind_products._joined(sites=_wind_sites()))
        )
    )
    return _with_blend_columns(frame=frame, domain=WIND)


class _Outputs(TypedDict):
    """Everything one domain's run produced."""

    losses: pl.DataFrame
    weights: pl.DataFrame
    predictions: pl.DataFrame


def _run_domain(
    *, domain: Domain, frame: pl.DataFrame, singles: pl.DataFrame, resume: bool
) -> _Outputs:
    """Fit every remaining arm, derive the stacks, and assemble the domain's outputs.

    Args:
        domain: The domain.
        frame: The domain's common rows.
        singles: The single-product arms already fitted at the published settings.
        resume: Whether to reuse fits already on disk.

    Returns:
        The losses of every arm, the stacks' weights, and every arm's predictions.
    """
    features = _arm_features(domain=domain)
    single_arms = {domain.single(product) for product in domain.products}
    jobs = _jobs(
        domain=domain, arms=[a for a in features if a not in single_arms], setting="pooled"
    )
    jobs += _jobs(domain=domain, arms=_sensitivity_arms(domain=domain), setting="sensitivity")
    done = {(arm, setting) for arm, setting in singles.select("arm", "setting").unique().rows()}
    fitted = pl.concat(
        [
            singles,
            _fitted(
                frame=frame,
                domain=domain,
                jobs=[job for job in jobs if (job[0], job[1]) not in done],
                resume=resume,
            ),
        ]
    )
    derived: list[pl.DataFrame] = []
    weights: list[pl.DataFrame] = []
    for setting in SETTINGS:
        stack_losses, stack_weights = _stacks(losses=fitted, domain=domain, setting=setting)
        derived.append(stack_losses)
        weights.append(stack_weights)
    seed_losses, seed_weights = _seed_stacks(losses=fitted, domain=domain)
    losses = pl.concat(
        [fitted.select(LOSS_COLUMNS), *derived, seed_losses], how="vertical_relaxed"
    ).with_columns(domain=pl.lit(domain.name))
    predictions = losses.join(
        frame.select("site", "time", "power_mw"), on=["site", "time"], how="left"
    ).select(
        "domain",
        "site",
        "time",
        "month",
        "fold",
        "seed",
        "arm",
        "setting",
        pl.col("power_mw").cast(pl.Float64),
        prediction_mw=pl.col("power_mw").cast(pl.Float64) + pl.col("signed_error_capped_mw"),
    )
    return {
        "losses": losses,
        "weights": pl.concat([*weights, seed_weights]),
        "predictions": predictions,
    }


def _intervals(*, losses: pl.DataFrame, domain: Domain) -> list[IntervalRecord]:
    """Bootstrap every contrast the report prints for one domain.

    Args:
        losses: The domain's losses, every arm and setting.
        domain: The domain.

    Returns:
        Every interval, labelled by section.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    second = losses.filter(pl.col("setting") == "sensitivity")
    records: list[IntervalRecord] = []
    deciding = _deciding_contrasts(domain=domain)
    for setting, at_setting in (("pooled", pooled), ("sensitivity", second)):
        records += [
            _interval(
                losses=at_setting, contrast=pair, domain=domain, setting=setting, section="deciding"
            )
            for pair in deciding
        ]
    if domain.name == "solar":
        records.append(
            _interval(
                losses=pooled,
                contrast=POSITIVE_CONTROL,
                domain=domain,
                setting="pooled",
                section="positive_control",
            )
        )
    records += [
        _interval(
            losses=pooled,
            contrast=(f"{domain.single(product)}_seed_stack", domain.single(product)),
            domain=domain,
            setting="pooled",
            section="seed_stack_control",
        )
        for product in sorted({blend.best_single for blend in domain.sets})
    ]
    records += [
        _interval(
            losses=pooled, contrast=pair, domain=domain, setting="pooled", section="exploratory"
        )
        for pair in _exploratory_contrasts(domain=domain)
    ]
    named_pairs = [
        pair
        for name in domain.named_sets
        for pair in (
            (f"{name}_xgb", f"{name}_control"),
            (f"{name}_xgb", domain.single(domain.blend_set(name).best_single)),
        )
    ]
    for scope, rows in _split_scopes(losses=pooled):
        records += [
            _interval(
                losses=rows,
                contrast=pair,
                domain=domain,
                setting="pooled",
                section="exploratory_split",
                scope=scope,
            )
            for pair in named_pairs
        ]
    return records


def _report(
    *,
    frames: dict[DomainType, pl.DataFrame],
    outputs: dict[DomainType, _Outputs],
    reproduction: list[ReproductionRow],
    records: list[IntervalRecord],
    power_version: int,
) -> str:
    """Assemble the markdown report.

    Args:
        frames: Each domain's common rows.
        outputs: Each domain's losses, weights and predictions.
        reproduction: The reproduction check.
        records: Every interval.
        power_version: The power Delta table's version.

    Returns:
        The report.
    """
    lines = ["### Blending weather products, on the two weather-product studies' own rows", ""]
    lines += [
        f"- {name}: {frame.height:,} common site-hours, {frame['site'].n_unique()} generators, "
        f"{frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d}."
        for name, frame in frames.items()
    ]
    lines += [f"- Power Delta table version {power_version}.", ""]
    lines += [
        (
            "Mean absolute error as a percentage of each site's P99 output. A negative ΔMAE means "
            "the treatment's error is lower."
        ),
        "",
        "#### Reproduction check: refitted single-product arms against the published losses",
        "",
        *_reproduction_lines(rows=reproduction),
        "",
    ]
    for domain in DOMAINS:
        lines += [*_feature_lines(domain=domain), ""]
    for domain in DOMAINS:
        lines += [*_error_tables(losses=outputs[domain.name]["losses"], domain=domain), ""]
    lines += [
        "#### Deciding contrasts, named before the run",
        "",
        (
            "Per named set: blend against its control, control against the best single, and "
            "blend against the best single beside them; then the stack against XGBoost on every "
            "column."
        ),
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="deciding", setting="pooled"),
        "",
        "#### The deciding contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="deciding", setting="sensitivity"),
        "",
        "#### Positive control: CAMS with ICON-D2 against ICON-D2 alone",
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="positive_control"),
        "",
        "#### Seed-ensemble stack control: each best single's three seeds, stacked",
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="seed_stack_control"),
        "",
        "#### Stack weights, primary setting, across sites, seeds and folds",
        "",
        *_weight_lines(
            weights=pl.concat([outputs[d.name]["weights"] for d in DOMAINS]).filter(
                ~pl.col("arm").str.ends_with("_in_sample")
            )
        ),
        "",
        "#### In-sample against cross-fitted stack weights (point estimates)",
        "",
        "| Domain | Setting | Set | Cross-fitted MAE | In-sample MAE | In-sample − cross-fitted |",
        "|---|---|---|---|---|---|",
    ]
    for domain in DOMAINS:
        lines += _gap_lines(losses=outputs[domain.name]["losses"], domain=domain)
    exploratory = [r for r in records if r["section"].startswith("exploratory")]
    excluding = sum(r["excludes_zero"] for r in exploratory)
    lines += [
        "",
        "#### Exploratory contrasts",
        "",
        (
            f"{len(exploratory)} exploratory intervals, of which {excluding} exclude zero. About 1 "
            f"in 20, so about {len(exploratory) / 20:.0f}, would exclude zero by chance alone."
        ),
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="exploratory"),
        "",
        "#### The named blends by site, season and era (exploratory)",
        "",
        (
            "The post era holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="exploratory_split"),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Refit the single products, check them, fit and stack every blend, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse per-arm fits a previous run left on disk."
    )
    resume = parser.parse_args().resume

    power_version = _power_version()
    frames: dict[DomainType, pl.DataFrame] = {"solar": _solar_frame(), "wind": _wind_frame()}
    for name, frame in frames.items():
        _LOG.info("%s common rows: %d", name, frame.height)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    singles: dict[DomainType, pl.DataFrame] = {}
    reproduction: list[ReproductionRow] = []
    for domain in DOMAINS:
        jobs = [
            job
            for setting in domain.published_settings
            for job in _jobs(
                domain=domain,
                arms=[domain.single(product) for product in domain.products],
                setting=setting,
            )
        ]
        singles[domain.name] = _fitted(
            frame=frames[domain.name], domain=domain, jobs=jobs, resume=resume
        )
        reproduction += _reproduction(fitted=singles[domain.name], domain=domain)
    gate = "\n".join(_reproduction_lines(rows=reproduction)) + "\n"
    (OUTPUT_DIR / "reproduction.md").write_text(gate)
    sys.stdout.write(gate)
    if not all(row["bit_identical"] for row in reproduction):
        _LOG.error("the single-product arms do not reproduce the published losses; stopping")
        return 1

    outputs = {
        domain.name: _run_domain(
            domain=domain,
            frame=frames[domain.name],
            singles=singles[domain.name],
            resume=resume,
        )
        for domain in DOMAINS
    }
    for key in ("losses", "weights", "predictions"):
        pl.concat([outputs[d.name][key] for d in DOMAINS], how="vertical_relaxed").write_parquet(
            OUTPUT_DIR / f"{'stack_weights' if key == 'weights' else key}.parquet"
        )
    records = [
        record
        for domain in DOMAINS
        for record in _intervals(losses=outputs[domain.name]["losses"], domain=domain)
    ]
    pl.DataFrame(records).write_parquet(OUTPUT_DIR / "intervals.parquet")
    report = _report(
        frames=frames,
        outputs=outputs,
        reproduction=reproduction,
        records=records,
        power_version=power_version,
    )
    (OUTPUT_DIR / "report.md").write_text(report)
    shutil.rmtree(OUTPUT_DIR / FITS_DIR_NAME)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
