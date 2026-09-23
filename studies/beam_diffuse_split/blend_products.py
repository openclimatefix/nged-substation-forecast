"""Measure whether blending weather products beats the best single product, for solar and wind.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/836>. It builds on the two
weather-product studies, `weather_products.py` (solar) and `wind_products.py` (wind), and refits
their single-product arms on their own common rows, folds and seeds, so every contrast here is
paired with theirs. It imports their column-building functions and never writes to their output
directories.

**Every published single-product arm must reproduce the published losses row for row before any
blend runs.** Each refitted arm is compared against its study's `losses.parquet`: the same (site,
time, fold, seed) keys and a bit-identical `signed_error_capped_mw`. If any differs, the rows or the
folds have changed, and the script stops.

**Each set of products is blended twice: from each product's plain columns, and from its enriched
columns.** The enriched single-product arm, `<product>_rich`, adds what the published solar page
found helps one product on its own: the neighbouring hours for every product, CAMS's own beam split,
and UKV's hour rebuilt from the snapshots at both ends. For wind it adds each product's hub-height
speed at ±1 h and ±2 h and its 10 m speed at ±1 h. The enriched contrasts are the deciding ones.
They were added after the first run, when the first science review found that much of each plain
blend's gain was available from one product with those additions.

**The enriched best single of a set is the best single-product arm measured**, chosen by mean error
on the common rows at each hyperparameter setting, among every published single-product arm and
every enriched arm of the set's products. The plain contrasts keep the best single fixed in advance
from the published tables, as the plan named them.

**Each set is blended four ways**, from its plain or its enriched columns:

- `<set>_xgb`, `<set>_rich_xgb`: XGBoost shown every product's columns, column subsampling at 1.
- `<set>_mean`, `<set>_rich_mean`: XGBoost shown the mean of the products' values.
- `<set>_stack`, `<set>_rich_stack`: a linear stack of the single-product models' out-of-fold
  predictions, with non-negative weights summing to 1, cross-fitted per generator, seed and fold by
  `studies.blending.stacked_errors`.
- `<set>_equal`, `<set>_rich_equal`: the equal-weight mean of the single-product predictions.

**Every XGBoost blend has a climatology control**, `<set>_control` or `<set>_rich_control`. The
control holds the best single's real columns and every other product's columns permuted among the
rows sharing a site, a month and an hour of day, one permutation per product.

**A synthetic product measures how small a gain the pipeline can detect.** `synthetic_xgb` is the
best single of the `everything` set shown one more column: the generator's own output as a fraction
of capacity, plus Gaussian noise large enough that the column carries only part of the target.
`synthetic_control` is shown the same column permuted within site, month and hour of day.

Run it with `uv run python studies/beam_diffuse_split/blend_products.py`, after both weather-product
studies have been run. `--resume` reuses the per-arm fits a previous run left in `fits/`.
`--report-only` rebuilds `report.md` from `losses.parquet`, `stack_weights.parquet` and
`intervals.parquet` already on disk, fitting nothing; move the current `report.md` to a
`superseded/` subfolder first, since this overwrites it. `--report-only` needs `--power-version`,
the power Delta table version the fits on disk read, which the replaced report prints: the table may
have gained versions since, so its current version is not the one the results rest on.
"""

import argparse
import logging
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict, cast

import numpy as np
import polars as pl
import weather_products
import wind_products
from build_dataset import POWER_DELTA_URI, _wind_sites
from deltalake import DeltaTable
from export_cap import with_export_cap
from run_experiment import SHARED_FEATURES as SOLAR_SHARED_FEATURES
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR
from studies.blending import PERMUTED_SUFFIX, climatology_permutation, stacked_errors
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    fold_t_interval,
    per_fold_differences,
)
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

RICH_PERMUTATION_SEED: Final[int] = 20260924
"""The same, for each product's enriched columns, which are permuted as one group per product."""

RICH_PERMUTED_SUFFIX: Final[str] = "_rich_shuffled"
"""Names an enriched column's permuted copy, apart from the plain column's `_shuffled` copy."""

PERMUTATION_GROUPS: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""The rows a control column's value may move between: one site, one month, one hour of day."""

SYNTHETIC_COLUMN: Final[str] = "synthetic_product"
"""The synthetic product: the generator's output as a fraction of capacity, plus noise."""

SYNTHETIC_SEED: Final[int] = 20260925
"""Seeds the synthetic product's noise, drawn once per row in the common rows' order."""

SYNTHETIC_PERMUTATION_SEED: Final[int] = 20260926
"""Seeds the synthetic product's climatology permutation."""

SYNTHETIC_ARMS: Final[tuple[str, str]] = ("synthetic_xgb", "synthetic_control")
"""The sensitivity positive control's arm and its climatology control."""

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

VariantType = Literal["plain", "rich"]
"""Whether a blend is built from each product's plain columns or from its enriched columns."""

VARIANTS: Final[tuple[VariantType, ...]] = ("plain", "rich")

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

LATENCY_HOURS: Final[dict[str, float]] = {
    "cams": 24.0,
    "era5": 120.0,
    "ukv": 4.0,
    "icon_d2": 1.5,
    "icon_eu": 3.5,
    "icon_global": 3.5,
}
"""How long after an hour each product's value for it is available, from the solar page's table."""

ENRICHED_LATENCY_HOURS: Final[dict[DomainType, float]] = {"solar": 1.0, "wind": 2.0}
"""How much later than its slowest product an enriched blend needs its neighbouring hours.

The recommended blends are enriched, and each reads the hour after the scored hour as well: the
irradiance context's furthest forward offset for solar (`with_irradiance_context`), and the wind
context's furthest forward offset for wind (`with_wind_context`, `HUB_OFFSETS_HOURS`)."""

HISTORY_ONLY_PRODUCTS: Final[frozenset[str]] = frozenset({"cams", "era5"})
"""Products too late for a live service: CAMS arrives about a day late and ERA5 about 5 days."""


def live_great_britain_wide(*, products: tuple[str, ...]) -> bool:
    """Return whether a live service anywhere in Great Britain could read every one of `products`.

    A product is out if it arrives too late for a live service (`HISTORY_ONLY_PRODUCTS`), or if
    it does not cover the whole of Great Britain, as ICON-D2 does not.

    Args:
        products: The products a single-product arm or a blend reads.

    Returns:
        Whether every product is both live and Great-Britain-wide.
    """
    return not (HISTORY_ONLY_PRODUCTS & set(products)) and "icon_d2" not in products


USABLE_FROM: Final[dict[DomainType, dict[str, str]]] = {
    "solar": {
        "cams": "2004-01",
        "era5": "1940-01",
        "ukv": "2022-03",
        "icon_d2": "2022-12",
        "icon_eu": "2022-11",
        "icon_global": "2022-11",
    },
    "wind": {
        "era5": "1940-01",
        "ukv": "2024-08",
        "icon_d2": "2022-12",
        "icon_eu": "2022-11",
        "icon_global": "2022-11",
    },
}
"""The first month each product's archive serves, from the two published pages.

UKV's solar archive before August 2024 is a backfill from a source Open-Meteo does not name, and its
hub-height wind starts on 12 August 2024.
"""

OUTPUT_BANDS: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
"""The lower edges of the measured-output bands, as fractions of capacity, for the wind table."""

HOURS_PER_DAY: Final[float] = 24.0

MOST_IMPROVED_SHARE: Final[float] = 0.05
"""The share of rows, the most improved first, whose part of the total gain the report prints."""


@dataclass(frozen=True)
class BlendSet:
    """A set of products blended together, and the plain single product it has to beat."""

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
    rich_columns: Callable[[str], tuple[str, ...]]
    rich_mean_width: int
    single_suffix: str
    named_sets: tuple[str, ...]
    published_losses: Path
    published_jobs: tuple[Job, ...]
    rich_published: dict[str, str]
    synthetic_noise: float

    def single(self, product: str) -> str:
        """Return a product's plain single-product arm, named as the published study names it.

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

    def product_of(self, arm: str) -> str:
        """Return the product a single-product arm reads.

        Args:
            arm: A published single-product arm or an enriched arm.

        Returns:
            The longest product name the arm starts with.
        """
        return max((p for p in self.products if arm.startswith(f"{p}_")), key=len)


def rich(product: str) -> str:
    """Return a product's enriched single-product arm.

    Args:
        product: The product.

    Returns:
        The arm name.
    """
    return f"{product}_rich"


def _solar_columns(product: str) -> tuple[str]:
    """Return a solar product's one plain feature column: its global horizontal irradiance.

    Args:
        product: A key of `weather_products.PRODUCTS`.

    Returns:
        The column name.
    """
    return (f"ghi_{product}",)


def _solar_rich_columns(product: str) -> tuple[str, ...]:
    """Return a solar product's enriched columns: the hour before, the hour, the hour after, more.

    The first three columns are always the hour before, the hour and the hour after, which the
    enriched mean arm averages across products. UKV's three are its hour rebuilt from the snapshots
    at both ends, as the published `ukv_trap_ctx_global` arm reads them; ICON-EU's are the published
    `icon_eu_ctx_global` arm's. CAMS adds its own beam split, which the published page found helps.

    Args:
        product: A key of `weather_products.PRODUCTS`.

    Returns:
        The column names.
    """
    if product == "ukv":
        return ("ghi_trap_previous_ukv", "ghi_trap_ukv", "ghi_trap_next_ukv")
    context = (f"ghi_previous_{product}", f"ghi_{product}", f"ghi_next_{product}")
    return (*context, "bhi_cams", "dhi_cams") if product == "cams" else context


def _wind_columns(product: str) -> tuple[str, str, str, str]:
    """Return a wind product's four plain feature columns, as the wind study names them.

    Args:
        product: A key of `fetch_wind_point.PRODUCTS`.

    Returns:
        The hub-height speed, that height's direction as sine and cosine, and the 10 m speed.
    """
    return wind_products._wind_columns(product=product)


def _wind_rich_columns(product: str) -> tuple[str, ...]:
    """Return a wind product's enriched columns: its plain four, then its neighbouring hours.

    Args:
        product: A key of `fetch_wind_point.PRODUCTS`.

    Returns:
        The column names.
    """
    return (*_wind_columns(product), *wind_products.context_columns(product=product))


def _solar_published_jobs() -> tuple[Job, ...]:
    """Return the published solar study's pooled arms, less the two the enriched arms reproduce.

    Returns:
        The jobs, as `weather_products.jobs` builds them.
    """
    return tuple(
        job
        for job in weather_products.jobs()
        if job[0] not in ("icon_eu_ctx_global", "ukv_trap_ctx_global")
    )


def _wind_published_jobs() -> tuple[Job, ...]:
    """Return the published wind study's arms on the common rows, less the step-period arms.

    The step indicator works as a date-regime feature, so an arm carrying it is not a fair single.
    The arms at each of `wind_products.ROW_SET_SETTINGS` are fitted on other row sets, so they are
    left out too.

    Returns:
        The jobs, as `wind_products.jobs` builds them.
    """
    return tuple(
        job
        for job in wind_products.jobs()
        if not job[0].endswith("_step") and job[1] not in wind_products.ROW_SET_SETTINGS
    )


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
    rich_columns=_solar_rich_columns,
    rich_mean_width=3,
    single_suffix="_global",
    named_sets=("everything", "cams_icon_eu", "live_all"),
    published_losses=STUDY_DATA_DIR / weather_products.OUTPUT_DIR_NAME / "losses.parquet",
    published_jobs=_solar_published_jobs(),
    rich_published={"icon_eu_rich": "icon_eu_ctx_global", "ukv_rich": "ukv_trap_ctx_global"},
    synthetic_noise=0.3,
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
    rich_columns=_wind_rich_columns,
    rich_mean_width=10,
    single_suffix="_wind",
    named_sets=("everything", "live_gb"),
    published_losses=STUDY_DATA_DIR / wind_products.OUTPUT_DIR_NAME / "losses.parquet",
    published_jobs=_wind_published_jobs(),
    rich_published={},
    synthetic_noise=0.45,
)

DOMAINS: Final[tuple[Domain, ...]] = (SOLAR, WIND)

POSITIVE_CONTROL: Final[tuple[str, str]] = ("cams_icon_d2_xgb", "icon_d2_global")
"""Solar, where a large gain must appear: ICON-D2 with CAMS against ICON-D2 alone."""

STEP_ARM: Final[str] = "everything_xgb_step"
"""The one exploratory wind arm shown ICON global's step indicator, `step_period`."""

BestType = dict[tuple[str, SettingType], str]
"""Each (set, setting)'s enriched best single: the lowest-error single-product arm measured."""


def _prefix(variant: VariantType) -> str:
    """Return the infix an arm name carries for a variant.

    Args:
        variant: Plain or enriched.

    Returns:
        `""` or `"rich_"`.
    """
    return "rich_" if variant == "rich" else ""


def _arm(blend: str, variant: VariantType, method: str) -> str:
    """Return a blend arm's name, such as `everything_rich_xgb`.

    Args:
        blend: The set's name.
        variant: Plain or enriched.
        method: `xgb`, `control`, `mean`, `stack`, `stack_in_sample`, or `equal`.

    Returns:
        The arm name.
    """
    return f"{blend}_{_prefix(variant)}{method}"


def _mean_columns(*, domain: Domain, blend: BlendSet, variant: VariantType) -> tuple[str, ...]:
    """Return a set's mean-of-inputs columns.

    Args:
        domain: The domain.
        blend: The set.
        variant: Plain or enriched.

    Returns:
        One column per position averaged, named `<stem>_mean_<set>` for plain blends and
        `rich_mean_<position>_<set>` for enriched ones.
    """
    if variant == "rich":
        return tuple(
            f"rich_mean_{position}_{blend.name}" for position in range(domain.rich_mean_width)
        )
    return tuple(
        f"{column.removesuffix(f'_{blend.best_single}')}_mean_{blend.name}"
        for column in domain.columns(blend.best_single)
    )


def _product_columns(*, domain: Domain, variant: VariantType) -> Callable[[str], tuple[str, ...]]:
    """Return the function naming a product's columns in one variant.

    Args:
        domain: The domain.
        variant: Plain or enriched.

    Returns:
        `domain.columns` or `domain.rich_columns`.
    """
    return domain.rich_columns if variant == "rich" else domain.columns


def _with_synthetic(*, frame: pl.DataFrame, domain: Domain) -> pl.DataFrame:
    """Add the synthetic product: each row's output over capacity, plus Gaussian noise.

    Args:
        frame: The domain's common rows, in their fixed order.
        domain: The domain, which sets the noise's standard deviation.

    Returns:
        `frame` with `SYNTHETIC_COLUMN`.
    """
    noise = np.random.default_rng(SYNTHETIC_SEED).normal(
        scale=domain.synthetic_noise, size=frame.height
    )
    fraction = pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw")
    return frame.with_columns((fraction + pl.Series(noise)).alias(SYNTHETIC_COLUMN))


def _with_blend_columns(*, frame: pl.DataFrame, domain: Domain) -> pl.DataFrame:
    """Add every set's mean columns, the synthetic product, and every climatology-permuted column.

    The mean of the wind products' direction sines and of their cosines together give the circular
    mean direction. No step reorders the rows, which the reproduction check relies on.

    Args:
        frame: The domain's common rows, with the enriched columns.
        domain: The domain.

    Returns:
        `frame` with the mean, the synthetic, and the permuted columns added.
    """
    means = [
        pl.mean_horizontal(
            *(domain.columns(product)[position] for product in blend.products)
        ).alias(name)
        for blend in domain.sets
        for position, name in enumerate(_mean_columns(domain=domain, blend=blend, variant="plain"))
    ]
    means += [
        pl.mean_horizontal(
            *(domain.rich_columns(product)[position] for product in blend.products)
        ).alias(name)
        for blend in domain.sets
        for position, name in enumerate(_mean_columns(domain=domain, blend=blend, variant="rich"))
    ]
    plain = climatology_permutation(
        frame=_with_synthetic(frame=frame.with_columns(means), domain=domain),
        column_groups=[domain.columns(product) for product in domain.products],
        by=PERMUTATION_GROUPS,
        seed=PERMUTATION_SEED,
    )
    enriched = climatology_permutation(
        frame=plain,
        column_groups=[domain.rich_columns(product) for product in domain.products],
        by=PERMUTATION_GROUPS,
        seed=RICH_PERMUTATION_SEED,
        suffix=RICH_PERMUTED_SUFFIX,
    )
    return climatology_permutation(
        frame=enriched,
        column_groups=[(SYNTHETIC_COLUMN,)],
        by=PERMUTATION_GROUPS,
        seed=SYNTHETIC_PERMUTATION_SEED,
    )


def _single_features(*, domain: Domain) -> dict[str, tuple[str, ...]]:
    """Return every single-product arm's feature columns: the published arms and the enriched ones.

    Args:
        domain: The domain.

    Returns:
        Arm name to feature columns.
    """
    arms = {job[0]: job[3] for job in domain.published_jobs}
    arms |= {rich(p): (*domain.shared_features, *domain.rich_columns(p)) for p in domain.products}
    return arms


def _blend_features(*, domain: Domain, best: BestType) -> dict[str, tuple[str, ...]]:
    """Return every fitted blend arm's feature columns, in the order the model sees them.

    Args:
        domain: The domain.
        best: Each set's enriched best single, which the enriched control keeps real.

    Returns:
        Arm name to feature columns.
    """
    shared = domain.shared_features
    singles = _single_features(domain=domain)
    arms: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        columns = _product_columns(domain=domain, variant=variant)
        suffix = RICH_PERMUTED_SUFFIX if variant == "rich" else PERMUTED_SUFFIX
        for blend in domain.sets:
            if variant == "rich":
                reference = best[blend.name, "pooled"]
                real = singles[reference][len(shared) :]
                kept = domain.product_of(reference)
            else:
                real, kept = domain.columns(blend.best_single), blend.best_single
            permuted = tuple(
                f"{column}{suffix}" for p in blend.products if p != kept for column in columns(p)
            )
            every = tuple(column for p in blend.products for column in columns(p))
            xgb_arm = _arm(blend.name, variant, "xgb")
            control_arm = _arm(blend.name, variant, "control")
            arms[xgb_arm] = (*shared, *every)
            arms[control_arm] = (*shared, *real, *permuted)
            if variant == "rich" and len(arms[control_arm]) != len(arms[xgb_arm]):
                msg = (
                    f"{control_arm} has {len(arms[control_arm])} columns, {xgb_arm} has "
                    f"{len(arms[xgb_arm])}: the control must keep every column the blend has"
                )
                raise ValueError(msg)
            arms[_arm(blend.name, variant, "mean")] = (
                *shared,
                *_mean_columns(domain=domain, blend=blend, variant=variant),
            )
    reference = singles[best["everything", "pooled"]]
    synthetic, synthetic_control = SYNTHETIC_ARMS
    arms[synthetic] = (*reference, SYNTHETIC_COLUMN)
    arms[synthetic_control] = (*reference, f"{SYNTHETIC_COLUMN}{PERMUTED_SUFFIX}")
    if domain.name == "wind":
        arms[STEP_ARM] = (*arms["everything_xgb"], "step_period")
    return arms


def _single_jobs(*, domain: Domain) -> list[Job]:
    """Return the fits every single-product arm needs, at the settings it is used at.

    Every published arm is refitted at the settings its study used, so each can be checked. Every
    enriched arm and every plain `<product><suffix>` arm is fitted at both settings, so that the
    stacks and the best single can be formed at both.

    Args:
        domain: The domain.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    jobs = list(domain.published_jobs)
    published = {(job[0], job[1]) for job in jobs}
    features = _single_features(domain=domain)
    for product in domain.products:
        for arm in (domain.single(product), rich(product)):
            jobs += [
                (arm, setting, "power_mw", features[arm], SETTINGS[setting], False)
                for setting in SETTINGS
                if (arm, setting) not in published
            ]
    return jobs


def _blend_jobs(*, domain: Domain, best: BestType) -> list[Job]:
    """Return every blend fit: each arm at the primary setting, the named ones at the second too.

    Args:
        domain: The domain.
        best: Each set's enriched best single.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    features = _blend_features(domain=domain, best=best)
    jobs: list[Job] = [
        (arm, "pooled", "power_mw", columns, SETTINGS["pooled"], False)
        for arm, columns in features.items()
    ]
    second = [
        _arm(name, variant, method)
        for name in domain.named_sets
        for variant in VARIANTS
        for method in ("xgb", "control")
    ]
    jobs += [
        (arm, "sensitivity", "power_mw", features[arm], SETTINGS["sensitivity"], False)
        for arm in second
    ]
    return jobs


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
        Every job's losses in `LOSS_COLUMNS`, one row per (site, time, seed, arm, setting).
    """
    missing = [
        job
        for job in jobs
        if not (resume and _fit_path(domain=domain, arm=job[0], setting=job[1]).exists())
    ]
    _LOG.info("%s: %d of %d fits to run", domain.name, len(missing), len(jobs))
    if missing:
        fresh = run_all(dataset=frame, jobs=missing)
        for arm, setting, *_ in missing:
            path = _fit_path(domain=domain, arm=arm, setting=setting)
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh.filter(pl.col("arm") == arm, pl.col("setting") == setting).sort(
                "site", "time", "seed"
            ).write_parquet(path)
    return pl.concat(
        pl.read_parquet(_fit_path(domain=domain, arm=arm, setting=setting)).select(LOSS_COLUMNS)
        for arm, setting, *_ in jobs
    )


class ReproductionRow(TypedDict):
    """One refitted single-product arm's comparison against the published losses."""

    domain: str
    setting: str
    arm: str
    published_arm: str
    rows: int
    published_rows: int
    keys_equal: bool
    bit_identical: bool
    max_abs_difference_mw: float


def _reproduction_pairs(*, domain: Domain) -> list[tuple[str, str, SettingType]]:
    """Return every (refitted arm, published arm, setting) the reproduction check compares.

    Args:
        domain: The domain.

    Returns:
        The published jobs under their own names, and each enriched arm that repeats a published
        arm's columns under that arm's name.
    """
    pairs: list[tuple[str, str, SettingType]] = [
        (job[0], job[0], "sensitivity" if job[1] == "sensitivity" else "pooled")
        for job in domain.published_jobs
    ]
    pairs += [(ours, theirs, "pooled") for ours, theirs in domain.rich_published.items()]
    return pairs


def _reproduction(*, fitted: pl.DataFrame, domain: Domain) -> list[ReproductionRow]:
    """Compare every refitted published arm with the published study's losses.

    Args:
        fitted: The refitted single-product arms' losses.
        domain: The domain.

    Returns:
        One row per arm and published setting.
    """
    published = pl.read_parquet(domain.published_losses)
    keys = ["site", "time", "fold", "seed"]
    rows: list[ReproductionRow] = []
    for ours_arm, theirs_arm, setting in _reproduction_pairs(domain=domain):
        ours, theirs = (
            losses.filter(pl.col("arm") == arm, pl.col("setting") == setting)
            .sort(keys)
            .select(*keys, "signed_error_capped_mw")
            for losses, arm in ((fitted, ours_arm), (published, theirs_arm))
        )
        keys_equal = ours.height == theirs.height and ours.select(keys).equals(theirs.select(keys))
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
                "arm": ours_arm,
                "published_arm": theirs_arm,
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
            "| Domain | Setting | Refitted arm | Published arm | Rows | Published rows "
            "| Keys equal | Bit-identical `signed_error_capped_mw` | Largest difference (MW) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['domain']} | {row['setting']} | {row['arm']} | {row['published_arm']} "
        f"| {row['rows']:,} | {row['published_rows']:,} "
        f"| {'yes' if row['keys_equal'] else '**no**'} "
        f"| {'yes' if row['bit_identical'] else '**no**'} | {row['max_abs_difference_mw']:.3g} |"
        for row in rows
    ]
    return lines


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


LeaderboardKind = Literal["single", "blend"]
"""Whether a leaderboard row is one product read alone, or a set of products blended."""


class LeaderboardRow(TypedDict):
    """One leaderboard row's arm and its bootstrapped absolute level."""

    arm: str
    kind: LeaderboardKind
    name: str
    variant: VariantType
    live_gb: bool
    mae_pp: float
    lower_95_pp: float
    upper_95_pp: float
    n_months: int


LEADERBOARD_HEADER: Final[tuple[str, str]] = (
    (
        "| Arm | Kind | Set or product | Variant | Live, Great-Britain-wide? "
        "| MAE (pp of capacity) | 95% interval, months and seed |"
    ),
    "|---|---|---|---|---|---|---|",
)


class LeaderboardArm(NamedTuple):
    """One arm Figure 1's leaderboard ranks, and what the arm reads."""

    arm: str
    kind: LeaderboardKind
    name: str
    variant: VariantType
    products: tuple[str, ...]


def _leaderboard_arms(*, domain: Domain) -> list[LeaderboardArm]:
    """Return every arm Figure 1's leaderboard ranks: each single product and each blend, twice.

    Args:
        domain: The domain.

    Returns:
        For each row: the arm name, whether it is a single product or a blend, the product or set
        name, plain or enriched, and the products it reads.
    """
    arms: list[LeaderboardArm] = []
    for product in domain.products:
        arms.append(LeaderboardArm(domain.single(product), "single", product, "plain", (product,)))
        arms.append(LeaderboardArm(rich(product), "single", product, "rich", (product,)))
    for blend in domain.sets:
        name, products = blend.name, blend.products
        arms.append(LeaderboardArm(_arm(name, "plain", "xgb"), "blend", name, "plain", products))
        arms.append(LeaderboardArm(_arm(name, "rich", "xgb"), "blend", name, "rich", products))
    return arms


def _leaderboard_rows(*, losses: pl.DataFrame, domain: Domain) -> list[LeaderboardRow]:
    """Bootstrap every leaderboard arm's absolute level, at the primary setting.

    Args:
        losses: The domain's losses, every arm and setting.
        domain: The domain.

    Returns:
        One row per arm in `_leaderboard_arms`, in that order.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    rows: list[LeaderboardRow] = []
    for arm, kind, name, variant, products in _leaderboard_arms(domain=domain):
        interval = bootstrap_absolute(losses=pooled, arm=arm, metric=METRIC)
        rows.append(
            {
                "arm": arm,
                "kind": kind,
                "name": name,
                "variant": variant,
                "live_gb": live_great_britain_wide(products=products),
                "mae_pp": interval["value"] * PERCENTAGE_POINTS,
                "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
                "n_months": interval["n_months"],
            }
        )
    return rows


def _leaderboard_lines(*, rows: list[LeaderboardRow], domain: Domain) -> list[str]:
    """Render one domain's leaderboard table.

    Args:
        rows: The output of `_leaderboard_rows`.
        domain: The domain.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            f"#### {domain.name.capitalize()}: leaderboard, every single product and blend, plain "
            "and enriched (main XGBoost settings)"
        ),
        "",
        (
            "Each arm's own absolute mean error, with its own 95% interval from resampling whole "
            "months and a seed. The interval is wide mainly because every arm's error rises and "
            "falls together from month to month, and resampling whole months carries that shared "
            "swing into each arm's own interval. A paired contrast below resamples the same months "
            "for both arms, which cancels the shared swing, so its interval is much narrower. "
            "Live, Great-Britain-wide: every product the arm reads is available within hours and "
            "covers all of Great Britain, as the latency table below sets out."
        ),
        "",
        *LEADERBOARD_HEADER,
    ]
    lines += [
        f"| {row['arm']} | {row['kind']} | {row['name']} | {row['variant']} "
        f"| {'yes' if row['live_gb'] else 'no'} | {row['mae_pp']:.3f} "
        f"| [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] |"
        for row in rows
    ]
    return lines


def _best_singles(*, losses: pl.DataFrame, domain: Domain) -> BestType:
    """Choose each set's enriched best single at each setting, by mean error on the common rows.

    The candidates are every single-product arm of the set's products fitted at that setting: at the
    primary setting every published arm and every enriched arm, and at the second setting the plain
    and the enriched arms.

    Args:
        losses: The single-product arms' losses, at both settings.
        domain: The domain.

    Returns:
        The lowest-error arm per (set, setting).
    """
    mae = {
        (arm, setting): value * PERCENTAGE_POINTS
        for arm, setting, value in losses.group_by("arm", "setting")
        .agg(pl.col(METRIC).mean())
        .rows()
    }
    best: BestType = {}
    for blend in domain.sets:
        for setting in SETTINGS:
            candidates = [
                arm for arm, at in mae if at == setting and domain.product_of(arm) in blend.products
            ]
            best[blend.name, setting] = min(candidates, key=lambda arm: mae[arm, setting])
    return best


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


def _models(*, domain: Domain, blend: BlendSet, variant: VariantType) -> list[str]:
    """Return the single-product arms a set's stack and equal-weight mean combine.

    Args:
        domain: The domain.
        blend: The set.
        variant: Plain or enriched.

    Returns:
        One arm per product.
    """
    return [rich(p) if variant == "rich" else domain.single(p) for p in blend.products]


def _stacks(
    *, losses: pl.DataFrame, domain: Domain, setting: SettingType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack, in-sample stack and equal-weight-average every set's single-product models.

    Args:
        losses: The fitted losses, carrying every plain and enriched single at `setting`.
        domain: The domain.
        setting: The setting.

    Returns:
        The derived arms' losses, and every stack's weights.
    """
    singles = [arm for p in domain.products for arm in (domain.single(p), rich(p))]
    wide = _wide_errors(losses=losses, arms=singles, setting=setting)
    derived: list[pl.DataFrame] = []
    weights: list[pl.DataFrame] = []
    for variant in VARIANTS:
        for blend in domain.sets:
            models = _models(domain=domain, blend=blend, variant=variant)
            errors = wide.select(models).to_numpy()
            for method, cross_fitted in (("stack", True), ("stack_in_sample", False)):
                result = stacked_errors(
                    errors=errors,
                    sites=wide["site"].to_numpy(),
                    folds=wide["fold"].to_numpy(),
                    seeds=wide["seed"].to_numpy(),
                    fit_rows=~wide["constrained"].to_numpy(),
                    cross_fitted=cross_fitted,
                )
                arm = _arm(blend.name, variant, method)
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
                    keys=wide,
                    errors=errors.mean(axis=1),
                    arm=_arm(blend.name, variant, "equal"),
                    setting=setting,
                )
            )
    return pl.concat(derived), pl.concat(weights)


def _seed_stack_arms(*, domain: Domain, best: BestType) -> list[str]:
    """Return the single-product arms whose seeds are stacked: every plain and enriched best single.

    Args:
        domain: The domain.
        best: Each set's enriched best single.

    Returns:
        The arms, sorted.
    """
    plain = {domain.single(blend.best_single) for blend in domain.sets}
    enriched = {arm for (_, setting), arm in best.items() if setting == "pooled"}
    return sorted(plain | enriched)


def _seed_stacks(
    *, losses: pl.DataFrame, domain: Domain, arms: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack each named single's three seeds, which measures what ensembling alone gives.

    The stacked error is one value per row, repeated under each seed so the paired bootstrap can
    draw a seed for the reference arm as it always does.

    Args:
        losses: The fitted losses at the primary setting.
        domain: The domain.
        arms: The single-product arms to stack the seeds of.

    Returns:
        The `<single>_seed_stack` arms' losses, and their weights.
    """
    derived: list[pl.DataFrame] = []
    weights: list[pl.DataFrame] = []
    for single in arms:
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
    fold_lower_95_pp: float
    fold_upper_95_pp: float
    site_lowest_pp: float
    site_highest_pp: float
    seed_spread_pp: float
    excludes_zero: bool
    folds_agreeing: int
    n_folds: int
    n_rows: int
    n_months: int


def _site_range(*, pair: pl.DataFrame, treatment: str, reference: str) -> tuple[float, float]:
    """Return the lowest and highest per-generator difference, averaged over rows and seeds.

    Args:
        pair: Losses holding both arms.
        treatment: The treatment arm.
        reference: The reference arm.

    Returns:
        The lowest and the highest generator's difference, as fractions of capacity.
    """
    by_site = (
        pair.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", treatment=pl.col(METRIC))
        .join(
            pair.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", reference=pl.col(METRIC)
            ),
            on=["site", "time", "seed"],
        )
        .group_by("site")
        .agg(difference=(pl.col("treatment") - pl.col("reference")).mean())
    )
    differences = by_site["difference"].to_numpy()
    return float(differences.min()), float(differences.max())


def _interval(
    *,
    losses: pl.DataFrame,
    contrast: tuple[str, str],
    domain: Domain,
    setting: str,
    section: str,
    scope: str = "all",
) -> IntervalRecord:
    """Bootstrap one paired contrast, count the folds agreeing, and add the fold and site spreads.

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
    fold_lower, fold_upper = (
        fold_t_interval(fold_differences=folds) if len(folds) > 1 else (np.nan, np.nan)
    )
    site_lowest, site_highest = _site_range(pair=pair, treatment=treatment, reference=reference)
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
        "fold_lower_95_pp": fold_lower * PERCENTAGE_POINTS,
        "fold_upper_95_pp": fold_upper * PERCENTAGE_POINTS,
        "site_lowest_pp": site_lowest * PERCENTAGE_POINTS,
        "site_highest_pp": site_highest * PERCENTAGE_POINTS,
        "seed_spread_pp": interval["seed_spread"] * PERCENTAGE_POINTS,
        "excludes_zero": interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0,
        "folds_agreeing": sum(np.sign(value) == np.sign(interval["difference"]) for value in folds),
        "n_folds": len(folds),
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }


CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Domain | Setting | Scope | Contrast | ΔMAE (pp of capacity) | 95% interval, months "
        "and seed | Excludes zero? | Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|---|---|",
)

HEADLINE_HEADER: Final[tuple[str, str]] = (
    (
        "| Domain | Setting | Contrast | ΔMAE (pp of capacity) | 95% interval, months and seed "
        "| 95% t-interval across the 5 folds | Generators, lowest to highest | Excludes zero? "
        "| Folds agreeing |"
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


def _headline_line(record: IntervalRecord) -> str:
    """Render one headline interval with its fold spread and its range across generators.

    Args:
        record: The interval.

    Returns:
        The row.
    """
    return (
        f"| {record['domain']} | {record['setting']} "
        f"| {record['treatment']} − {record['reference']} | {record['difference_pp']:+.3f} "
        f"| [{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}] "
        f"| [{record['fold_lower_95_pp']:+.3f}, {record['fold_upper_95_pp']:+.3f}] "
        f"| {record['site_lowest_pp']:+.3f} to {record['site_highest_pp']:+.3f} "
        f"| {'**yes**' if record['excludes_zero'] else 'no'} "
        f"| {record['folds_agreeing']} of {record['n_folds']} |"
    )


def _section_lines(
    *,
    records: list[IntervalRecord],
    section: str,
    setting: str | None = None,
    render: Callable[[IntervalRecord], str] = _line,
) -> list[str]:
    """Render every interval in one section of the report, optionally at one setting only.

    Args:
        records: Every interval.
        section: The section to render.
        setting: The setting to keep, or `None` for both.
        render: How to render one row.

    Returns:
        Markdown rows.
    """
    return [
        render(record)
        for record in records
        if record["section"] == section and setting in (None, record["setting"])
    ]


def _named_contrasts(
    *, domain: Domain, variant: VariantType, reference: Callable[[str], str]
) -> list[tuple[str, str]]:
    """Return per named set: blend against control, control against single, blend against single.

    Args:
        domain: The domain.
        variant: Plain or enriched.
        reference: Each set's best single, by set name.

    Returns:
        (treatment, reference) pairs, then the stack against XGBoost on every column.
    """
    contrasts: list[tuple[str, str]] = []
    for name in domain.named_sets:
        xgb, control = _arm(name, variant, "xgb"), _arm(name, variant, "control")
        contrasts += [(xgb, control), (control, reference(name)), (xgb, reference(name))]
    return [*contrasts, (_arm("everything", variant, "stack"), _arm("everything", variant, "xgb"))]


def _deciding_contrasts(
    *, domain: Domain, best: BestType, setting: SettingType
) -> list[tuple[str, str]]:
    """Return the deciding contrasts: every named set's enriched blend against its best single.

    Args:
        domain: The domain.
        best: Each set's enriched best single.
        setting: The setting, which sets the best single.

    Returns:
        (treatment, reference) pairs.
    """
    return _named_contrasts(
        domain=domain, variant="rich", reference=lambda name: best[name, setting]
    )


def _plain_contrasts(*, domain: Domain) -> list[tuple[str, str]]:
    """Return the contrasts the plan named before the first run, now secondary.

    Args:
        domain: The domain.

    Returns:
        (treatment, reference) pairs, each against the best single fixed from the published tables.
    """
    return _named_contrasts(
        domain=domain,
        variant="plain",
        reference=lambda name: domain.single(domain.blend_set(name).best_single),
    )


def _exploratory_contrasts(*, domain: Domain, best: BestType) -> list[tuple[str, str]]:
    """Return every set's method contrasts not already reported, and the enriched singles' gains.

    Args:
        domain: The domain.
        best: Each set's enriched best single.

    Returns:
        (treatment, reference) pairs.
    """
    reported = {
        *_deciding_contrasts(domain=domain, best=best, setting="pooled"),
        *_plain_contrasts(domain=domain),
    }
    contrasts: list[tuple[str, str]] = []
    for variant in VARIANTS:
        for blend in domain.sets:
            single = (
                best[blend.name, "pooled"]
                if variant == "rich"
                else domain.single(blend.best_single)
            )
            name = blend.name
            contrasts += [
                pair
                for pair in (
                    (_arm(name, variant, "xgb"), _arm(name, variant, "control")),
                    (_arm(name, variant, "control"), single),
                    (_arm(name, variant, "xgb"), single),
                    (_arm(name, variant, "mean"), single),
                    (_arm(name, variant, "stack"), single),
                    (_arm(name, variant, "equal"), single),
                    (_arm(name, variant, "stack"), _arm(name, variant, "xgb")),
                    (_arm(name, variant, "stack"), f"{single}_seed_stack"),
                )
                if pair not in reported
            ]
    contrasts += [(rich(p), domain.single(p)) for p in domain.products]
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


def _single_table(*, losses: pl.DataFrame, domain: Domain) -> list[str]:
    """Render every single-product arm's error at both settings.

    Args:
        losses: The domain's losses.
        domain: The domain.

    Returns:
        Markdown lines.
    """
    by_setting = {s: losses.filter(pl.col("setting") == s) for s in SETTINGS}
    lines = [
        f"#### {domain.name.capitalize()}: every single-product arm, each a candidate best single",
        "",
        "| Product | Arm | MAE, primary setting | MAE, second setting |",
        "|---|---|---|---|",
    ]
    for arm in sorted(_single_features(domain=domain), key=domain.product_of):
        second = _mae(losses=by_setting["sensitivity"], arm=arm)
        lines.append(
            f"| {domain.product_of(arm)} | {arm} "
            f"| {_mae(losses=by_setting['pooled'], arm=arm):.3f} "
            f"| {'–' if np.isnan(second) else f'{second:.3f}'} |"
        )
    return lines


def _blend_table(
    *, losses: pl.DataFrame, domain: Domain, variant: VariantType, best: BestType
) -> list[str]:
    """Render every set's blends in one variant, primary setting with the second in brackets.

    Args:
        losses: The domain's losses.
        domain: The domain.
        variant: Plain or enriched.
        best: Each set's enriched best single.

    Returns:
        Markdown lines.
    """
    by_setting = {s: losses.filter(pl.col("setting") == s) for s in SETTINGS}
    methods = ("xgb", "control", "mean", "stack", "equal")
    label = "enriched columns" if variant == "rich" else "plain columns"
    lines = [
        f"#### {domain.name.capitalize()}: blends of {label}, primary setting (second in brackets)",
        "",
        "| Set | Products | Best single, primary | Best single, second | "
        + " | ".join(f"`{m}`" for m in methods)
        + " |",
        "|---" * (4 + len(methods)) + "|",
    ]
    for blend in domain.sets:
        if variant == "rich":
            firsts = best[blend.name, "pooled"], best[blend.name, "sensitivity"]
        else:
            firsts = (domain.single(blend.best_single),) * 2
        best_cells = [
            f"{arm} {_mae(losses=by_setting[setting], arm=arm):.3f}"
            for arm, setting in zip(firsts, SETTINGS, strict=True)
        ]
        cells = []
        for method in methods:
            arm = _arm(blend.name, variant, method)
            second = _mae(losses=by_setting["sensitivity"], arm=arm)
            cell = f"{_mae(losses=by_setting['pooled'], arm=arm):.3f}"
            cells.append(cell if np.isnan(second) else f"{cell} ({second:.3f})")
        lines.append(
            f"| {blend.name} | {', '.join(blend.products)} | "
            + " | ".join(best_cells)
            + " | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def _usable_from(*, domain: Domain, blend: BlendSet) -> str:
    """Return the first month every product in a set serves, the latest of their starts.

    Args:
        domain: The domain.
        blend: The set.

    Returns:
        The month, as `YYYY-MM`.
    """
    return max(USABLE_FROM[domain.name][p] for p in blend.products)


def _latency_label(*, hours: float) -> str:
    """Return a latency in words, in days once it reaches one, rounded to one decimal place.

    Args:
        hours: The latency.

    Returns:
        Such as `about 3.5 h` or `about 5.1 days`.
    """
    days = round(hours / HOURS_PER_DAY, 1)
    if days < 1.0:
        return f"about {hours:g} h"
    return f"about {days:g} day{'s' if days > 1.0 else ''}"


def _coverage_lines(*, frames: dict[DomainType, pl.DataFrame]) -> list[str]:
    """Render each set's latency, whether a live service could use it, and its coverage.

    Args:
        frames: Each domain's common rows, for the date the scored rows start.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            "| Domain | Set | Products | Available after (slowest input, plus the enriched "
            "blend's own neighbouring hours) "
            "| Live, Great-Britain-wide? | History only (CAMS or ERA5)? "
            "| Needs ICON-D2's domain? | Archive serves every product from | Scored rows start |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for domain in DOMAINS:
        first_row = f"{frames[domain.name]['time'].min():%Y-%m-%d}"
        for blend in domain.sets:
            context = ENRICHED_LATENCY_HOURS[domain.name]
            slowest = max(LATENCY_HOURS[p] for p in blend.products) + context
            history_only = bool(HISTORY_ONLY_PRODUCTS & set(blend.products))
            regional = "icon_d2" in blend.products
            live_gb = live_great_britain_wide(products=blend.products)
            latency = _latency_label(hours=slowest)
            lines.append(
                f"| {domain.name} | {blend.name} | {', '.join(blend.products)} | {latency} "
                f"| {'yes' if live_gb else 'no'} | {'yes' if history_only else 'no'} "
                f"| {'yes' if regional else 'no'} | {_usable_from(domain=domain, blend=blend)} "
                f"| {first_row} |"
            )
    return lines


def _band_lines(
    *, losses: pl.DataFrame, frame: pl.DataFrame, contrasts: list[tuple[str, str]]
) -> list[str]:
    """Break each contrast's gain down by measured output, and find the most-improved rows' share.

    Each row's difference is averaged over the seeds first.

    Args:
        losses: The domain's primary-setting losses.
        frame: The domain's common rows, for the measured output.
        contrasts: The (treatment, reference) pairs to break down.

    Returns:
        Markdown lines.
    """
    labels = [f"{low:.1f} to {high:.1f}" for low, high in pairwise(OUTPUT_BANDS)]
    labels.append(f"{OUTPUT_BANDS[-1]:.1f} and above")
    band = pl.lit(labels[0])
    for low, label in zip(OUTPUT_BANDS[1:], labels[1:], strict=True):
        band = pl.when(pl.col("fraction") >= low).then(pl.lit(label)).otherwise(band)
    output = frame.select(
        "site",
        "time",
        fraction=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw"),
    ).with_columns(band=band)
    lines = [
        (
            "| Contrast | Measured output, fraction of capacity | Share of rows | Reference MAE "
            "| ΔMAE within the band | Share of the whole gain |"
        ),
        "|---|---|---|---|---|---|",
    ]
    shares: list[str] = []
    for treatment, reference in contrasts:
        rows = (
            losses.filter(pl.col("arm") == treatment)
            .select("site", "time", "seed", treatment=pl.col(METRIC))
            .join(
                losses.filter(pl.col("arm") == reference).select(
                    "site", "time", "seed", reference=pl.col(METRIC)
                ),
                on=["site", "time", "seed"],
            )
            .group_by("site", "time")
            .agg(pl.col("treatment").mean(), pl.col("reference").mean())
            .with_columns(difference=pl.col("treatment") - pl.col("reference"))
            .join(output, on=["site", "time"])
        )
        total = float(rows["difference"].sum())
        by_band = (
            rows.group_by("band")
            .agg(
                share=pl.len() / rows.height,
                reference=pl.col("reference").mean(),
                difference=pl.col("difference").mean(),
                of_gain=pl.col("difference").sum() / total,
                low=pl.col("fraction").min(),
            )
            .sort("low")
        )
        lines += [
            f"| {treatment} − {reference} | {row['band']} | {row['share']:.1%} "
            f"| {row['reference'] * PERCENTAGE_POINTS:.3f} "
            f"| {row['difference'] * PERCENTAGE_POINTS:+.3f} | {row['of_gain']:.0%} |"
            for row in by_band.iter_rows(named=True)
        ]
        ordered = np.sort(rows["difference"].to_numpy())
        top = ordered[: int(len(ordered) * MOST_IMPROVED_SHARE)]
        peak_share = float(np.cumsum(ordered).min()) / total
        shares.append(
            f"{treatment} − {reference}: the most-improved {MOST_IMPROVED_SHARE:.0%} of rows "
            f"carry {top.sum() / total:.0%} of the whole gain, and "
            f"{(ordered < 0).mean():.0%} of rows improve. The running sum peaks at "
            f"{peak_share:.0%} of the net gain where the improving hours end, before the "
            "worsening hours bring it back down. A share above 100% means the other rows are "
            "worse in total."
        )
    return [*lines, "", *(f"- {share}" for share in shares)]


def _feature_lines(*, domain: Domain, best: BestType) -> list[str]:
    """Render every fitted arm's feature columns, and which settings it was fitted at.

    Args:
        domain: The domain.
        best: Each set's enriched best single.

    Returns:
        Markdown lines.
    """
    settings: dict[str, list[str]] = {}
    for arm, setting, *_ in (
        *_single_jobs(domain=domain),
        *_blend_jobs(domain=domain, best=best),
    ):
        settings.setdefault(arm, []).append("primary" if setting == "pooled" else "second")
    features = _single_features(domain=domain) | _blend_features(domain=domain, best=best)
    lines = [
        f"#### {domain.name.capitalize()}: every fitted arm's feature columns",
        "",
        "| Arm | Settings | Columns | Feature columns |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {arm} | {', '.join(settings[arm])} | {len(features[arm])} | "
        + ", ".join(f"`{f}`" for f in features[arm])
        + " |"
        for arm in settings
    ]
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
        for variant in VARIANTS:
            for blend in domain.sets:
                cross = _mae(losses=at_setting, arm=_arm(blend.name, variant, "stack"))
                in_sample = _mae(
                    losses=at_setting, arm=_arm(blend.name, variant, "stack_in_sample")
                )
                lines.append(
                    f"| {domain.name} | {setting} | {_arm(blend.name, variant, 'stack')} "
                    f"| {cross:.3f} | {in_sample:.3f} | {in_sample - cross:+.4f} |"
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
        The rows, with the enriched columns and every blend column added.
    """
    frame = with_export_cap(
        dataset=weather_products.with_eras(
            frame=_add_time_features(
                dataset=weather_products.common_rows(frame=weather_products.joined())
            )
        )
    )
    return _with_blend_columns(
        frame=weather_products.with_irradiance_context(frame=frame), domain=SOLAR
    )


def _wind_frame() -> pl.DataFrame:
    """Build the wind study's common rows exactly as `wind_products.main` builds them.

    Returns:
        The rows, with the enriched columns and every blend column added.
    """
    frame = weather_products.with_eras(
        frame=_add_time_features(
            dataset=wind_products.common_rows(frame=wind_products.joined(sites=_wind_sites()))
        )
    )
    return _with_blend_columns(frame=wind_products.with_wind_context(frame=frame), domain=WIND)


class _Outputs(TypedDict):
    """Everything one domain's run produced."""

    losses: pl.DataFrame
    weights: pl.DataFrame
    predictions: pl.DataFrame
    best: BestType


def _run_domain(
    *, domain: Domain, frame: pl.DataFrame, singles: pl.DataFrame, resume: bool
) -> _Outputs:
    """Choose the best singles, fit every blend, derive the stacks, and assemble the outputs.

    Args:
        domain: The domain.
        frame: The domain's common rows.
        singles: Every single-product arm, already fitted.
        resume: Whether to reuse fits already on disk.

    Returns:
        The losses of every arm, the stacks' weights, every arm's predictions, and the best singles.
    """
    best = _best_singles(losses=singles, domain=domain)
    _LOG.info("%s best singles: %s", domain.name, best)
    fitted = pl.concat(
        [
            singles,
            _fitted(
                frame=frame,
                domain=domain,
                jobs=_blend_jobs(domain=domain, best=best),
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
    seed_losses, seed_weights = _seed_stacks(
        losses=fitted, domain=domain, arms=_seed_stack_arms(domain=domain, best=best)
    )
    losses = pl.concat([fitted, *derived, seed_losses], how="vertical_relaxed").with_columns(
        domain=pl.lit(domain.name)
    )
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
        "best": best,
    }


def _intervals(*, losses: pl.DataFrame, domain: Domain, best: BestType) -> list[IntervalRecord]:
    """Bootstrap every contrast the report prints for one domain.

    Args:
        losses: The domain's losses, every arm and setting.
        domain: The domain.
        best: Each set's enriched best single.

    Returns:
        Every interval, labelled by section.
    """
    by_setting = {s: losses.filter(pl.col("setting") == s) for s in SETTINGS}
    pooled = by_setting["pooled"]
    records: list[IntervalRecord] = []
    for setting, at_setting in by_setting.items():
        for section, contrasts in (
            ("deciding", _deciding_contrasts(domain=domain, best=best, setting=setting)),
            ("plain", _plain_contrasts(domain=domain)),
        ):
            records += [
                _interval(
                    losses=at_setting,
                    contrast=pair,
                    domain=domain,
                    setting=setting,
                    section=section,
                )
                for pair in contrasts
            ]
    controls: list[tuple[str, tuple[str, str]]] = [
        ("sensitivity_control", (SYNTHETIC_ARMS[0], best["everything", "pooled"])),
        ("sensitivity_control", SYNTHETIC_ARMS),
    ]
    if domain.name == "solar":
        controls.append(("positive_control", POSITIVE_CONTROL))
    controls += [
        ("seed_stack_control", (f"{arm}_seed_stack", arm))
        for arm in _seed_stack_arms(domain=domain, best=best)
    ]
    controls += [("exploratory", pair) for pair in _exploratory_contrasts(domain=domain, best=best)]
    records += [
        _interval(losses=pooled, contrast=pair, domain=domain, setting="pooled", section=section)
        for section, pair in controls
    ]
    named_pairs = [
        pair
        for name in domain.named_sets
        for pair in (
            (_arm(name, "rich", "xgb"), _arm(name, "rich", "control")),
            (_arm(name, "rich", "xgb"), best[name, "pooled"]),
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
    reproduction_lines: list[str],
    records: list[IntervalRecord],
    power_version: int,
) -> str:
    """Assemble the markdown report.

    Args:
        frames: Each domain's common rows.
        outputs: Each domain's losses, weights, predictions and best singles.
        reproduction_lines: The reproduction check's rendered lines, from `_reproduction_lines` or
            re-read from a previous run's `reproduction.md`.
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
    lines += [
        f"- Power Delta table version {power_version}.",
        (
            f"- Synthetic product noise, standard deviation as a fraction of capacity: solar "
            f"{SOLAR.synthetic_noise}, wind {WIND.synthetic_noise}."
        ),
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. A negative ΔMAE means "
            "the treatment's error is lower. Every 95% interval resamples whole months and one of "
            "the three fitting seeds, so it covers month-to-month weather and the fitting seed "
            "only; the t-interval across the five folds adds what one set of trained models, "
            "rather than another, contributes."
        ),
        "",
        "#### Reproduction check: refitted single-product arms against the published losses",
        "",
        *reproduction_lines,
        "",
    ]
    for domain in DOMAINS:
        lines += [
            *_leaderboard_lines(
                rows=_leaderboard_rows(losses=outputs[domain.name]["losses"], domain=domain),
                domain=domain,
            ),
            "",
        ]
    lines += [
        "#### Deciding contrasts: each named set's enriched blend against its enriched best single",
        "",
        (
            "Added after the first run (post hoc). Per named set: the enriched blend against its "
            "climatology control, the control against the best single, and the blend against the "
            "best single; then the enriched stack against the enriched XGBoost blend. The best "
            "single is chosen at each setting, by mean error on the common rows."
        ),
        "",
        *HEADLINE_HEADER,
        *_section_lines(records=records, section="deciding", render=_headline_line),
        "",
        "#### Secondary: the plain contrasts named before the first run",
        "",
        *HEADLINE_HEADER,
        *_section_lines(records=records, section="plain", render=_headline_line),
        "",
        "#### Sensitivity positive control: a synthetic product carrying part of the target",
        "",
        *HEADLINE_HEADER,
        *_section_lines(records=records, section="sensitivity_control", render=_headline_line),
        "",
        "#### Positive control: CAMS with ICON-D2 against ICON-D2 alone",
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="positive_control"),
        "",
        "#### Latency and coverage of each set",
        "",
        *_coverage_lines(frames=frames),
        "",
    ]
    for domain in DOMAINS:
        losses = outputs[domain.name]["losses"]
        best = outputs[domain.name]["best"]
        lines += [*_single_table(losses=losses, domain=domain), ""]
        for variant in ("rich", "plain"):
            lines += [
                *_blend_table(losses=losses, domain=domain, variant=variant, best=best),
                "",
            ]
    wind_best = outputs["wind"]["best"]
    band_contrasts = [
        (_arm(name, "rich", "xgb"), wind_best[name, "pooled"]) for name in WIND.named_sets
    ]
    band_contrasts.append(("everything_xgb", WIND.single("icon_d2")))
    lines += [
        "#### Wind: where the gain comes from, by measured output (main XGBoost settings)",
        "",
        *_band_lines(
            losses=outputs["wind"]["losses"].filter(pl.col("setting") == "pooled"),
            frame=frames["wind"],
            contrasts=band_contrasts,
        ),
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
        (
            "| Domain | Setting | Stack | Cross-fitted MAE | In-sample MAE "
            "| In-sample − cross-fitted |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for domain in DOMAINS:
        lines += _gap_lines(losses=outputs[domain.name]["losses"], domain=domain)
    exploratory = [r for r in records if r["section"] == "exploratory"]
    splits = [r for r in records if r["section"] == "exploratory_split"]
    against_control = [r for r in splits if r["reference"].endswith("_control")]
    against_best = [r for r in splits if not r["reference"].endswith("_control")]
    post_months = {r["domain"]: r["n_months"] for r in splits if r["scope"] == "era post"}
    lines += [
        "",
        "#### Exploratory contrasts",
        "",
        (
            f"{len(exploratory)} exploratory intervals, of which "
            f"{sum(r['excludes_zero'] for r in exploratory)} exclude zero. Many compare arms "
            "expected to differ, such as a blend against its own control, and they share rows and "
            "models with one another, so they are not independent tests and no count of chance "
            "exclusions is quoted."
        ),
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="exploratory"),
        "",
        "#### The named enriched blends by site, season and era (exploratory)",
        "",
        (
            f"{len(splits)} intervals, of which {sum(r['excludes_zero'] for r in splits)} exclude "
            f"zero: {len(against_best)} of a blend against its best single, of which "
            f"{sum(r['excludes_zero'] for r in against_best)} exclude zero, and "
            f"{len(against_control)} of a blend against its control, of which "
            f"{sum(r['excludes_zero'] for r in against_control)} exclude zero. The splits reuse "
            "the rows of the whole-record contrasts, and the seasons and eras share each "
            "generator's models, so they are not independent tests of anything. The post era holds "
            + " and ".join(f"{n} months for {d}" for d, n in post_months.items())
            + ", so its intervals rest on so few clusters that they under-cover; read its "
            "fold-sign counts alongside them."
        ),
        "",
        *CONTRAST_HEADER,
        *_section_lines(records=records, section="exploratory_split"),
        "",
    ]
    for domain in DOMAINS:
        lines += [*_feature_lines(domain=domain, best=outputs[domain.name]["best"]), ""]
    return "\n".join(lines)


def _report_only_outputs(*, frame: pl.DataFrame, domain: Domain) -> _Outputs:
    """Rebuild one domain's outputs from `losses.parquet` and `stack_weights.parquet`, with no fit.

    `frame` is only the domain's common rows, built by reading data already on disk; no arm is
    fitted. `best` is recomputed from the saved losses, which is a deterministic lookup, not a fit.

    Args:
        frame: Unused; kept so this function's signature matches `_run_domain`'s callers, and
            `frame.height` can be logged the same way in both modes.
        domain: The domain.

    Returns:
        The domain's losses and weights as saved, an empty predictions frame (the report does not
        read it), and the best singles recomputed from the saved losses.
    """
    del frame
    losses = pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("domain") == domain.name)
    weights = pl.read_parquet(OUTPUT_DIR / "stack_weights.parquet").filter(
        pl.col("domain") == domain.name
    )
    singles = losses.filter(pl.col("arm").is_in(list(_single_features(domain=domain))))
    return {
        "losses": losses,
        "weights": weights,
        "predictions": pl.DataFrame(),
        "best": _best_singles(losses=singles, domain=domain),
    }


def main() -> int:
    """Refit the single products, check them, fit and stack every blend, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse per-arm fits a previous run left on disk."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Rebuild report.md from losses.parquet, stack_weights.parquet and intervals.parquet "
            "already on disk, with no refit. Move the current report.md aside first."
        ),
    )
    parser.add_argument(
        "--power-version",
        type=int,
        help=(
            "With --report-only, the power Delta table version the fits on disk read, as the "
            "replaced report prints it."
        ),
    )
    arguments = parser.parse_args()
    if arguments.report_only != (arguments.power_version is not None):
        parser.error("--power-version goes with --report-only, and --report-only needs it")
    resume = arguments.resume
    started = datetime.now(tz=UTC)

    power_version = arguments.power_version if arguments.report_only else _power_version()
    frames: dict[DomainType, pl.DataFrame] = {"solar": _solar_frame(), "wind": _wind_frame()}
    for name, frame in frames.items():
        _LOG.info("%s common rows: %d", name, frame.height)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.report_only:
        reproduction_lines = (OUTPUT_DIR / "reproduction.md").read_text().splitlines()
        outputs = {
            domain.name: _report_only_outputs(frame=frames[domain.name], domain=domain)
            for domain in DOMAINS
        }
        records = cast(
            "list[IntervalRecord]", pl.read_parquet(OUTPUT_DIR / "intervals.parquet").to_dicts()
        )
    else:
        singles: dict[DomainType, pl.DataFrame] = {}
        reproduction: list[ReproductionRow] = []
        for domain in DOMAINS:
            singles[domain.name] = _fitted(
                frame=frames[domain.name],
                domain=domain,
                jobs=_single_jobs(domain=domain),
                resume=resume,
            )
            reproduction += _reproduction(fitted=singles[domain.name], domain=domain)
        gate = "\n".join(_reproduction_lines(rows=reproduction)) + "\n"
        (OUTPUT_DIR / "reproduction.md").write_text(gate)
        sys.stdout.write(gate)
        if not all(row["bit_identical"] for row in reproduction):
            _LOG.error("the single-product arms do not reproduce the published losses; stopping")
            return 1
        reproduction_lines = gate.splitlines()

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
            pl.concat(
                [outputs[d.name][key] for d in DOMAINS], how="vertical_relaxed"
            ).write_parquet(OUTPUT_DIR / f"{'stack_weights' if key == 'weights' else key}.parquet")
        records = [
            record
            for domain in DOMAINS
            for record in _intervals(
                losses=outputs[domain.name]["losses"],
                domain=domain,
                best=outputs[domain.name]["best"],
            )
        ]
        pl.DataFrame(records).write_parquet(OUTPUT_DIR / "intervals.parquet")
        shutil.rmtree(OUTPUT_DIR / FITS_DIR_NAME)

    report = _report(
        frames=frames,
        outputs=outputs,
        reproduction_lines=reproduction_lines,
        records=records,
        power_version=power_version,
    )
    (OUTPUT_DIR / "report.md").write_text(report)
    sys.stdout.write(report)
    _LOG.info("finished in %s", datetime.now(tz=UTC) - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
