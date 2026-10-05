"""Describe the solar and wind domains of the blending study, and build the rows each domain fits.

Written for the blending study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/836>. A `Domain` names a domain's
products, its blend sets, and its feature columns. `solar_frame` and `wind_frame` build the common
rows with every blend column added, and `contrast_interval` and `contrast_line` turn a paired
contrast into the interval and the report row every later study reuses. Scripts in
`studies/nwp_forecast_comparison/` and `studies/past_weather/` import it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
import polars as pl

from studies import solar_product_frames, wind_product_frames
from studies.arm_runner import SHARED_FEATURES as SOLAR_SHARED_FEATURES
from studies.arm_runner import Job, add_time_features
from studies.blending import climatology_permutation
from studies.bootstrap import (
    bootstrap_difference,
    fold_t_interval,
    per_fold_differences,
)
from studies.export_cap import with_export_cap
from studies.pv_dataset import wind_sites
from studies.sources import STUDY_DATA_DIR

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


PERCENTAGE_POINTS: Final[float] = 100.0


METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""


DomainType = Literal["solar", "wind"]


VariantType = Literal["plain", "rich"]
"""Whether a blend is built from each product's plain columns or from its enriched columns."""


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


def solar_columns(product: str) -> tuple[str]:
    """Return a solar product's one plain feature column: its global horizontal irradiance.

    Args:
        product: A key of `solar_product_frames.PRODUCTS`.

    Returns:
        The column name.
    """
    return (f"ghi_{product}",)


def solar_rich_columns(product: str) -> tuple[str, ...]:
    """Return a solar product's enriched columns: the hour before, the hour, the hour after, more.

    The first three columns are always the hour before, the hour and the hour after, which the
    enriched mean arm averages across products. UKV's three are its hour rebuilt from the snapshots
    at both ends, as the published `ukv_trap_ctx_global` arm reads them; ICON-EU's are the published
    `icon_eu_ctx_global` arm's. CAMS adds its own beam split, which the published page found helps.

    Args:
        product: A key of `solar_product_frames.PRODUCTS`.

    Returns:
        The column names.
    """
    if product == "ukv":
        return ("ghi_trap_previous_ukv", "ghi_trap_ukv", "ghi_trap_next_ukv")
    context = (f"ghi_previous_{product}", f"ghi_{product}", f"ghi_next_{product}")
    return (*context, "bhi_cams", "dhi_cams") if product == "cams" else context


def wind_columns(product: str) -> tuple[str, str, str, str]:
    """Return a wind product's four plain feature columns, as the wind study names them.

    Args:
        product: A key of `wind_product_frames.PRODUCTS`.

    Returns:
        The hub-height speed, that height's direction as sine and cosine, and the 10 m speed.
    """
    return wind_product_frames.wind_columns(product=product)


def wind_rich_columns(product: str) -> tuple[str, ...]:
    """Return a wind product's enriched columns: its plain four, then its neighbouring hours.

    Args:
        product: A key of `wind_product_frames.PRODUCTS`.

    Returns:
        The column names.
    """
    return (*wind_columns(product), *wind_product_frames.context_columns(product=product))


def solar_published_jobs() -> tuple[Job, ...]:
    """Return the published solar study's pooled arms, less the two the enriched arms reproduce.

    Returns:
        The jobs, as `solar_product_frames.jobs` builds them.
    """
    return tuple(
        job
        for job in solar_product_frames.jobs()
        if job[0] not in ("icon_eu_ctx_global", "ukv_trap_ctx_global")
    )


def wind_published_jobs() -> tuple[Job, ...]:
    """Return the published wind study's arms on the common rows, less the step-period arms.

    The step indicator works as a date-regime feature, so an arm carrying it is not a fair single.
    The arms at each of `wind_product_frames.ROW_SET_SETTINGS` are fitted on other row sets, so they
    are left out too.

    Returns:
        The jobs, as `wind_product_frames.jobs` builds them.
    """
    return tuple(
        job
        for job in wind_product_frames.jobs()
        if not job[0].endswith("_step") and job[1] not in wind_product_frames.ROW_SET_SETTINGS
    )


SOLAR: Final[Domain] = Domain(
    name="solar",
    products=tuple(solar_product_frames.PRODUCTS),
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
        BlendSet("everything", tuple(solar_product_frames.PRODUCTS), "cams", "the upper bound"),
    ),
    shared_features=(*SOLAR_SHARED_FEATURES, "era_code"),
    columns=solar_columns,
    rich_columns=solar_rich_columns,
    rich_mean_width=3,
    single_suffix="_global",
    named_sets=("everything", "cams_icon_eu", "live_all"),
    published_losses=STUDY_DATA_DIR / solar_product_frames.OUTPUT_DIR_NAME / "losses.parquet",
    published_jobs=solar_published_jobs(),
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
    shared_features=wind_product_frames.SHARED_FEATURES,
    columns=wind_columns,
    rich_columns=wind_rich_columns,
    rich_mean_width=10,
    single_suffix="_wind",
    named_sets=("everything", "live_gb"),
    published_losses=STUDY_DATA_DIR / wind_product_frames.OUTPUT_DIR_NAME / "losses.parquet",
    published_jobs=wind_published_jobs(),
    rich_published={},
    synthetic_noise=0.45,
)


def mean_columns(*, domain: Domain, blend: BlendSet, variant: VariantType) -> tuple[str, ...]:
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


def with_synthetic(*, frame: pl.DataFrame, domain: Domain) -> pl.DataFrame:
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


def with_blend_columns(*, frame: pl.DataFrame, domain: Domain) -> pl.DataFrame:
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
        for position, name in enumerate(mean_columns(domain=domain, blend=blend, variant="plain"))
    ]
    means += [
        pl.mean_horizontal(
            *(domain.rich_columns(product)[position] for product in blend.products)
        ).alias(name)
        for blend in domain.sets
        for position, name in enumerate(mean_columns(domain=domain, blend=blend, variant="rich"))
    ]
    plain = climatology_permutation(
        frame=with_synthetic(frame=frame.with_columns(means), domain=domain),
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


def site_range(*, pair: pl.DataFrame, treatment: str, reference: str) -> tuple[float, float]:
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


def contrast_interval(
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
    site_lowest, site_highest = site_range(pair=pair, treatment=treatment, reference=reference)
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


def contrast_line(record: IntervalRecord) -> str:
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


def solar_frame() -> pl.DataFrame:
    """Build the solar study's common rows exactly as `weather_products.main` builds them.

    Returns:
        The rows, with the enriched columns and every blend column added.
    """
    frame = with_export_cap(
        dataset=solar_product_frames.with_eras(
            frame=add_time_features(
                dataset=solar_product_frames.common_rows(frame=solar_product_frames.joined())
            )
        )
    )
    return with_blend_columns(
        frame=solar_product_frames.with_irradiance_context(frame=frame), domain=SOLAR
    )


def wind_frame() -> pl.DataFrame:
    """Build the wind study's common rows exactly as `wind_products.main` builds them.

    Returns:
        The rows, with the enriched columns and every blend column added.
    """
    frame = solar_product_frames.with_eras(
        frame=add_time_features(
            dataset=wind_product_frames.common_rows(
                frame=wind_product_frames.joined(sites=wind_sites())
            )
        )
    )
    return with_blend_columns(frame=wind_product_frames.with_wind_context(frame=frame), domain=WIND)
