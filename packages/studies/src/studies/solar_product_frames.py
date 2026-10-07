"""Build the solar past-weather study's common rows, which every later solar study starts from.

Written for the past-weather studies in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>. The rows join the
per-product builds of `studies/beam_diffuse_split/build_dataset.py` onto one frame, keep the hours
every product covers, and add the era, neighbouring-hour, and irradiance-context columns. The
sunshine page's report script, the blending study, and the ENS forecast study all start from these
rows, so they live here and not in a script. Scripts in `studies/nwp_forecast_comparison/` and
`studies/past_weather/` import it.
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl

from studies.arm_runner import (
    SHARED_FEATURES,
    Job,
    dataset_path_for,
)
from studies.commissioning import drop_commissioning_ramp
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    UKV_UPGRADE_MONTH,
    assign_folds,
)
from studies.neighbouring_hours import with_neighbouring_hours
from studies.pv_dataset import CAMS_PATH, nearest_era5_cell, pv_sites, read_era5, solar_hourly_power
from studies.solar import extraterrestrial_horizontal, zenith
from studies.sources import (
    OPEN_METEO_MODELS,
    UNSCORED_EXTRACTED_SPLITS,
    SourceType,
    point_output_path_for,
)

OUTPUT_DIR_NAME: Final[str] = "beam_diffuse_weather_products"
"""The first round's results directory under `STUDY_DATA_DIR`, which the `published` panel names.

The blending study and the chart script read the first round's losses and report from here.
"""


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


UPGRADE_DAY: Final[datetime] = datetime(2026, 1, 21, tzinfo=UTC)
"""The upgrade instant.

The rows from here to the end of January carry the pre-upgrade month label but post-upgrade UKV,
so they are dropped from every arm.
"""


ICON_EU_CORRUPT_BLOCK: Final[tuple[datetime, datetime]] = (
    datetime(2023, 6, 21, 1, tzinfo=UTC),
    datetime(2023, 6, 21, 6, tzinfo=UTC),
)
"""ICON-EU's one known corrupt block, dropped from every arm (see `sources.OPEN_METEO_MODELS`)."""


NEW_PRODUCTS: Final[dict[str, str]] = {
    "sarah3": "sarah-3",
    "icon_dream": "icon-dream-eu",
    "ifs_hres": "ecmwf-ifs-hres",
    "arpege": "arpege-europe",
    "dmi_harmonie": "dmi-harmonie-arome",
    "knmi_harmonie": "knmi-harmonie-arome",
}
"""The six products the second round adds, by arm prefix, as `PRODUCTS` lists the first six."""


ALL_PRODUCTS: Final[dict[str, str]] = PRODUCTS | NEW_PRODUCTS
"""Every product either round scores, by arm prefix."""


UNUSABLE_SPLITS: Final[frozenset[str]] = frozenset(
    prefix
    for prefix, source in NEW_PRODUCTS.items()
    if source in UNSCORED_EXTRACTED_SPLITS
    or (source in OPEN_METEO_MODELS and not OPEN_METEO_MODELS[source].split_scored)
)
"""Products whose own split no arm reads, so they get neither a split arm nor an Erbs arm.

Read from `sources.OPEN_METEO_MODELS` and `sources.UNSCORED_EXTRACTED_SPLITS`, which say why for
each: DMI's served direct flux is exactly zero or exceeds its own global flux on a share of daytime
hours `check_new_products.py` prints into `product_checks.md`, and Open-Meteo derives ARPEGE's and
KNMI's direct beam from each model's own global irradiance
with a separation model, which it documents for both
(<https://open-meteo.com/en/docs/meteofrance-api>, <https://open-meteo.com/en/docs/knmi-api>), as it
does for SARAH-3's, modelled by CM SAF from SARAH-3's own global flux. A split arm would measure the
defect or the separation model, not the weather model, and an Erbs arm exists only as the split
arm's reference.
"""


UKV_SNAPSHOT_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "ukv_trap_global": ("ghi_trap_ukv",),
    "ukv_pair_global": ("ghi_instant_previous_ukv", "ghi_instant_ukv"),
    "ukv_trap_ctx_global": ("ghi_trap_previous_ukv", "ghi_trap_ukv", "ghi_trap_next_ukv"),
    "icon_eu_ctx_global": ("ghi_previous_icon_eu", "ghi_icon_eu", "ghi_next_icon_eu"),
}
"""Four post hoc arms: two build UKV's hour from its own snapshots, and two give UKV's snapshot mean
and ICON-EU's hourly mean the same context.

UKV publishes an instantaneous field each hour, and Open-Meteo's served hourly value is the
snapshot at the hour's end rescaled by a ratio of cosines, where every ICON product serves a true
mean over the hour. `ukv_trap_global` is the mean of the snapshots at both ends of the hour, and
`ukv_pair_global` shows the model both snapshots. The two `_ctx` arms add the hour before and the
hour after. Every arm here is post hoc.
"""


def named(column: str, product: str) -> str:
    """Return a product's copy of an irradiance column, such as `bhi_icon_eu`.

    Args:
        column: The build's column name, such as `bhi_w_m2`.
        product: The product prefix.

    Returns:
        The joined frame's column name.
    """
    return f"{column.removesuffix('_w_m2')}_{product}"


def joined(*, products: tuple[str, ...] = tuple(PRODUCTS)) -> pl.DataFrame:
    """Inner-join every product on the site-hours all of them cover.

    Args:
        products: The arm prefixes to join, keys of `ALL_PRODUCTS`, which must include
            `BASE_PRODUCT`. The default is the first round's six.

    Returns:
        One row per common site-hour, carrying `ghi_<p>`, `bhi_<p>`, `dhi_<p>`, `erbs_bhi_<p>`, and
        `erbs_dhi_<p>` for every product `p`, and the base product's power, geometry, and
        temperature. Where the products include UKV, the rows are those on which UKV's snapshots
        also exist, and where they include ICON-EU, those on which ICON-EU's neighbouring hours do.

    Raises:
        ValueError: If `products` leaves out `BASE_PRODUCT`, whose power every row takes.
    """
    if BASE_PRODUCT not in products:
        msg = f"every panel needs {BASE_PRODUCT}, whose power and temperature the rows take"
        raise ValueError(msg)
    irradiance = ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2", "erbs_bhi_w_m2", "erbs_dhi_w_m2")
    base = pl.read_parquet(dataset_path_for(source=ALL_PRODUCTS[BASE_PRODUCT]))
    frame = base.with_columns(
        pl.col(column).alias(named(column, BASE_PRODUCT)) for column in irradiance
    ).drop(*irradiance)
    for product in products:
        if product == BASE_PRODUCT:
            continue
        other = pl.read_parquet(dataset_path_for(source=ALL_PRODUCTS[product])).select(
            "site",
            "time",
            *(pl.col(column).alias(named(column, product)) for column in irradiance),
        )
        frame = frame.join(other, on=["site", "time"], how="inner")
    if "ukv" in products:
        frame = frame.join(ukv_snapshots(), on=["site", "time"], how="inner")
    if "icon_eu" in products:
        frame = frame.join(icon_eu_context(), on=["site", "time"], how="inner")
    return frame.sort("site", "time")


def neighbours(*, frame: pl.DataFrame, column: str, prefix: str, product: str) -> pl.DataFrame:
    """Return one column's value in the hour before and the hour after each row.

    Args:
        frame: One row per (site, time), carrying `column`.
        column: The column to shift.
        prefix: The joined frame's name stem, such as `ghi` or `ghi_trap`.
        product: The product suffix.

    Returns:
        One row per (site, time) with `<prefix>_previous_<product>` and `<prefix>_next_<product>`.
    """
    shifted = {
        "previous": frame.select("site", pl.col("time").dt.offset_by("1h"), pl.col(column)),
        "next": frame.select("site", pl.col("time").dt.offset_by("-1h"), pl.col(column)),
    }
    previous, following = (
        shifted[side].rename({column: f"{prefix}_{side}_{product}"})
        for side in ("previous", "next")
    )
    return previous.join(following, on=["site", "time"], how="inner")


MAX_INSTANT_OVER_TOA: Final[float] = 1.1
"""How far `ghi_instant_w_m2` may exceed the top-of-atmosphere flux at its own instant before the
row is dropped as a sunrise spike.

Open-Meteo serves the `_instant` columns by multiplying its stored hourly mean by the
instantaneous-over-hour-mean cosine of the solar zenith angle, computed without refraction
(`verify_ukv_lineage.GEOMETRY_FACTOR_RANGE` names and bounds the same factor, for a different,
stricter purpose — picking instants clean enough to compare against the Met Office's own files,
not identifying which served values are unusable). Near sunrise that factor departs from one on
most hours without producing an implausible value, so filtering on the factor's range would drop
more than half of every product's rows. What actually makes a row unusable is the factor's size:
in the first hours after sunrise it can reach the hundreds, and the stored mean at those hours is
still small (1 to 28 W/m2 at the rows this rule drops), so multiplying the two produces a value
with no physical meaning — measured at up to 16,537 W/m2 in this archive, against a real GHI
ceiling of about 1,400 W/m2. The stored mean's own 1 W/m2 rounding is a minor contributor,
explaining the excess over the ceiling in only about one row in ten. The row is instead flagged
directly against the physical ceiling: the flux a horizontal surface receives with no atmosphere
at all, from `studies.solar.extraterrestrial_horizontal`, evaluated with pvlib's apparent
(refracted) solar zenith angle. Near the horizon that refracted angle allows more flux than
Open-Meteo's refraction-free geometry does, and the 10% margin is a tolerance for that mismatch,
not for a physical process such as cloud enhancement: every row this rule drops has the sun below
4.5 degrees of apparent elevation.
"""


MIN_TOP_OF_ATMOSPHERE_W_M2: Final[float] = 1e-6
"""Floors the top-of-atmosphere flux so a true night-side instant gets a ceiling of (near) zero
rather than exactly zero, which any served value above zero then correctly fails."""


def without_sunrise_spikes(*, frame: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows where `ghi_instant_w_m2` exceeds what is physically possible at that instant.

    Args:
        frame: Rows carrying `site`, `time`, and `ghi_instant_ukv`, one row per (site, time).
        sites: The site list, carrying `site`, `latitude`, and `longitude`.

    Returns:
        `frame`, with every row whose `ghi_instant_ukv` exceeds `MAX_INSTANT_OVER_TOA` times the
        top-of-atmosphere flux at that instant removed.
    """
    coordinates = {
        str(row["site"]): (float(row["latitude"]), float(row["longitude"]))
        for row in sites.to_dicts()
    }
    kept: list[pl.DataFrame] = []
    for (site,), rows in frame.sort("site", "time").group_by(["site"], maintain_order=True):
        latitude, longitude = coordinates[str(site)]
        stamps = rows["time"]
        angle = zenith(stamps=stamps, latitude=latitude, longitude=longitude)
        top_of_atmosphere = extraterrestrial_horizontal(stamps=stamps, zenith_deg=angle)
        ceiling = MAX_INSTANT_OVER_TOA * np.maximum(top_of_atmosphere, MIN_TOP_OF_ATMOSPHERE_W_M2)
        plausible = rows["ghi_instant_ukv"].to_numpy() <= ceiling
        kept.append(rows.filter(pl.Series(plausible)))
    return pl.concat(kept)


def ukv_snapshots() -> pl.DataFrame:
    """Return UKV's instantaneous global irradiance at both ends of each hour, and their mean.

    Every row whose `ghi_instant_ukv` exceeds what solar geometry allows at that instant (see
    `without_sunrise_spikes`) is dropped before the trapezoid mean and the neighbouring-hour
    context are built from it, so a dropped instant also drops the hour that would have averaged
    it in and the neighbouring hour that would have used it as context.

    Returns:
        One row per (site, time) with `ghi_instant_previous_ukv`, `ghi_instant_ukv` and
        `ghi_trap_ukv`, for every hour whose both snapshots were served and neither was a sunrise
        spike.
    """
    download = pl.read_parquet(point_output_path_for(source="ukv")).select(
        "site", "time", ghi_instant_ukv=pl.col("ghi_instant_w_m2")
    )
    clean = without_sunrise_spikes(frame=download, sites=pv_sites())
    previous = clean.select(
        "site",
        time=pl.col("time").dt.offset_by("1h"),
        ghi_instant_previous_ukv=pl.col("ghi_instant_ukv"),
    )
    snapshots = clean.join(previous, on=["site", "time"], how="inner").with_columns(
        ghi_trap_ukv=(pl.col("ghi_instant_previous_ukv") + pl.col("ghi_instant_ukv")) / 2.0
    )
    context = neighbours(frame=snapshots, column="ghi_trap_ukv", prefix="ghi_trap", product="ukv")
    return snapshots.join(context, on=["site", "time"], how="inner")


def icon_eu_context() -> pl.DataFrame:
    """Return ICON-EU's hourly mean in the hour before and the hour after each hour.

    The neighbours are read from the raw download, so on 2023-06-21 the 07:00 UTC row carries the
    corrupt 06:00 value as its previous hour in `icon_eu_ctx_global`, the one exception to that
    block being dropped from every arm. It touches five rows of one post hoc arm.

    Returns:
        One row per (site, time) with `ghi_previous_icon_eu` and `ghi_next_icon_eu`.
    """
    download = pl.read_parquet(point_output_path_for(source="icon-eu")).select(
        "site", "time", "ghi_w_m2"
    )
    return neighbours(frame=download, column="ghi_w_m2", prefix="ghi", product="icon_eu")


CONTEXT_PRODUCTS: Final[tuple[str, ...]] = ("cams", "era5", "icon_d2", "icon_global")
"""The products `with_irradiance_context` adds neighbouring hours for.

UKV's and ICON-EU's neighbouring hours are already in `joined`'s frame, as `ghi_trap_previous_ukv`,
`ghi_trap_next_ukv`, `ghi_previous_icon_eu`, and `ghi_next_icon_eu`.
"""


CONTEXT_SOURCES: Final[dict[str, SourceType]] = {"icon_d2": "icon-d2", "icon_global": "icon-global"}
"""The downloads the per-site weather models' neighbouring hours are read from."""


def irradiance_download(*, product: str) -> pl.DataFrame:
    """Return one product's served global irradiance at every site and hour it was downloaded.

    ERA5 is gridded, so each site reads its nearest cell, as `build_dataset.py` does.

    Args:
        product: A key of `CONTEXT_PRODUCTS`.

    Returns:
        One row per (site, time) with `ghi_w_m2`.
    """
    if product == "era5":
        gridded = read_era5(source="open-meteo")
        cells = nearest_era5_cell(sites=pv_sites(), era5=gridded)
        return cells.join(
            gridded,
            left_on=["cell_latitude", "cell_longitude"],
            right_on=["latitude", "longitude"],
        ).select("site", "time", "ghi_w_m2")
    path = (
        CAMS_PATH if product == "cams" else point_output_path_for(source=CONTEXT_SOURCES[product])
    )
    return pl.read_parquet(path).select("site", "time", "ghi_w_m2")


def with_irradiance_context(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the hour before and the hour after each row for every product in `CONTEXT_PRODUCTS`.

    The neighbours are read from each product's own download, not from the scored rows, which
    exclude hours by the target. Each download is checked to reproduce the frame's own column at
    offset zero first, so a neighbour cannot come from a series labelled differently.

    Args:
        frame: The common rows, carrying `ghi_<product>` for every product.

    Returns:
        `frame`, in its own row order, with `ghi_previous_<product>` and `ghi_next_<product>`.

    Raises:
        ValueError: If a download does not reproduce the frame's column at offset zero.
    """
    for product in CONTEXT_PRODUCTS:
        own = f"ghi_{product}"
        frame = with_neighbouring_hours(
            frame=frame,
            source=irradiance_download(product=product),
            columns={
                f"{own}_at_zero": ("ghi_w_m2", 0),
                f"ghi_previous_{product}": ("ghi_w_m2", -1),
                f"ghi_next_{product}": ("ghi_w_m2", 1),
            },
        )
        if not frame[f"{own}_at_zero"].cast(pl.Float64).equals(frame[own].cast(pl.Float64)):
            msg = f"the {product} download does not reproduce {own} at offset zero"
            raise ValueError(msg)
        frame = frame.drop(f"{own}_at_zero")
    return frame


def common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows no product should be scored on, by rules no product's values decide.

    Args:
        frame: The joined frame.

    Returns:
        The frame without zero-half-hour hours, ICON-EU's corrupt block, the post-upgrade tail of
        January 2026, and the commissioning ramp.
    """
    zero_hours = (
        solar_hourly_power(sites=pv_sites())
        .filter(pl.col("has_zero_half_hour"))
        .select("site", "time")
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


def with_eras(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Label each row's era, add the era feature, and cut folds inside each era.

    Args:
        frame: The common rows, carrying `month`.

    Returns:
        The frame with `era`, `era_code` and `fold`.
    """
    post = pl.col("month") >= UKV_UPGRADE_MONTH
    labelled = frame.with_columns(
        era=pl.when(post).then(pl.lit("post")).otherwise(pl.lit("pre")),
        era_code=post.cast(pl.Int8),
    )
    return assign_folds(dataset=labelled, by=("site", "era"))


def arm_columns(
    *, products: tuple[str, ...], with_snapshot_arms: bool
) -> dict[str, tuple[str, ...]]:
    """Return every pooled arm's irradiance columns: three per product, and the UKV snapshot arms.

    Args:
        products: The arm prefixes, keys of `ALL_PRODUCTS`.
        with_snapshot_arms: Whether to add `UKV_SNAPSHOT_ARMS`, which need UKV and ICON-EU.

    Returns:
        Each arm's irradiance columns, in the order the arms are fitted.
    """
    arms: dict[str, tuple[str, ...]] = {}
    for product in products:
        ghi = named("ghi_w_m2", product)
        arms[f"{product}_global"] = (ghi,)
        if product in UNUSABLE_SPLITS:
            continue
        arms[f"{product}_split"] = (ghi, named("bhi_w_m2", product), named("dhi_w_m2", product))
        arms[f"{product}_erbs"] = (
            ghi,
            named("erbs_bhi_w_m2", product),
            named("erbs_dhi_w_m2", product),
        )
    if with_snapshot_arms:
        arms |= UKV_SNAPSHOT_ARMS
    return arms


def jobs(
    *, products: tuple[str, ...] = tuple(PRODUCTS), with_snapshot_arms: bool = True
) -> list[Job]:
    """Return the three arms per product: global only, its own split, and Erbs on its own global.

    A product in `UNUSABLE_SPLITS` gets its global arm alone.

    Args:
        products: The arm prefixes, keys of `ALL_PRODUCTS`. The default is the first round's six.
        with_snapshot_arms: Whether to add `UKV_SNAPSHOT_ARMS`, which need UKV and ICON-EU.

    Returns:
        One job per arm, every arm shown the shared features and the era.
    """
    shared = (*SHARED_FEATURES, "era_code")
    return [
        (arm, "pooled", "power_mw", (*shared, *columns), PRIMARY_HYPER_PARAMETERS, False)
        for arm, columns in arm_columns(
            products=products, with_snapshot_arms=with_snapshot_arms
        ).items()
    ]
