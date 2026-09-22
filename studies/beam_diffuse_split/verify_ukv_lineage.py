"""Check Open-Meteo's UKV mirror against the Met Office's own files, and say which lead it serves.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/800>.

**This is a gate rather than a diagnostic: no model is trained on UKV until it has run and been
read.** The Centre for Environmental Data Analysis (CEDA) serves a UKV archive
statistically different from the live UKV, so a mirror can carry a model's
name without being that model, and one matching timestamp is not evidence across four and a half
years and a science upgrade.

The script answers two questions, and only the second is about trust:

- **Which forecast lead does the archive hold?** Open-Meteo ingests every hourly UKV run, and a
  later run overwrites an earlier one for the same valid time, so the archive should hold the T+0
  analysis. A T+0 UKV field is the analysis rather than a forecast, and UKV's 4D-Var assimilates a
  large volume of satellite-derived cloud, so the arm is partly downstream of the same geostationary
  satellite CAMS retrieves from. That changes what the between-product contrast means, which is why
  the lead is measured rather than assumed.
- **Does the mirror reproduce the native field?** Sampling both sides of the PS47 science upgrade on
  2026-01-21 is what would catch an ingest that changed there.

**Neither question can be asked of the archive before 2024-08-12**, when Open-Meteo's UKV downloader
first existed. The Met Office's bucket holds a rolling two-year window, and the earlier half of the
archive was backfilled from a source Open-Meteo does not name, so no sampling can reach it. What
covers that era instead is `studies.served_column_checks.check_direct_is_not_a_separation_model`,
which needs no reference product.

The comparison is against the `_instant` columns, never the default hourly ones. The native file
holds an instantaneous snapshot; Open-Meteo's default is a backward-looking hourly mean derived from
it. Comparing the mean against the snapshot would make a faithful mirror look broken.

Coordinates are read at run time from the private roster and never written: the table names
anonymised site labels and differences in W m⁻².

Run it with `uv run --with netcdf4 python
studies/beam_diffuse_split/verify_ukv_lineage.py`.
"""

import argparse
import datetime as dt
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
import xarray as xr
from build_dataset import _pv_sites
from fetch_open_meteo_point import (
    HOURLY_VARIABLES,
    INSTANT_SUFFIX,
    _renamed,
    _solar_geometry,
    fetch_point_frame,
)
from pyproj import CRS
from sources import OPEN_METEO_MODELS
from studies.grid_sampling import sample_nearest_cell

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("verify_ukv_lineage")

BUCKET_URL: Final[str] = (
    "https://met-office-atmospheric-model-data.s3-eu-west-2.amazonaws.com/uk-deterministic-2km"
)
"""The Met Office's own UKV archive, served over plain HTTPS.

The bucket is public and unsigned, so no credentials are involved and no object-store client is
needed. The bucket holds a **rolling two-year window**, so the span this script can reach shortens
by a day every day.
"""

NATIVE_FILE_NAMES: Final[dict[str, str]] = {
    "ghi": "radiation_flux_in_shortwave_total_downward_at_surface",
    "bhi": "radiation_flux_in_shortwave_direct_downward_at_surface",
}
"""The bucket stores one variable per file, so each instant and lead needs one read per flux."""

SERVED_INSTANT_COLUMNS: Final[dict[str, str]] = {
    "ghi": "ghi_instant_w_m2",
    "bhi": "bhi_instant_w_m2",
}
"""Which served column each native file is compared against, after `_renamed`."""

PS47_OPERATIONAL: Final[dt.date] = dt.date(2026, 1, 21)
"""When the Met Office's PS47 science package went operational.

The archive's encoding changed at this date, so Open-Meteo's ingest could have changed too, and an
unstratified sample would average across the boundary rather than test it.
"""

EXPECTED_LEAD_HOURS: Final[int] = 0
"""Which lead the archive is expected to hold, from reading Open-Meteo's downloader.

UKV runs hourly, Open-Meteo ingests every run with a delay of a little over four hours, radiation is
written at every step including the first, and a later run overwrites an earlier one for the same
valid time. The last writer for a valid time is therefore the run initialised at that instant.
"""

SWEPT_LEAD_HOURS: Final[tuple[int, ...]] = (0, 1, 3, 6)
"""Which leads are pulled for each sampled instant.

Wide enough that the matching lead wins by a margin rather than by a rounding.
"""

MAX_MEDIAN_DIFFERENCE_W_M2: Final[float] = 2.0
"""How far the matching lead may sit from the served value before the mirror is not the model.

Measured at one meter's coordinates across five instants spanning both sides of PS47, the T+0
nearest-cell
difference runs from 0.11 to 0.55 W m⁻², which is the scale of the 1 W m⁻² rounding Open-Meteo's
stored column carries. Every other lead sat between 1.3 and 350 W m⁻² away, so the threshold
separates a faithful mirror from the nearest wrong answer by a wide margin rather than a fine one.
"""

BROKEN_CLOUD_CLEARNESS: Final[tuple[float, float]] = (0.25, 0.65)
"""The clearness indices that count as broken cloud.

**The sample is stratified by sky condition because otherwise the sweep cannot resolve what it is
for.** Under broken cloud the native field varies by more than 100 W m⁻² across a handful of grid
cells, so that is the condition which discriminates hardest between leads. The clearest hours the
window offers are sampled alongside it, because a flat field is where a wrong grid cell would show
up as agreement that means nothing.

**There is no absolute clear-sky threshold, because the window that brackets PS47 cannot meet one.**
PS47 went operational on 2026-01-21, and at 53 N the midwinter sun reaches 17 degrees of elevation
at noon. Measured over the 21 days either side, no hour has every meter above a clearness index of
0.55, so a threshold of 0.7 selects nothing and the run has nothing to verify. `_sample_instants`
takes the clearest hours available instead and reports the clearness it actually got.
"""

GEOMETRY_FACTOR_RANGE: Final[tuple[float, float]] = (0.8, 1.25)
"""How far the hour's solar geometry may depart from flat for an instant to be sampled.

The factor is the instantaneous cosine of the solar zenith angle over its mean across the preceding
hour. Open-Meteo multiplies the stored hourly mean by it to serve the `_instant` column, so where
the factor is far from one it also multiplies that column's 1 W m⁻² rounding, producing a difference
that says nothing about lineage. **The condition is written on the factor rather than on solar
elevation because the factor is what does the damage**: a low midwinter noon sun has a factor near
one and is perfectly usable, whereas an hour near sunrise at any time of year is not.
"""


class SampledInstant(NamedTuple):
    """One valid time to pull from both sources.

    Attributes:
        valid_time: The instant, in UTC.
        era: `pre-PS47` or `post-PS47`.
        sky: `clear` or `broken`.
    """

    valid_time: dt.datetime
    era: str
    sky: str


def _native_url(*, valid_time: dt.datetime, lead_hours: int, flux: str) -> str:
    """Return the bucket URL for one flux at one valid time and lead.

    Args:
        valid_time: The instant the file is valid at.
        lead_hours: How far after its run's initialisation that instant falls.
        flux: `ghi` or `bhi`.

    Returns:
        The full HTTPS URL of the netCDF file.
    """
    run = valid_time - dt.timedelta(hours=lead_hours)
    return (
        f"{BUCKET_URL}/{run:%Y%m%dT%H%M}Z/{valid_time:%Y%m%dT%H%M}Z"
        f"-PT{lead_hours:04d}H00M-{NATIVE_FILE_NAMES[flux]}.nc"
    )


def _read_native_at_sites(
    *, valid_time: dt.datetime, lead_hours: int, flux: str, sites: pl.DataFrame
) -> dict[str, float] | None:
    """Read one native file and return the nearest grid cell's value at each site.

    One file covers the whole 970-by-1042 domain, so a single read serves every meter. The file is
    HDF5 and `xarray`'s default engine cannot open a file-like object, so the bytes land in a
    temporary file first, the way `build_dataset` already handles the Copernicus archives.

    The projection comes from the file's own CF grid-mapping attributes rather than from constants
    copied out of a downloader, so the mapping cannot drift from the data it indexes.

    Args:
        valid_time: The instant to read.
        lead_hours: Which run's copy of that instant to read.
        flux: `ghi` or `bhi`.
        sites: The roster, carrying `site`, `latitude`, and `longitude`.

    Returns:
        The flux in W m⁻² at each site's nearest grid cell, or `None` if the bucket has no such
        file — which is the ordinary answer outside the rolling window, and for a lead a shortened
        run never reached.
    """
    url = _native_url(valid_time=valid_time, lead_hours=lead_hours, flux=flux)
    try:
        payload = urllib.request.urlopen(url, timeout=300).read()
    except urllib.error.HTTPError as refusal:
        # Only "there is no such object" is an ordinary answer. A 5xx means the bucket is
        # struggling, and reading it as an absent file would drop that lead from the comparison
        # without saying so. S3 answers a missing key with 403 when anonymous listing is denied.
        if refusal.code in (403, 404):
            return None
        raise

    handle, path = tempfile.mkstemp(suffix=".nc")
    try:
        with os.fdopen(handle, "wb") as scratch:
            scratch.write(payload)
        with xr.open_dataset(path) as dataset:
            return sample_nearest_cell(
                field=dataset[_flux_variable_name(dataset=dataset)],
                sites=sites,
                crs=CRS.from_cf(dict(dataset["lambert_azimuthal_equal_area"].attrs)),
            )
    finally:
        Path(path).unlink(missing_ok=True)


def _flux_variable_name(*, dataset: xr.Dataset) -> str:
    """Return the one two-dimensional flux variable in a native file.

    Args:
        dataset: One opened native file.

    Returns:
        The variable's name.

    Raises:
        ValueError: If the file does not hold exactly one such variable.
    """
    candidates = [
        name
        for name, variable in dataset.data_vars.items()
        if variable.ndim == 2 and "bnds" not in str(name)
    ]
    if len(candidates) != 1:
        msg = f"expected one gridded flux in the native file, found {candidates}"
        raise ValueError(msg)
    return str(candidates[0])


def _sample_instants(
    *, served: pl.DataFrame, era: str, per_stratum: int, seed: int
) -> list[SampledInstant]:
    """Choose the clearest and the broken-cloud instants from one era's served rows.

    An instant qualifies only where the whole roster agrees about the sky, so that all six meters
    are sampled under the condition the stratum names rather than one of them.

    Args:
        served: One window of served rows, carrying the clearness index and the solar geometry.
        era: The label to stamp on the chosen instants.
        per_stratum: How many instants to take from each sky condition.
        seed: Seeds the choice among the broken-cloud candidates, so a rerun samples the same ones.

    Returns:
        The chosen instants, fewer than asked for where the window is short of either sky.
    """
    lowest_factor, highest_factor = GEOMETRY_FACTOR_RANGE
    factor = pl.col("cos_zenith_instant") / pl.col("cos_zenith_hour_mean")
    by_instant = (
        served.filter(
            (pl.col("ghi_w_m2") > 20.0) & (factor > lowest_factor) & (factor < highest_factor)
        )
        .group_by("time")
        .agg(
            lowest=pl.col("clearness_index").min(),
            highest=pl.col("clearness_index").max(),
            sites=pl.len(),
        )
        .filter(pl.col("sites") == served["site"].n_unique())
    )
    broken = by_instant.filter(
        (pl.col("lowest") >= BROKEN_CLOUD_CLEARNESS[0])
        & (pl.col("highest") <= BROKEN_CLOUD_CLEARNESS[1])
    )
    strata = {
        "clearest": by_instant.sort("lowest", descending=True).head(per_stratum),
        "broken": broken.sample(n=min(per_stratum, broken.height), seed=seed),
    }
    chosen: list[SampledInstant] = []
    for sky, candidates in strata.items():
        if candidates.height == 0:
            _LOG.warning("no %s instant in the %s window", sky, era)
            continue
        _LOG.info(
            "%s %s: %d instants, roster-minimum clearness %.2f to %.2f",
            era,
            sky,
            candidates.height,
            candidates["lowest"].min(),
            candidates["lowest"].max(),
        )
        chosen += [
            SampledInstant(valid_time=row["time"], era=era, sky=sky)
            for row in candidates.sort("time").to_dicts()
        ]
    return chosen


def _compare_one(
    *, instant: SampledInstant, served: pl.DataFrame, sites: pl.DataFrame
) -> list[dict[str, object]]:
    """Compare every swept lead against the served snapshot at one instant.

    Args:
        instant: The valid time and its stratum.
        served: The era's served rows.
        sites: The roster, carrying `site`, `latitude`, and `longitude`.

    Returns:
        One record per (lead, flux) with the median and worst absolute difference across sites.
    """
    at_instant = served.filter(pl.col("time") == instant.valid_time)
    records: list[dict[str, object]] = []
    for lead_hours in SWEPT_LEAD_HOURS:
        for flux, served_column in SERVED_INSTANT_COLUMNS.items():
            native = _read_native_at_sites(
                valid_time=instant.valid_time, lead_hours=lead_hours, flux=flux, sites=sites
            )
            if native is None:
                _LOG.info("%s T+%d %s: no file in the bucket", instant.valid_time, lead_hours, flux)
                continue
            differences = np.array(
                [
                    abs(native[str(row["site"])] - float(row[served_column]))
                    for row in at_instant.to_dicts()
                ]
            )
            records.append(
                {
                    "valid_time": instant.valid_time,
                    "era": instant.era,
                    "sky": instant.sky,
                    "lead_hours": lead_hours,
                    "flux": flux,
                    "median_difference_w_m2": float(np.median(differences)),
                    "worst_difference_w_m2": float(differences.max()),
                }
            )
    return records


def _report(*, results: pl.DataFrame) -> None:
    """Log the per-lead agreement table."""
    summary = (
        results.group_by("era", "sky", "lead_hours")
        .agg(
            median_difference_w_m2=pl.col("median_difference_w_m2").median(),
            worst_difference_w_m2=pl.col("worst_difference_w_m2").max(),
        )
        .sort("era", "sky", "lead_hours")
    )
    # Polars shows ten rows by default, which would truncate the table this gate exists to print.
    with pl.Config(tbl_rows=-1, tbl_width_chars=120):
        _LOG.info("per-lead agreement against the Met Office's own files:\n%s", summary)


def _assert_lead_is_as_expected(*, results: pl.DataFrame) -> None:
    """Assert the expected lead both wins every stratum and agrees to the rounding.

    Winning alone would not be enough: the closest of four wrong answers still wins. Agreeing to the
    rounding alone would not be enough either, because a second lead agreeing just as well would
    mean the sweep had not established which one the archive holds.

    Args:
        results: One row per sampled (instant, lead, flux).

    Raises:
        ValueError: If another lead matches better, or the expected lead does not agree.
    """
    by_lead = (
        results.group_by("lead_hours")
        .agg(median_difference_w_m2=pl.col("median_difference_w_m2").median())
        .sort("median_difference_w_m2")
    )
    best = by_lead.row(0, named=True)
    if int(best["lead_hours"]) != EXPECTED_LEAD_HOURS:
        msg = (
            f"the archive matches T+{best['lead_hours']} better than the expected "
            f"T+{EXPECTED_LEAD_HOURS} ({best['median_difference_w_m2']:.2f} W m-2). What the arm's "
            "cloud field is has changed, so re-read the downloader before training on it."
        )
        raise ValueError(msg)
    if float(best["median_difference_w_m2"]) > MAX_MEDIAN_DIFFERENCE_W_M2:
        msg = (
            f"T+{EXPECTED_LEAD_HOURS} is the closest lead but still differs by "
            f"{best['median_difference_w_m2']:.2f} W m-2, against a threshold of "
            f"{MAX_MEDIAN_DIFFERENCE_W_M2}. The mirror is not reproducing the native field."
        )
        raise ValueError(msg)
    _LOG.info(
        "T+%d matches to %.2f W m-2, and is the closest of %s",
        EXPECTED_LEAD_HOURS,
        best["median_difference_w_m2"],
        list(SWEPT_LEAD_HOURS),
    )


def _window_for(*, era: str, days: int) -> tuple[dt.date, dt.date]:
    """Return the date window one era is sampled from.

    The pre-PS47 window ends the day before the upgrade and the post-PS47 window starts on it, so
    neither straddles the boundary the stratification exists to test.

    Args:
        era: `pre-PS47` or `post-PS47`.
        days: How long each window runs.

    Returns:
        The first and last date to request.
    """
    if era == "pre-PS47":
        last = PS47_OPERATIONAL - dt.timedelta(days=1)
        return last - dt.timedelta(days=days - 1), last
    return PS47_OPERATIONAL, PS47_OPERATIONAL + dt.timedelta(days=days - 1)


def main() -> int:
    """Sample both eras and both sky conditions, and assert which lead the archive holds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instants-per-stratum",
        type=int,
        default=2,
        help="How many instants to sample from each era-and-sky stratum.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=21,
        help="How long each era's sampling window runs, in days.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seeds the choice of instants.")
    arguments = parser.parse_args()

    sites = _pv_sites()
    instants: list[SampledInstant] = []
    served_by_era: dict[str, pl.DataFrame] = {}
    for era in ("pre-PS47", "post-PS47"):
        first, last = _window_for(era=era, days=arguments.window_days)
        served = _solar_geometry(
            frame=_renamed(
                frame=fetch_point_frame(
                    sites=sites,
                    variables=HOURLY_VARIABLES
                    + tuple(f"{name}{INSTANT_SUFFIX}" for name in HOURLY_VARIABLES),
                    models_parameter=OPEN_METEO_MODELS["ukv"].models_parameter,
                    first_date=f"{first:%Y-%m-%d}",
                    last_date=f"{last:%Y-%m-%d}",
                )
            ),
            sites=sites,
        )
        served_by_era[era] = served
        instants += _sample_instants(
            served=served,
            era=era,
            per_stratum=arguments.instants_per_stratum,
            seed=arguments.seed,
        )
    _LOG.info(
        "sampling %d instants, %d leads, %d fluxes: %d reads of about 1.5 MB each",
        len(instants),
        len(SWEPT_LEAD_HOURS),
        len(NATIVE_FILE_NAMES),
        len(instants) * len(SWEPT_LEAD_HOURS) * len(NATIVE_FILE_NAMES),
    )

    records: list[dict[str, object]] = []
    for instant in instants:
        _LOG.info("comparing %s (%s, %s)", instant.valid_time, instant.era, instant.sky)
        records += _compare_one(instant=instant, served=served_by_era[instant.era], sites=sites)
    if not records:
        msg = "no native file could be read, so nothing was verified"
        raise RuntimeError(msg)

    results = pl.DataFrame(records)
    _report(results=results)
    _assert_lead_is_as_expected(results=results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
