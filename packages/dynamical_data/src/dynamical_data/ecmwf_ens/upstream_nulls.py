"""Measuring upstream corruption on the raw NWP grid, before the H3 aggregation sees it.

The H3 aggregation renormalises each cell over the grid points that supplied a value, so a corrupt
grid point costs only its own share of its cell. That is what makes the stored cells robust, and it
is also why counting null *cells* is a poor proxy for how corrupt the feed was. This module counts
the nulls where they arrive.
"""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import xarray as xr

_LEAD_0: Final[np.timedelta64] = np.timedelta64(0, "ns")
"""Lead-0, as a unit-bearing timedelta.

The unit is required, not decoration: a bare ``np.timedelta64(0)`` raises a ``DeprecationWarning``
about the generic timedelta unit, and this repo turns warnings into errors. ``ns`` compares
correctly against a ``lead_time`` coordinate in any unit.
"""

_PER_VARIABLE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "variable": pl.String,
        "n_null": pl.Int64,
        "n_affected_slices": pl.Int64,
        "n_total": pl.Int64,
    }
)
"""Columns of :attr:`UpstreamNullRate.per_variable`.

Declared rather than inferred for the empty-``variables`` case, which no caller in this repo reaches
but a reusable package should survive: an inferred empty frame carries no columns at all, and the
properties below would raise ``ColumnNotFoundError`` inside a warning path."""


@dataclass(frozen=True)
class UpstreamNullRate:
    """How much of one ingested NWP run arrived null on the **raw grid**.

    This is the provider channel: the number to quote to Dynamical.org when asking whether their
    feed is degrading. It counts grid points on the 0.25° lat/lon box we downloaded, before any H3
    aggregation.

    Read it alongside, never instead of, :class:`contracts.weather_schemas.NwpQualityReport`, which
    counts null H3 *cells* and answers the different question of how much the model lost. The two
    are not comparable as rates: different units over different populations.

    See
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
    """

    per_variable: pl.DataFrame
    """One row per counted variable, with its ``n_null``, ``n_affected_slices`` and ``n_total``
    grid-point counts, sorted by variable name. Every scalar below is derived from it, so a
    breakdown and a total cannot disagree."""

    @property
    def n_null_nwp_grid_points(self) -> int:
        """Null grid points in the counted variables and steps."""
        return int(self.per_variable["n_null"].sum())

    @property
    def n_total_nwp_grid_points(self) -> int:
        """The denominator: counted variables × ensemble members × counted steps × grid points."""
        return int(self.per_variable["n_total"].sum())

    @property
    def n_affected_nwp_slices(self) -> int:
        """``(variable, ensemble_member, lead_time)`` slices carrying at least one null grid point.

        Separates "one bad slice" from "a hundred" at the same overall rate, which the fraction
        alone cannot.
        """
        return int(self.per_variable["n_affected_slices"].sum())

    @property
    def affected_nwp_variables(self) -> tuple[str, ...]:
        """The counted variables carrying at least one null grid point, sorted."""
        return tuple(self.per_variable.filter(pl.col("n_null") > 0)["variable"])

    @property
    def null_nwp_grid_point_fraction(self) -> float:
        """Null grid points as a fraction of those counted; ``0.0`` when none were counted.

        A run with no step left to count has nothing to measure, and a warning path must not
        raise
        ([rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
        """
        if self.n_total_nwp_grid_points == 0:
            return 0.0
        return self.n_null_nwp_grid_points / self.n_total_nwp_grid_points

    @property
    def is_healthy(self) -> bool:
        """True when no counted grid point arrived null."""
        return self.n_null_nwp_grid_points == 0


def assess_upstream_grid_point_nulls(
    ds: xr.Dataset, variables: Collection[str], exclude_lead_0: bool
) -> UpstreamNullRate:
    """Count nulls on the raw NWP grid of one downloaded run.

    Pure and Dagster-free (unit-testable in isolation); the ``ecmwf_ens`` asset calls it twice, once
    per null population, and publishes each result on its own WARN check.

    Args:
        ds: One downloaded ECMWF ENS run, as returned by
            :func:`dynamical_data.ecmwf_ens.download.download_ecmwf_ens_data` — dimensions
            ``(lead_time, ensemble_member, latitude, longitude)``, with ``init_time`` already
            reduced to a scalar coordinate.
        variables: The variables to count over, named as ``ds`` names them rather than as the
            ``Nwp`` contract does — the two differ on wind, so
            :data:`dynamical_data.ecmwf_ens.download.ECMWF_ENS_INSTANTANEOUS_VARS` exists to be
            passed here. Their nulls must share one meaning, because a rate pooled over variables
            with opposite null semantics measures nothing: the asset passes the de-accumulated
            variables, whose nulls are known upstream corruption, and the instantaneous ones, whose
            nulls are anomalous, on separate calls.
        exclude_lead_0: Skip the lead-0 step. True for the de-accumulated variables, which are null
            there by design, so counting it would report every healthy run as corrupt. False for
            the instantaneous ones, where lead-0 is an ordinary step and a null in it means what a
            null in any other step means.
    """
    # Selected per variable rather than once on `ds`: this runs inside `ecmwf_ens`, whose ECMWF
    # concurrency pool exists because the download is memory-intensive, and slicing the whole
    # dataset would copy all thirteen downloaded variables to read three.
    beyond_lead_0 = ds.lead_time > _LEAD_0
    rows = []
    for name in sorted(variables):
        values = ds[name].isel(lead_time=beyond_lead_0) if exclude_lead_0 else ds[name]
        nulls_per_slice = values.isnull().sum(dim=["latitude", "longitude"])
        rows.append(
            {
                "variable": name,
                "n_null": int(nulls_per_slice.sum()),
                "n_affected_slices": int((nulls_per_slice > 0).sum()),
                "n_total": values.size,
            }
        )
    return UpstreamNullRate(per_variable=pl.DataFrame(rows, schema=_PER_VARIABLE_SCHEMA))
