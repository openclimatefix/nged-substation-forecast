"""Measuring upstream corruption on the raw NWP grid, before the H3 aggregation sees it.

The H3 aggregation renormalises each cell over the grid points that supplied a value, so a
corrupt grid point costs only its own share of its cell. That renormalisation is what makes the
stored cells robust. That renormalisation is also why counting null *cells* is a poor proxy for
how corrupt the feed was. This module counts the nulls where they arrive, before that
renormalisation absorbs most of them. The aggregation mechanics are documented, along with the
measurements behind the claim that a corrupt grid point takes only its own share out of its cell,
at
<https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#spatial-aggregation-is-where-a-grid-points-null-is-resolved>.
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
"""Columns of `UpstreamNullRate.per_variable`.

Declared rather than inferred for the empty-``variables`` case. No caller in this repo reaches that
case, but a reusable package should survive it. An inferred empty frame carries no columns at all,
and the properties below would raise ``ColumnNotFoundError`` inside a warning path."""


@dataclass(frozen=True)
class UpstreamNullRate:
    """How much of one ingested NWP run arrived null on the **raw grid**.

    This is the provider channel: the number to quote to Dynamical.org when asking whether their
    feed is degrading. It counts grid points on the 0.25° lat/lon box we downloaded, before any
    H3 aggregation.

    Read it alongside `contracts.weather_schemas.NwpQualityReport`, never instead of that report.
    `NwpQualityReport` counts null H3 *cells* and answers the different question of how much the
    power-forecasting model lost. The two are not comparable as rates: different units over
    different populations.

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
        """The counted variables carrying at least one null grid point.

        In ``per_variable``'s row order, which is sorted by variable name because
        `assess_upstream_grid_point_nulls` builds it that way. ``filter`` preserves row order
        rather than imposing a row order.
        """
        return tuple(self.per_variable.filter(pl.col("n_null") > 0)["variable"])

    @property
    def null_nwp_grid_point_fraction(self) -> float:
        """Null grid points as a fraction of those counted; ``0.0`` when none were counted.

        A run with no step left to count has nothing to measure, and a warning path must not
        raise ([rule
        7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
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

    Pure and Dagster-free (unit-testable in isolation). The ``ecmwf_ens`` asset calls it twice, once
    per null population, and publishes each result on its own WARN check.

    Args:
        ds: One downloaded ECMWF ENS run, as returned by
            `dynamical_data.ecmwf_ens.download.download_ecmwf_ens_data` — dimensions
            ``(lead_time, ensemble_member, latitude, longitude)``, with ``init_time`` already
            reduced to a scalar coordinate.
        variables: The variables to count over, named as ``ds`` names them rather than as the
            ``Nwp`` contract does — the two differ on wind, so
            `dynamical_data.ecmwf_ens.download.ECMWF_ENS_INSTANTANEOUS_VARS` exists to be
            passed here. Their nulls must share one meaning, because it measures nothing to pool a
            rate over variables with opposite null semantics. The asset therefore makes two separate
            calls. One call passes the de-accumulated variables, which Dynamical.org differences
            from ECMWF's running totals into rates (``W m-2`` for the radiation fluxes, ``kg m-2
            s-1`` for precipitation), so their nulls are known upstream corruption. The other call
            passes the instantaneous variables, whose nulls are anomalous.
        exclude_lead_0: Skip the lead-0 step. True for the de-accumulated variables, which are null
            there by design, so counting it would report every healthy run as corrupt. False for
            the instantaneous ones, where lead-0 is an ordinary step and a null in it means what a
            null in any other step means.

    Returns:
        An `UpstreamNullRate` whose `per_variable` frame holds one row per counted variable, sorted
        by variable name. Each row gives that variable's null grid-point count (`n_null`), the count
        of (ensemble_member, lead_time) slices holding at least one null (`n_affected_slices`), and
        the total number of grid points counted (`n_total`).
    """
    # Selected per variable rather than once on `ds`. This code runs inside `ecmwf_ens` while the
    # whole downloaded run is still held in memory. Slicing the whole dataset would copy all 13
    # downloaded variables to read either the three de-accumulated variables or the nine
    # instantaneous variables, depending on the call. The 13th is
    # categorical_precipitation_type_surface, which neither call counts.
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
