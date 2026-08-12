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
import xarray as xr

_LEAD_0: Final[np.timedelta64] = np.timedelta64(0, "ns")
"""Lead-0, as a unit-bearing timedelta.

The unit is required, not decoration: a bare ``np.timedelta64(0)`` raises a ``DeprecationWarning``
about the generic timedelta unit, and this repo turns warnings into errors. ``ns`` compares
correctly against a ``lead_time`` coordinate in any unit.
"""


@dataclass(frozen=True)
class UpstreamNullRate:
    """How much of one ingested NWP run arrived null on the **raw grid**, beyond lead-0.

    This is the provider channel: the number to quote to Dynamical.org when asking whether their
    feed is degrading. It counts grid points on the 0.25° lat/lon box we downloaded, before any H3
    aggregation, so it is free of our H3 resolution, our grid spacing and our aggregation policy —
    all of which move a cell-level count without anything upstream having changed.

    Read it alongside, never instead of, :class:`contracts.weather_schemas.NwpQualityReport`, which
    counts null H3 *cells* and answers the different question of how much the model lost. The two
    are not comparable as rates: different units over different populations. What they do share is
    the slice filter — both ignore lead-0, where the de-accumulated variables are null by design.

    See
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
    """

    n_null_nwp_grid_points: int
    """Null grid points in the counted variables, beyond lead-0."""

    n_total_nwp_grid_points: int
    """The denominator: counted variables × ensemble members × steps beyond lead-0 × grid points."""

    n_affected_nwp_slices: int
    """``(variable, ensemble_member, lead_time)`` slices carrying at least one null grid point.

    Separates "one bad slice" from "a hundred" at the same overall rate, which the fraction alone
    cannot. Whether an affected slice is *wholly* null is answered better by ``NwpQualityReport``'s
    ``n_whole_null_h3_slices``: a slice null at every grid point leaves every cell it feeds with
    zero contributing weight, so it reaches the stored cells intact.
    """

    affected_nwp_variables: tuple[str, ...]
    """The counted variables carrying at least one null grid point, sorted."""

    @property
    def null_nwp_grid_point_fraction(self) -> float:
        """Null grid points as a fraction of those counted; ``0.0`` when none were counted.

        A run carrying no step beyond lead-0 has nothing to measure. Returning ``0.0`` rather than
        dividing is what keeps this a warning path: a partial upstream publication that lands only
        lead-0 is absent input, and
        [rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
        forbids a warning path failing the run it is warning about.
        """
        if self.n_total_nwp_grid_points == 0:
            return 0.0
        return self.n_null_nwp_grid_points / self.n_total_nwp_grid_points

    @property
    def is_healthy(self) -> bool:
        """True when no counted grid point arrived null beyond lead-0."""
        return self.n_null_nwp_grid_points == 0


def assess_upstream_grid_point_nulls(
    ds: xr.Dataset, variables: Collection[str]
) -> UpstreamNullRate:
    """Count nulls on the raw NWP grid of one downloaded run, beyond lead-0.

    Pure and Dagster-free (unit-testable in isolation); the ``ecmwf_ens`` asset publishes the result
    on the ``nwp_has_no_unexpected_nulls`` WARN check.

    Args:
        ds: One downloaded ECMWF ENS run, as returned by
            :func:`dynamical_data.ecmwf_ens.download.download_ecmwf_ens_data` — dimensions
            ``(lead_time, ensemble_member, latitude, longitude)``, with ``init_time`` already
            reduced to a scalar coordinate.
        variables: The weather variables to count over. The asset passes
            :attr:`contracts.weather_schemas.Nwp.deaccumulated_var_names`, whose nulls are known
            upstream corruption. Pooling variables with different null semantics into one rate
            would make that rate meaningless, so a caller wanting the instantaneous variables asks
            for them separately rather than adding them here.
    """
    beyond_lead_0 = ds.sel(lead_time=ds.lead_time > _LEAD_0)
    n_null = 0
    n_total = 0
    n_affected_slices = 0
    affected_variables: list[str] = []
    for name in sorted(variables):
        nulls_per_slice = beyond_lead_0[name].isnull().sum(dim=["latitude", "longitude"])
        variable_n_null = int(nulls_per_slice.sum())
        n_null += variable_n_null
        n_total += beyond_lead_0[name].size
        n_affected_slices += int((nulls_per_slice > 0).sum())
        if variable_n_null:
            affected_variables.append(name)
    return UpstreamNullRate(
        n_null_nwp_grid_points=n_null,
        n_total_nwp_grid_points=n_total,
        n_affected_nwp_slices=n_affected_slices,
        affected_nwp_variables=tuple(affected_variables),
    )
