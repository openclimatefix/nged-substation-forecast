"""The cross-arm misfit test: capacity under no confound, misfit, ratios, and thresholds.

An arm vector holds the capacity each of the six detector arms fits to one series, followed by the
daily fit. For injected PV of true capacity `C`, arm `a` returns about `R_a * C`, where `R_a` is
that arm's recovery ratio. A series with a cloud-driven load breaks the proportionality, and the
misfit `T` measures by how much.
"""

from dataclasses import dataclass, field

import e00040_common as common
import numpy as np

N_ARMS = len(common.ARMS)


def capacity_and_misfit(
    *, vectors: np.ndarray, ratios: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the no-confound capacity `C` and the relative misfit `T` of each arm vector.

    Args:
        vectors: Arm vectors, one per row; only the first six columns are used.
        ratios: The recovery ratios `R`, at least six values.

    Returns:
        `C = (R . c) / (R . R)` and `T = sqrt(mean((c - R * C)^2)) / C` per row. `T` is infinite
        when `C` is not positive.
    """
    r = ratios[:N_ARMS]
    c = np.atleast_2d(vectors)[:, :N_ARMS]
    capacity = c @ r / (r @ r)
    misfit = np.sqrt(np.mean((c - np.outer(capacity, r)) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(capacity > 0.0, misfit / capacity, np.inf)
    return capacity, relative


@dataclass(frozen=True)
class Unit:
    """The arm vectors of one (Primary, farm, season-year) unit.

    Attributes:
        site: Primary label.
        farm: Solar farm label.
        year: Season-year.
        pv: Arm vector of the normalised farm output alone (true capacity 1 MW).
        shapes: Arm vector of each confound shape alone, on the unit's shared rows.
    """

    site: str
    farm: str
    year: int
    pv: np.ndarray
    shapes: dict[str, np.ndarray]


@dataclass
class Detector:
    """Recovery ratios, real capacities, and thresholds, with every leave-farm-out cached.

    Attributes:
        units: Every unit.
        real: The real arm vector of each (Primary, season-year).
        noise: The 200 bootstrap noise vectors of each (Primary, season-year), shape (200, 7).
    """

    units: list[Unit]
    real: dict[tuple[str, int], np.ndarray]
    noise: dict[tuple[str, int], np.ndarray]
    _ratios: dict[tuple[str, frozenset[str]], np.ndarray | None] = field(default_factory=dict)
    _taus: dict[tuple[str, int, frozenset[str]], float] = field(default_factory=dict)

    def ratios(self, *, site: str, excluded: frozenset[str]) -> np.ndarray | None:
        """Return `R` at a Primary, the median over units of farms outside `excluded`."""
        key = (site, excluded)
        if key not in self._ratios:
            kept = [u.pv for u in self.units if u.site == site and u.farm not in excluded]
            self._ratios[key] = np.median(np.array(kept), axis=0) if kept else None
        return self._ratios[key]

    def capacity(self, *, site: str, year: int, excluded: frozenset[str]) -> float:
        """Return the real no-confound capacity `C` of a Primary-year, `R` leaving out farms."""
        ratios = self.ratios(site=site, excluded=excluded)
        assert ratios is not None
        return float(
            capacity_and_misfit(vectors=self.real[site, year][None, :], ratios=ratios)[0][0]
        )

    def tau(self, *, site: str, year: int, excluded: frozenset[str]) -> float:
        """Return the flag threshold of a Primary-year, built without the farms in `excluded`.

        The null is every unit of a farm outside `excluded` at the Primary (any season-year),
        scaled to the Primary-year's real capacity, plus each noise draw. The `R` of each null
        unit leaves out its own farm as well.
        """
        key = (site, year, excluded)
        if key not in self._taus:
            scale = self.capacity(site=site, year=year, excluded=excluded)
            noise = self.noise[site, year][:, :N_ARMS]
            values = []
            for unit in self.units:
                if unit.site != site or unit.farm in excluded:
                    continue
                ratios = self.ratios(site=site, excluded=excluded | {unit.farm})
                if ratios is None:
                    continue
                vectors = scale * unit.pv[None, :N_ARMS] + noise
                values.append(capacity_and_misfit(vectors=vectors, ratios=ratios)[1])
            self._taus[key] = (
                float(np.quantile(np.concatenate(values), common.FLAG_QUANTILE))
                if values
                else float("nan")
            )
        return self._taus[key]
