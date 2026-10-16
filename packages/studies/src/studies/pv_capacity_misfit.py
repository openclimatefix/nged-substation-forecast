"""The cross-arm misfit test for embedded-solar capacity, and the capacity and bound it gives.

A fit of embedded-solar capacity to a Primary substation's demand returns, for each of six
least-squares fits (the detector arms), a capacity in megawatts. A seventh fit, the daily fit, is
carried along and is not a detector arm. Together the seven numbers are an *arm vector*. For pure
solar output of capacity `C`, arm `a` returns about `R[a] * C`, where `R[a]` is that arm's recovery
ratio, measured on solar-farm output added to the demand. A cloud-driven demand load breaks the
proportionality, and the relative misfit `T` measures by how much.

Every function here reads arm vectors that an earlier run already fitted, so the module does no
fitting. The arm vectors, and the experiment that fitted them, are described on the
[embedded-solar capacity study
page](https://openclimatefix.github.io/nged-substation-forecast/studies/embedded-pv-capacity/).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("M1", "M1L", "M2", "M2h", "M1_s", "M2h_s")
"""The six detector arms, in the order of every arm vector."""
DAILY_FIT: Final[str] = "M1d"
"""The daily fit, carried after the six arms and not part of the detector."""
ALL_FITS: Final[tuple[str, ...]] = (*ARMS, DAILY_FIT)
"""Every column of an arm vector."""
N_ARMS: Final[int] = len(ARMS)
DAILY_FIT_INDEX: Final[int] = ALL_FITS.index(DAILY_FIT)
AMPLITUDES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 1.0)
"""Sizes of the injected confound load, each as a share of the true capacity, measured as the
load's leakage into arm M1."""
POWER_AMPLITUDE: Final[float] = 0.25
"""The amplitude at which the detector's power is reported."""
POWER_AMPLITUDE_INDEX: Final[int] = AMPLITUDES.index(POWER_AMPLITUDE)
MIN_POWER: Final[float] = 0.8
"""A shape is detected at an amplitude when the test flags at least this share of draws."""
FLAG_QUANTILE: Final[float] = 0.95
"""The share of the no-confound misfit distribution that stays below the flag threshold."""
MIN_LEAKAGE: Final[float] = 0.5
"""The smallest M1 leakage of a shape that can be rescaled to a stated amplitude."""


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
    """The arm vectors of one (Primary, solar farm, season-year) unit.

    Attributes:
        site: Primary label.
        farm: Anonymised solar farm label.
        year: Season-year, named by the calendar year of its September.
        pv: Arm vector of the farm's normalised output alone (true capacity 1 MW).
        shapes: Arm vector of each confound shape alone, on the unit's rows.
    """

    site: str
    farm: str
    year: int
    pv: np.ndarray
    shapes: dict[str, np.ndarray]


@dataclass
class Detector:
    """Recovery ratios, real capacities, and flag thresholds, with each leave-farm-out cached.

    Attributes:
        units: Every unit.
        real: The real arm vector of each (Primary, season-year).
        noise: The bootstrap noise vectors of each (Primary, season-year): the resampled arm
            vectors minus their mean, shape (draws, 7).
    """

    units: list[Unit]
    real: dict[tuple[str, int], np.ndarray]
    noise: dict[tuple[str, int], np.ndarray]
    _ratios: dict[tuple[str, frozenset[str]], np.ndarray | None] = field(default_factory=dict)
    _taus: dict[tuple[str, int, frozenset[str]], float] = field(default_factory=dict)

    def ratios(self, *, site: str, excluded: frozenset[str]) -> np.ndarray | None:
        """Return `R` at a Primary: the median over units of the farms outside `excluded`."""
        key = (site, excluded)
        if key not in self._ratios:
            kept = [u.pv for u in self.units if u.site == site and u.farm not in excluded]
            self._ratios[key] = np.median(np.array(kept), axis=0) if kept else None
        return self._ratios[key]

    def capacity(self, *, site: str, year: int, excluded: frozenset[str]) -> float:
        """Return the real no-confound capacity `C` of a Primary-year, `R` without `excluded`."""
        ratios = self.ratios(site=site, excluded=excluded)
        assert ratios is not None, f"No unit of {site} is outside {sorted(excluded)}"
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
                float(np.quantile(np.concatenate(values), FLAG_QUANTILE))
                if values
                else float("nan")
            )
        return self._taus[key]


def scenario(
    *,
    scale: float,
    pv: np.ndarray,
    leakage: np.ndarray | None,
    amplitude: float,
    noise: np.ndarray,
    ratios: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the flags and the `C / truth` of the noise draws for one injected scenario.

    The truth is `scale`. The confound adds `amplitude * scale` of M1 leakage, in the direction
    of the shape's arm vector `leakage`.
    """
    vector = pv[:N_ARMS].copy()
    if leakage is not None:
        assert abs(leakage[0]) > MIN_LEAKAGE, f"M1 leakage of the shape is {leakage[0]}"
        vector = vector + amplitude * leakage[:N_ARMS] / abs(leakage[0])
    capacity, misfit = capacity_and_misfit(
        vectors=scale * vector[None, :] + noise[:, :N_ARMS], ratios=ratios
    )
    return misfit > tau, capacity / scale


@dataclass(frozen=True)
class ShapeBound:
    """The detector's power and the undetected error of one confound shape at one Primary-year.

    Attributes:
        power_by_amplitude: Share of draws flagged at each amplitude in `AMPLITUDES`.
        median_error_by_amplitude: Median `|C / truth - 1|` over the undetected draws at each
            amplitude (`nan` when every draw was flagged).
        bound: The median error at the largest amplitude whose power is under `MIN_POWER`, or 0
            when the power reaches `MIN_POWER` at every amplitude.
        bound_amplitude: That amplitude, or `None` when `bound` is 0 for lack of a weak amplitude.
    """

    power_by_amplitude: tuple[float, ...]
    median_error_by_amplitude: tuple[float, ...]
    bound: float
    bound_amplitude: float | None


def shape_bound(*, detector: Detector, site: str, year: int, leakage: np.ndarray) -> ShapeBound:
    """Return the power and bound of one shape (its real arm vector `leakage`) at a Primary-year.

    Every unit of the Primary stands in for the real solar output in turn, with its own farm left
    out of the recovery ratios and the threshold, as in the calibration of the test.

    Args:
        detector: The detector holding the units, real vectors, and noise.
        site: The Primary label.
        year: The season-year.
        leakage: The shape's arm vector on the Primary-year's own rows.

    Returns:
        The power and median error at each amplitude, and the bound.
    """
    units = [u for u in detector.units if u.site == site]
    power, error = [], []
    for amplitude in AMPLITUDES:
        flags, errors = [], []
        for unit in units:
            left_out = frozenset({unit.farm})
            unit_ratios = detector.ratios(site=site, excluded=left_out)
            if unit_ratios is None:
                continue
            flagged, ratio = scenario(
                scale=detector.capacity(site=site, year=year, excluded=left_out),
                pv=unit.pv,
                leakage=leakage,
                amplitude=amplitude,
                noise=detector.noise[site, year],
                ratios=unit_ratios,
                tau=detector.tau(site=site, year=year, excluded=left_out),
            )
            flags.append(flagged)
            errors.append(np.abs(ratio[~flagged] - 1.0))
        kept = np.concatenate(errors)
        power.append(float(np.mean(np.concatenate(flags))))
        error.append(float(np.median(kept)) if kept.size else float("nan"))
    weak = [i for i, p in enumerate(power) if p < MIN_POWER]
    if not weak:
        return ShapeBound(tuple(power), tuple(error), 0.0, None)
    last = weak[-1]
    return ShapeBound(tuple(power), tuple(error), max(error[last], 0.0), AMPLITUDES[last])


def block_bootstrap_capacities(
    *, draws: np.ndarray, ratios: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the capacity `C` and the daily-fit capacity `M1d / R` of each bootstrap draw.

    Args:
        draws: The arm vectors fitted to the resampled data, one per row, seven columns.
        ratios: The recovery ratios `R` of the Primary, seven values.

    Returns:
        The cross-arm `C` per draw, and the daily fit divided by its recovery ratio per draw.
    """
    capacity = capacity_and_misfit(vectors=draws, ratios=ratios)[0]
    return capacity, draws[:, DAILY_FIT_INDEX] / ratios[DAILY_FIT_INDEX]


def undetected_bias(
    *, ratios: np.ndarray, pv: np.ndarray, leakage: np.ndarray, amplitude: float
) -> tuple[float, float]:
    """Return the relative bias of `C` and of the daily fit for a noise-free undetected load.

    The load has `amplitude` times the truth as M1 leakage, in the direction of the shape's own M1
    leakage: a load that rises with sunshine has negative M1 leakage and biases the capacity low
    (the truth is 1 MW, so `pv` is its arm vector). The daily fit's bias is its leaked capacity
    over its recovery ratio.

    Args:
        ratios: The recovery ratios `R` of the Primary, seven values.
        pv: The arm vector of the truth: normalised solar output of 1 MW.
        leakage: The load's arm vector; its sign gives the direction of the bias.
        amplitude: The load's M1 leakage, as a multiple of the true capacity.

    Returns:
        `(bias of C, bias of M1d / R)`, each as `estimate / truth - 1`.
    """
    assert abs(leakage[0]) > MIN_LEAKAGE, f"M1 leakage of the shape is {leakage[0]}"
    scale = amplitude / abs(leakage[0])
    vector = pv + scale * leakage
    cross_arm = float(capacity_and_misfit(vectors=vector[None, :], ratios=ratios)[0][0])
    daily = float(vector[DAILY_FIT_INDEX] / ratios[DAILY_FIT_INDEX])
    return cross_arm - 1.0, daily - 1.0


def season_year_of(*, timestamps: pl.Expr) -> pl.Expr:
    """Return the season-year (September to August, named by its September's year) of UTC times.

    The year is taken on local (Europe/London) time, so a half-hour at the turn of a month falls
    in the season-year a reader would expect.

    Args:
        timestamps: An expression of UTC datetimes.

    Returns:
        An expression of integer season-years.
    """
    local = timestamps.dt.convert_time_zone("Europe/London")
    return local.dt.year() - (local.dt.month() < 9).cast(pl.Int32)


def load_detector(
    *,
    unit_vectors_path: Path,
    real_vectors_path: Path,
    bootstrap_path: Path,
    sites: Sequence[str] | None = None,
) -> Detector:
    """Build the `Detector` from the stored unit, real, and bootstrap arm-vector tables.

    Args:
        unit_vectors_path: Parquet of the unit arm vectors (columns `site`, `farm`, `year`, `kind`
            and the seven fits), with `kind == "pv"` for solar output alone.
        real_vectors_path: Parquet of the real Primary-years' vectors (`kind == "real"`) and of the
            confound shapes on the real rows.
        bootstrap_path: Parquet of the bootstrap draws (`site`, `year`, `draw` and the seven fits).
        sites: Restrict to these Primaries; all when `None`.

    Returns:
        The detector.
    """
    keep = pl.col("site").is_in(list(sites)) if sites is not None else pl.lit(True)
    table = pl.read_parquet(unit_vectors_path).filter(keep)
    units = []
    for row in table.filter(pl.col("kind") == "pv").iter_rows(named=True):
        shapes = table.filter(
            (pl.col("site") == row["site"])
            & (pl.col("farm") == row["farm"])
            & (pl.col("year") == row["year"])
            & (pl.col("kind") != "pv")
        )
        units.append(
            Unit(
                site=row["site"],
                farm=row["farm"],
                year=row["year"],
                pv=np.array([row[a] for a in ALL_FITS]),
                shapes={
                    r["kind"]: np.array([r[a] for a in ALL_FITS])
                    for r in shapes.iter_rows(named=True)
                },
            )
        )
    real_table = pl.read_parquet(real_vectors_path).filter(keep)
    real = {
        (r["site"], r["year"]): np.array([r[a] for a in ALL_FITS])
        for r in real_table.filter(pl.col("kind") == "real").iter_rows(named=True)
    }
    noise = {}
    for (site, year), draws in (
        pl.read_parquet(bootstrap_path).filter(keep).group_by("site", "year", maintain_order=True)
    ):
        values = draws.sort("draw").select(ALL_FITS).to_numpy()
        noise[site, year] = values - values.mean(axis=0)
    return Detector(units=units, real=real, noise=noise)
