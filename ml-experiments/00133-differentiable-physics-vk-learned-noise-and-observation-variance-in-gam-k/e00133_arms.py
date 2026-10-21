"""The arms of experiment 133 and the pieces each one is built from.

| Arm | Filter | Spread |
|---|---|---|
| `fix` | experiment 111's stored settings | refitted, factors tuned |
| `grid` | the cell chosen per substation (`e00133_select.py`) | the same |
| `pool` | one shared cell (`e00133_select.py`) | the same |
| `vk0` | `grid`'s cell with the online observation variance | the same |
| `vk1` | the same filter | scaled by the online variance's tail factor |
| `trail` | `grid`'s cell | scaled by the trailing innovation root mean square |

Every arm drives the same stored monthly fits of experiment 111's `gamk` with the same calibration
and solar shrink, so only the filter's trace and the spread differ.
"""

import json
import pickle
from dataclasses import dataclass
from typing import Final

import e00133_common as common
import e00133_ens as ens
import e00133_filters as filters
import numpy as np
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, usable_cutoff_positions
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit

ARM_NAMES: Final[tuple[str, ...]] = ("fix", "grid", "pool", "vk0", "vk1", "trail")
TAIL_ARMS: Final[dict[str, str]] = {"vk1": "variance", "trail": "innovation"}
"""The arms whose spread is scaled, and which statistic scales it."""
FACTOR_CLIP: Final[tuple[float, float]] = (0.8, 1.5)
MIN_RHO_EXPONENT: Final[int] = 9
RHO_GRID: Final[tuple[float, ...]] = tuple(float(np.exp(-x)) for x in (9, 7, 5))
"""The variance random walk's variance per half-hour, chosen on the tuning months."""


@dataclass(frozen=True)
class SiteContext:
    """A substation's stored inputs: the ensemble setup, the monthly fits, and the proxy stream.

    Attributes:
        site: The substation's label.
        cache: The feature cache.
        setup: The ensemble setup, with experiment 111's calibration and solar shrink.
        fits: Experiment 111's monthly fits of `gamk`.
        stream: Observed minus the proxy-driven fits' forecast on the grid, in points.
    """

    site: str
    cache: FeatureCache
    setup: ens.SiteSetup
    fits: list[GamFit]
    stream: np.ndarray


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink, filter values, spread factors)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, without its solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def load_site(*, site: str, cache: FeatureCache | None = None) -> SiteContext:
    """Load a substation's setup, fits, and proxy-driven residual stream.

    Args:
        site: The substation's label.
        cache: The feature cache, loaded if not given.

    Returns:
        The context.
    """
    cache = cache if cache is not None else common.load_cache()
    calibration = load_calibration(site=site).with_solar_shrink(
        solar_shrink=read_tuned()["solar_shrink"]
    )
    setup = ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)
    fits = pickle.loads((common.FITS_DIR / "gamk" / f"{site}.pkl").read_bytes())
    stream = proxy_residual_stream(
        cache=proxy_cache(cache=cache, context=setup.context), fits=fits, site=site
    )
    return SiteContext(site=site, cache=cache, setup=setup, fits=fits, stream=stream)


def fix_hyper(*, site: str) -> np.ndarray:
    """Return experiment 111's four unconstrained values at a site."""
    return np.array(read_tuned()["filter_hyper"]["gamk"][site])


def settings_of_hyper(*, hyper: np.ndarray, rho_a: float = 0.0) -> filters.FilterSettings:
    """Return the filter setting of four unconstrained values.

    Args:
        hyper: Log `q`, logit `phi`, log `sigma_z`, and log `sigma_eps`.
        rho_a: The online variance's random-walk variance per half-hour, 0 for a constant variance.

    Returns:
        The setting.
    """
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=rho_a,
    )


def tuning_runs(*, context: SiteContext) -> list[int]:
    """Return the tuning-month runs that have a fit at their origin."""
    first = context.fits[0].cut
    from datetime import timedelta

    limit = first + timedelta(hours=DEFAULT_DELAY_HOURS)
    return [
        i for i in context.setup.indices(periods={"tuning"}) if context.setup.origins[i] >= limit
    ]


def tail_reference(*, run: filters.StreamRun, kind: str, frame_positions: dict) -> np.ndarray:
    """Return the median of the tail statistic per band, over the tuning-month cases.

    Args:
        run: The filter's run over the stream.
        kind: `variance` for the online variance, `innovation` for the trailing innovations.
        frame_positions: Keys `cutoff`, `target`, and `band` of the tuning-month rows.

    Returns:
        Shape (4,): each band's median statistic.
    """
    statistic = _statistic(run=run, kind=kind)
    values = filters.value_at_cutoffs(
        positions=run.trace.positions,
        values=statistic,
        cutoff=frame_positions["cutoff"],
        band=frame_positions["band"],
    )
    return np.array(
        [np.nanmedian(values[frame_positions["band"] == b]) for b in range(len(statistic))]
    )


def _statistic(*, run: filters.StreamRun, kind: str) -> tuple[np.ndarray, ...]:
    if kind == "variance":
        return tuple(np.exp(a) for a in run.log_variance)
    return tuple(r**2 for r in run.innovation_rms)


def tail_factors(
    *, run: filters.StreamRun, kind: str, reference: np.ndarray, rows: dict
) -> np.ndarray:
    """Return the spread factor of each row: `clip(sqrt(statistic / reference), 0.8, 1.5)`.

    Args:
        run: The filter's run over the stream.
        kind: `variance` or `innovation`.
        reference: Each band's median statistic over the tuning-month rows.
        rows: Keys `cutoff`, `target`, and `band`.

    Returns:
        One factor per row, 1 where the row's band has no reading before the cutoff.
    """
    values = filters.value_at_cutoffs(
        positions=run.trace.positions,
        values=_statistic(run=run, kind=kind),
        cutoff=rows["cutoff"],
        band=rows["band"],
    )
    factor = np.sqrt(values / reference[rows["band"]])
    factor = np.clip(factor, *FACTOR_CLIP)
    return np.where(np.isfinite(factor), factor, 1.0)


def rows_of_frame(*, context: SiteContext, frame) -> dict:  # noqa: ANN001
    """Return the cutoff, target position, and band of each row of a forecast frame."""
    cache = context.cache
    origins = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=origins, delay_hours=DEFAULT_DELAY_HOURS
    )
    target = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return {
        "cutoff": cutoff,
        "target": target,
        "band": cache.calendar.band[target].astype(np.int64),
    }
