"""The residual panel, the scaled residuals, and the leave-one-out factor of experiment 108.

Every contributor's out-of-sample residuals are the frozen fits' (`e00108_fits.py`) residuals,
stitched across the 1 July refits with `studies.dp_assimilation.stitch_out_of_sample_residuals`, so
no residual is in-sample. No switching mask enters a stream, a scale, or a loading.

For a scored site `s`, the scaled residual of a contributor `j` is `u_j(t) = r_j(t) / sigma_{j,b}`,
where `sigma_{j,b}` is 1.4826 times the median absolute deviation of `j`'s in-sample residuals in
hour band `b` over the 12 months before the fit that supplies `t`. The factor `F_{-s}(t)` is the
median of `u_j(t)` over the contributors `j != s` with a reading at `t`, and it is missing where
fewer than `MIN_CONTRIBUTORS` have one. A site contributes at `t` only if it has a fit covering `t`
(at least 6 months of readings before that fit's cut).
"""

import warnings
from dataclasses import dataclass
from datetime import datetime

import e00108_common as common
import e00108_fits as fits
import numpy as np
from studies.dp_assimilation import stitch_out_of_sample_residuals
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SiteFit:
    """One contributor's stored A0 fit at one cut.

    Attributes:
        site: The substation's label.
        year: The year of the 1 July cut.
        cut_position: The grid position of the last half-hour ending at or before the cut.
        index: The grid rows of the scored year.
        mean: The mean forecast at `index`, in points.
        in_sample_index: The grid rows of the 12 months before the cut.
        in_sample_residual: Observed minus fitted at `in_sample_index`, in points.
        sigma: The residual scale per hour band, in points.
    """

    site: str
    year: int
    cut_position: int
    index: np.ndarray
    mean: np.ndarray
    in_sample_index: np.ndarray
    in_sample_residual: np.ndarray
    sigma: np.ndarray


def _band_sigma(*, residual: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Return 1.4826 times the median absolute deviation of the residuals in each hour band."""
    sigma = np.empty(N_BANDS)
    for band_index in range(N_BANDS):
        values = residual[band == band_index]
        sigma[band_index] = MAD_TO_SIGMA * np.median(np.abs(values - np.median(values)))
    return sigma


def load_fits(*, cache: FeatureCache) -> dict[str, dict[int, SiteFit]]:
    """Read every stored fit, keyed by site and year of the cut."""
    loaded: dict[str, dict[int, SiteFit]] = {}
    for site in common.SCORED_SITES:
        for year in common.CUT_YEARS:
            path = fits.fit_path(site=site, year=year)
            if not path.exists():
                continue
            with np.load(path) as stored:
                in_sample = stored["in_sample_index"]
                residual = stored["in_sample_residual"]
                loaded.setdefault(site, {})[year] = SiteFit(
                    site=site,
                    year=year,
                    cut_position=common.grid_position_of_cut(
                        cache=cache, cut=common.cut_of(year=year)
                    ),
                    index=stored["index"],
                    mean=stored["mean"],
                    in_sample_index=in_sample,
                    in_sample_residual=residual,
                    sigma=_band_sigma(residual=residual, band=cache.calendar.band[in_sample]),
                )
    return loaded


def residual_on_grid(*, cache: FeatureCache, fit: SiteFit) -> np.ndarray:
    """Return one fit's out-of-sample residuals on the grid, `NaN` elsewhere."""
    stream = np.full(len(cache.end_times), np.nan)
    stream[fit.index] = cache.sites[fit.site].y[fit.index] - fit.mean
    return stream


def stitched_residuals(
    *, cache: FeatureCache, site_fits: dict[int, SiteFit], first_year: int
) -> np.ndarray:
    """Join a site's out-of-sample residuals from the fit of `first_year` onwards."""
    years = sorted(year for year in site_fits if year >= first_year)
    return stitch_out_of_sample_residuals(
        cut_positions=[site_fits[year].cut_position for year in years],
        residuals=[residual_on_grid(cache=cache, fit=site_fits[year]) for year in years],
    )


def sigma_on_grid(*, cache: FeatureCache, site_fits: dict[int, SiteFit]) -> np.ndarray:
    """Return, per grid row, the residual scale of the fit that supplies the row, in points.

    Rows at or before a site's first cut get `NaN`.
    """
    n = len(cache.end_times)
    sigma = np.full(n, np.nan)
    band = cache.calendar.band
    years = sorted(site_fits)
    for position, year in enumerate(years):
        start = site_fits[year].cut_position + 1
        end = site_fits[years[position + 1]].cut_position + 1 if position + 1 < len(years) else n
        sigma[start:end] = site_fits[year].sigma[band[start:end]]
    return sigma


@dataclass(frozen=True)
class Panel:
    """The contributors' streams on the grid.

    Attributes:
        residual: Each contributor's stitched out-of-sample residuals, in points.
        sigma: Each contributor's residual scale per grid row, in points.
        scaled: Each contributor's scaled residuals, `residual / sigma`.
        factor: For each site, the leave-one-out factor `F_{-s}` over the other contributors.
    """

    residual: dict[str, np.ndarray]
    sigma: dict[str, np.ndarray]
    scaled: dict[str, np.ndarray]
    factor: dict[str, np.ndarray]


def panel_from_residuals(
    *,
    residual: dict[str, np.ndarray],
    sigma: dict[str, np.ndarray],
    min_contributors: int = common.MIN_CONTRIBUTORS,
) -> Panel:
    """Build the scaled residuals and every site's leave-one-out factor from residual streams.

    Args:
        residual: Each contributor's stitched out-of-sample residuals on the grid, in points.
        sigma: Each contributor's residual scale per grid row, in points.
        min_contributors: How many contributors must have a reading for the factor to exist.

    Returns:
        The panel.
    """
    scaled = {site: residual[site] / sigma[site] for site in residual}
    factor: dict[str, np.ndarray] = {}
    for site in common.SCORED_SITES:
        others = np.stack([scaled[other] for other in scaled if other != site])
        counts = np.isfinite(others).sum(axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(others, axis=0)
        factor[site] = np.where(counts >= min_contributors, median, np.nan)
    return Panel(residual=residual, sigma=sigma, scaled=scaled, factor=factor)


def build_panel(
    *,
    cache: FeatureCache,
    all_fits: dict[str, dict[int, SiteFit]],
    first_year: int = min(common.CUT_YEARS),
    min_contributors: int = common.MIN_CONTRIBUTORS,
) -> Panel:
    """Build every contributor's streams and every site's leave-one-out factor.

    Args:
        cache: The feature cache.
        all_fits: The stored fits.
        first_year: The first fit year whose residuals enter the streams.
        min_contributors: How many contributors must have a reading for the factor to exist.

    Returns:
        The panel.
    """
    residual = {
        site: stitched_residuals(cache=cache, site_fits=site_fits, first_year=first_year)
        for site, site_fits in all_fits.items()
    }
    sigma = {
        site: sigma_on_grid(cache=cache, site_fits=site_fits)
        for site, site_fits in all_fits.items()
    }
    return panel_from_residuals(residual=residual, sigma=sigma, min_contributors=min_contributors)


def shift_in_time(*, stream: np.ndarray, half_hours: int) -> np.ndarray:
    """Return the stream as it was `half_hours` earlier: out[t] = stream[t - half_hours]."""
    shifted = np.full(len(stream), np.nan)
    if half_hours >= 0:
        shifted[half_hours:] = stream[: len(stream) - half_hours]
    else:
        shifted[:half_hours] = stream[-half_hours:]
    return shifted


def window_positions(*, cache: FeatureCache, after: datetime, through: datetime) -> slice:
    """Return the grid positions in (`after`, `through`] as a slice."""
    start = common.grid_position_of_cut(cache=cache, cut=after) + 1
    end = common.grid_position_of_cut(cache=cache, cut=through) + 1
    return slice(start, end)


def keep_window(*, stream: np.ndarray, window: slice) -> np.ndarray:
    """Return a copy of the stream with `NaN` outside the window."""
    kept = np.full(len(stream), np.nan)
    kept[window] = stream[window]
    return kept


HUBER_EPSILON = 1.345
HUBER_ITERATIONS = 100


def huber_loading(*, scaled: np.ndarray, factor: np.ndarray) -> float:
    """Return the Huber slope of a site's scaled residual on the factor, with a fitted intercept.

    The fit is iteratively reweighted least squares, with the Huber threshold at
    `HUBER_EPSILON` times a robust scale (1.4826 times the median absolute residual) that is
    re-estimated at each step.

    Args:
        scaled: The site's scaled residuals over the fitting rows.
        factor: The site's leave-one-out factor over the same rows.

    Returns:
        The slope; zero if fewer than 100 rows have both.
    """
    keep = np.isfinite(scaled) & np.isfinite(factor)
    if keep.sum() < 100:
        return 0.0
    design = np.column_stack([np.ones(int(keep.sum())), factor[keep]])
    target = scaled[keep]
    weights = np.ones(len(target))
    coefficients = np.zeros(2)
    for _ in range(HUBER_ITERATIONS):
        root = np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
        residual = target - design @ updated
        scale = MAD_TO_SIGMA * float(np.median(np.abs(residual))) or 1e-9
        threshold = HUBER_EPSILON * scale
        weights = np.where(np.abs(residual) <= threshold, 1.0, threshold / np.abs(residual))
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return float(coefficients[1])


def shrink_loadings(*, raw: dict[str, float], sites: tuple[str, ...]) -> dict[str, float]:
    """Shrink each decision site's loading halfway to the mean over the decision sites.

    Contributors that are not decision sites (and any site without a loading) take the mean.
    """
    mean = float(np.mean([raw[site] for site in sites]))
    return {
        site: (0.5 * raw[site] + 0.5 * mean) if site in raw and site in sites else mean
        for site in common.SCORED_SITES
    }
