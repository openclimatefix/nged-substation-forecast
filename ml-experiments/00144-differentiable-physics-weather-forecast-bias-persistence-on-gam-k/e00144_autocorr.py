"""Step 1 of experiment 144: how far the ENS day-ahead weather error persists from day to day.

For each day D, the forecast error is the ENS ensemble mean for D, from the run that was
published the day before D, minus the analysis proxy for D (the stitched control members of
experiment 111's store). Both are daily means over the 48 half-hours that end on the UTC date D.
The script prints the lag-1 to lag-7 autocorrelation of that daily error, by season, at the seven
stored sites, for temperature and for shortwave irradiance. Wind is not in the store.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np

SRC: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/"
    "00111-differentiable-physics-6-ens-forecast-engine-against-bg2/work/ens_store"
)
OUT: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/"
    "00144-differentiable-physics-weather-forecast-bias-persistence-on-gam-k/work"
)
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
MAX_LAG: Final[int] = 7


def daily_errors(*, site: str) -> dict[str, np.ndarray]:
    """Return the UTC dates and, per variable, the daily mean error (forecast minus proxy).

    Args:
        site: The substation's label.

    Returns:
        A dict with `dates` and one array per variable, NaN where a day has no forecast or proxy.
    """
    with np.load(SRC / f"{site}.npz") as stored:
        origins = stored["origins"].astype("datetime64[m]")
        arrays = {"temperature": stored["temperature"], "shortwave": stored["shortwave"]}
    step = np.timedelta64(30, "m")
    n_runs = len(origins)
    n_grid = (origins[-1] - origins[0]) // step + 48
    proxy = {name: np.full(n_grid, np.nan) for name in arrays}
    for run in range(n_runs):
        first = (origins[run] - origins[0]) // step
        for name, values in arrays.items():
            proxy[name][first : first + 48] = values[run, 0, :48]
    day0 = origins[0].astype("datetime64[D]")
    n_days = n_runs - 1
    result: dict[str, np.ndarray] = {"dates": day0 + np.arange(1, n_days + 1)}
    for name, values in arrays.items():
        error = np.full(n_days, np.nan)
        for run in range(n_days):
            # Run `run` is published at origin `run` and forecasts the UTC date day0 + run + 1.
            date = day0 + np.timedelta64(run + 1, "D")
            ends = date.astype("datetime64[m]") + np.arange(1, 49) * step
            forecast_index = (ends - origins[run]) // step - 1
            proxy_index = (ends - origins[0]) // step - 1
            if forecast_index.max() >= values.shape[2] or proxy_index.max() >= n_grid:
                continue
            forecast = values[run][:, forecast_index].astype(np.float64).mean(axis=0)
            error[run] = np.mean(forecast - proxy[name][proxy_index])
        result[name] = error
    return result


def autocorrelation(*, values: np.ndarray, lag: int) -> float:
    """Return the Pearson correlation of a daily series with itself `lag` days later."""
    first, second = values[:-lag], values[lag:]
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 10:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def main() -> None:
    """Print and store the autocorrelations by variable, season, lag, and anomaly definition."""
    by_site = {site: daily_errors(site=site) for site in SITES}
    months = by_site[SITES[0]]["dates"].astype("datetime64[M]").astype(int) % 12 + 1
    results: dict[str, dict] = {}
    for name in ("temperature", "shortwave"):
        for kind in ("raw", "anomaly"):
            for season, season_months in SEASONS.items():
                in_season = np.isin(months, season_months)
                lags = []
                for lag in range(1, MAX_LAG + 1):
                    per_site = []
                    for site in SITES:
                        error = by_site[site][name].copy()
                        if kind == "anomaly":
                            # Remove each calendar month's mean error, so a constant bias drops out.
                            for month in range(1, 13):
                                mask = months == month
                                error[mask] -= np.nanmean(error[mask])
                        # Pair days whose first day is in the season; the lag crosses no gap.
                        first, second = error[:-lag], error[lag:]
                        keep = in_season[:-lag] & np.isfinite(first) & np.isfinite(second)
                        if keep.sum() >= 10:
                            per_site.append(np.corrcoef(first[keep], second[keep])[0, 1])
                    lags.append(float(np.mean(per_site)))
                results[f"{name}/{kind}/{season}"] = {"mean_over_sites": lags}
                print(f"{name:11s} {kind:8s} {season}", " ".join(f"{x:+.2f}" for x in lags))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "autocorrelation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
