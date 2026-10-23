"""Set each arm's solar levels from stored forecasts and experiment 269, before any new forecast.

Two kinds of level, both written to `work/levels.json`:

- **Shape levels** from real-series fits like experiment 269's, on rows before 2025-07-01 (the
  start of the scored year): `shape_swap` is the
  capacity fitted under the empirical shape divided by the capacity fitted under the physical
  fleet shape (so swapping the shape leaves the solar output level unchanged), and `fitted` is the
  capacity fitted under the empirical shape divided by the pinned capacity (the capacity the real
  series supports, in units of the empirical shape).
- **April-to-June factor** from the control's stored forecasts of the tuning and replicate months
  only (2025-04 to 2025-06, which lie before the scored year): on clear midday half-hours at leads
  of 12 to 36 hours, the factor is one minus the mean forecast shortfall (actual minus median)
  over the mean solar output the control subtracts. It is bounded to 0.5 to 1.2.

    uv run --no-sync python ml-experiments/00280-*/e00280_levels.py
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import e00280_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, derated_unit_shape, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CONTROL_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00180-dp-sunny-daylight-fix-d5-solar-shape"
    / "work"
    / "arms"
    / "c"
    / "dp_preds"
)
LEAD_RANGE_H: Final[tuple[float, float]] = (12.0, 36.0)
MIN_SHAPE: Final[float] = 0.5
"""A half-hour counts as midday when the derated fleet shape is at least this."""
FACTOR_BOUNDS: Final[tuple[float, float]] = (0.5, 1.2)
CALIBRATION_MONTHS: Final[tuple[tuple[int, int], ...]] = ((2025, 4), (2025, 5), (2025, 6))
LAST_DATA_SEASON_YEAR: Final[int] = 2024
CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)


def fitted_capacity_points(*, cache: FeatureCache, site: str, shape: np.ndarray) -> float:
    """Return the capacity in points that experiment 129's convex fit finds under a shape.

    The fit uses rows before `CUT` only, and the capacity is that of the last season-year with
    rows (2024), which the fit holds after the cut.
    """
    features = cache.sites[site]
    shaped = replace(cache, sites={**cache.sites, site: replace(features, pv_unit_shape=shape)})
    fit = fit_gam(
        cache=shaped,
        site=site,
        specs=[GamSpec(ridge=common.RIDGE, capacity="fitted")],
        cut=CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        device="cpu",
        dtype=torch.float64,
    )
    return float(fit.capacity_points()[LAST_DATA_SEASON_YEAR])


def shape_levels() -> dict[str, dict[str, float]]:
    """Return each substation's two shape levels from real-series fits before the scored year."""
    cache = common.load_cache()
    out = {}
    for site in common.SCORED_SITES:
        features = cache.sites[site]
        fleet = features.pv_unit_shape
        table = common.empirical_table(site=site)
        empirical = table.predict(
            irradiance_w_m2=np.nan_to_num(features.ghi_w_m2), elevation_deg=features.elevation_deg
        )
        empirical = np.clip(empirical, 0.0, common.MAX_EMPIRICAL_SHAPE)
        empirical[features.elevation_deg <= 0.0] = 0.0
        empirical[~np.isfinite(features.ghi_w_m2)] = np.nan
        both = np.isfinite(fleet) & np.isfinite(empirical)
        physical_points = fitted_capacity_points(
            cache=cache, site=site, shape=np.where(both, fleet, np.nan)
        )
        empirical_points = fitted_capacity_points(
            cache=cache, site=site, shape=np.where(both, empirical, np.nan)
        )
        held = pinned_capacity(vintages=read_vintages(), table=features.table, cut=CUT).held_points
        out[site] = {
            "shape_swap": empirical_points / physical_points,
            "fitted": empirical_points / held,
            "fleet_fitted": physical_points / held,
            "physical_points": physical_points,
            "empirical_points": empirical_points,
            "pinned_points": float(held),
        }
    return out


def april_june_factor(*, site: str) -> dict[str, float]:
    """Return the April-to-June factor at a substation from the control's calibration months."""
    cache = common.load_cache()
    features = cache.sites[site]
    held = pinned_capacity(vintages=read_vintages(), table=features.table, cut=CUT).held_points
    frames = [
        pl.read_parquet(CONTROL_DIR / f"a2_{period}_{site}.parquet")
        for period in ("tuning", "replicate")
    ]
    frame = pl.concat(frames).filter(
        ~pl.col("switching_mask"),
        pl.col("actual").is_finite(),
        pl.col("lead_hours").is_between(*LEAD_RANGE_H),
    )
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    index = np.searchsorted(cache.end_times, valid)
    shape = derated_unit_shape(features=features, index=index)
    months = pl.Series(valid).dt.year(), pl.Series(valid).dt.month()
    in_months = np.zeros(len(index), dtype=bool)
    for year, month in CALIBRATION_MONTHS:
        in_months |= ((months[0] == year) & (months[1] == month)).to_numpy()
    keep = in_months & (shape >= MIN_SHAPE)
    shortfall = float(np.mean((frame["actual"] - frame["q50"]).to_numpy()[keep]))
    subtracted = float(np.mean(held * shape[keep]))
    raw = 1.0 - shortfall / subtracted
    low, high = FACTOR_BOUNDS
    return {
        "april_june": float(np.clip(raw, low, high)),
        "raw": raw,
        "n_rows": int(keep.sum()),
        "shortfall_points": shortfall,
        "subtracted_points": subtracted,
    }


def main() -> None:
    """Write `work/levels.json` and print it."""
    shapes = shape_levels()
    levels = {
        site: {**shapes[site], **{f"aj_{k}": v for k, v in april_june_factor(site=site).items()}}
        for site in common.SCORED_SITES
    }
    common.write_json_atomic(path=common.WORK_DIR / "levels.json", content=levels)
    for site, values in levels.items():
        print(site, {k: round(v, 3) for k, v in values.items()})


if __name__ == "__main__":
    main()
