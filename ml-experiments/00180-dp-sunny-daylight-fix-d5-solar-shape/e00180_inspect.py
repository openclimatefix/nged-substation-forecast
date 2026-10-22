"""Experiment 180, stage 0: inspect the solar properties of the six substations (no fit).

Prints, per substation, on clear summer half-hours (clear-sky index of CAMS irradiance 0.9 or more,
May to August, scored years before 2026-07-01): the half-hour of the net-power minimum against the
half-hour of the fleet shape's peak, and the mean net power by decile of the fleet shape, which
flattens at the top if the solar output is clipped.

    uv run --no-sync python ml-experiments/00180-*/e00180_inspect.py
"""

import json
from typing import Final

import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CACHE_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
CLEAR_INDEX: Final[float] = 0.9


def main() -> None:
    """Print the stage 0 tables."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    month = cache.calendar.local_date.astype("datetime64[M]").astype(np.int64) % 12 + 1
    half_hour = cache.calendar.half_hour.astype(np.int64)
    stored = {}
    for site in SITES:
        f = cache.sites[site]
        clear = haurwitz_w_m2(apparent_zenith_deg=90.0 - f.elevation_deg)
        csi = np.where(clear > 20, f.ghi_w_m2 / np.where(clear > 20, clear, 1), np.nan)
        ok = np.isfinite(f.y) & np.isfinite(f.pv_unit_shape) & (csi >= CLEAR_INDEX)
        summer = ok & np.isin(month, [5, 6, 7, 8]) & (f.pv_unit_shape > 0.01)
        frame = pl.DataFrame(
            {"hh": half_hour[summer], "y": f.y[summer], "u": f.pv_unit_shape[summer]}
        )
        prof = frame.group_by("hh").agg(y=pl.col("y").mean(), u=pl.col("u").mean(), n=pl.len())
        prof = prof.filter(pl.col("n") >= 30).sort("hh")
        night = np.isfinite(f.y) & (f.elevation_deg < -5)
        night_mean = float(np.mean(f.y[night]))
        edges = np.quantile(frame["u"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
        deciles = (
            frame.with_columns(d=pl.Series(np.searchsorted(edges, frame["u"].to_numpy())))
            .group_by("d")
            .agg(u=pl.col("u").mean(), y=pl.col("y").mean())
            .sort("d")
        )
        slope = (night_mean - deciles["y"]) / deciles["u"]
        print(f"== {site}: night mean {night_mean:.1f} pts; summer clear half-hours {frame.height}")
        stored[site] = [float(s) for s in slope]
        print(
            "  net-power minimum at half-hour",
            int(prof["hh"][int(np.argmin(prof["y"].to_numpy()))]),
            "; shape peak at half-hour",
            int(prof["hh"][int(np.argmax(prof["u"].to_numpy()))]),
        )
        print(
            "  implied points of solar per unit shape, by shape decile:",
            [round(float(s), 1) for s in slope],
        )

    target = (
        ML_EXPERIMENTS_DATA_DIR
        / "00180-dp-sunny-daylight-fix-d5-solar-shape"
        / "work"
        / "stage0.json"
    )
    target.write_text(json.dumps(stored))


if __name__ == "__main__":
    main()
