"""Test why the extended default beats climatology at 9 to 10 days, and describe the ensemble steps.

Three checks, none of which changes a score. (1) The ensemble-mean 24-hour temperature's anomaly
correlation with ERA5, by 24-hour lead band, over the scored rows. A positive value at 216 to 240
hours says the weather forecast still carries information there. (2) The gain of the extended
default over climatology split by the valid month (winter, November to March, against the rest),
because a gain from heating demand sits in winter and a gain from the calendar profile does not.
(3) The step width of the stored ensemble by lead band, found as the share of half-hours where the
ensemble-mean temperature's second difference is zero (a straight line between stored steps).

    uv run --no-sync python ml-experiments/00317-*/e00317_weather_skill.py
"""

import itertools

import e00317_common as common
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache

CACHE_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
BAND_EDGES = np.arange(0.0, 241.0, 24.0)
WINDOW = 49


def centred_mean(*, values: np.ndarray, width: int) -> np.ndarray:
    """Return the centred moving average with NaN-aware windows."""
    finite = np.isfinite(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    ccount = np.concatenate([[0.0], np.cumsum(finite)])
    half = width // 2
    idx = np.arange(len(values))
    low, high = np.clip(idx - half, 0, len(values)), np.clip(idx + half + 1, 0, len(values))
    total, number = csum[high] - csum[low], ccount[high] - ccount[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(number >= width * 0.8, total / number, np.nan)


def main() -> None:
    """Write `weather_skill.json`."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    out: dict = {"anomaly_correlation": {}, "gain_by_season": {}, "ensemble_steps": {}}
    labels = [f"{int(a)}-{int(b)} h" for a, b in itertools.pairwise(BAND_EDGES)]
    pieces = []
    for site in common.SCORED_SITES:
        frame = (
            pl.read_parquet(
                common.CHAIN_DIR / f"{site}.parquet",
                columns=[
                    "valid_time",
                    "lead_hours",
                    "ens_mean_c",
                    "period",
                    "switching_mask",
                    "actual",
                ],
            )
            .filter(
                (pl.col("period") == "scored")
                & pl.col("ens_mean_c").is_finite()
                & pl.col("actual").is_finite()
            )
            .filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
        )
        obs = centred_mean(values=cache.sites[site].temperature_c.astype(np.float64), width=WINDOW)
        position = np.searchsorted(
            cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        valid = position < len(obs)
        value = np.full(frame.height, np.nan)
        value[valid] = obs[position[valid]]
        pieces.append(frame.with_columns(obs_c=pl.Series(value), site=pl.lit(site)))
    rows = pl.concat(pieces).filter(pl.col("obs_c").is_finite())
    month = rows["valid_time"].dt.month().to_numpy()
    band = np.clip(
        np.searchsorted(BAND_EDGES, rows["lead_hours"].to_numpy(), side="left") - 1,
        0,
        len(labels) - 1,
    )
    ens = rows["ens_mean_c"].to_numpy()
    obs = rows["obs_c"].to_numpy()
    site = rows["site"].to_numpy()
    for i, label in enumerate(labels):
        mask = band == i
        anomaly_e, anomaly_o = ens.copy(), obs.copy()
        for s in common.SCORED_SITES:
            for m in range(1, 13):
                cell = mask & (site == s) & (month == m)
                if cell.any():
                    anomaly_e[cell] -= ens[cell].mean()
                    anomaly_o[cell] -= obs[cell].mean()
        r = float(np.corrcoef(anomaly_e[mask], anomaly_o[mask])[0, 1])
        out["anomaly_correlation"][label] = {
            "r": r,
            "n": int(mask.sum()),
            "rmse_anomaly_c": float(np.sqrt(np.mean((anomaly_e[mask] - anomaly_o[mask]) ** 2))),
            "sd_obs_anomaly_c": float(anomaly_o[mask].std()),
        }
    # (3) Step width: ensemble-mean temperature of one substation's runs, second difference zero.
    with np.load(
        common.ML_EXPERIMENTS_DATA_DIR
        / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
        / "work"
        / "ens_store"
        / "D10.npz"
    ) as stored:
        temperature = np.nanmean(stored["temperature"][:100].astype(np.float64), axis=1)
        out["ensemble_steps"]["store_shape"] = list(stored["temperature"].shape)
    d2 = np.abs(temperature[:, 2:] - 2 * temperature[:, 1:-1] + temperature[:, :-2])
    lead = (np.arange(1, d2.shape[1] + 1) + 1) / 2.0
    for i, label in enumerate([*labels, "240-351 h"]):
        lo, hi = (BAND_EDGES[i], BAND_EDGES[i + 1]) if i < len(labels) else (240.0, 352.0)
        mask = (lead > lo) & (lead <= hi)
        out["ensemble_steps"][label] = {
            "share_second_difference_zero": float(np.mean(d2[:, mask] < 1e-6))
        }
    # (2) Gain over climatology by season of the valid time, 168 to 240 hours.
    scored = pl.read_parquet(common.REPORT_DIR / "scored_rows.parquet")
    actual = scored["actual"].to_numpy().astype(np.float64)
    loss = {
        k: 2.0
        * np.mean(
            np.maximum(
                common.LEVELS * (actual[:, None] - q), (common.LEVELS - 1) * (actual[:, None] - q)
            ),
            axis=1,
        )
        for k, q in (
            (
                "A",
                scored.select([f"A_{c}" for c in common.QUANTILES]).to_numpy().astype(np.float64),
            ),
            (
                "CL",
                scored.select([f"C_{c}" for c in common.QUANTILES]).to_numpy().astype(np.float64),
            ),
            (
                "MH",
                scored.select([f"M_{c}" for c in common.QUANTILES]).to_numpy().astype(np.float64),
            ),
        )
    }
    vm = scored["valid_time"].dt.month().to_numpy()
    lead_s = scored["lead_hours"].to_numpy()
    winter = np.isin(vm, [11, 12, 1, 2, 3])
    for name, m in (("winter (Nov-Mar)", winter), ("rest (Apr-Oct)", ~winter)):
        for win, w in (("168-240 h", lead_s > 168.0), ("216-240 h", lead_s > 216.0)):
            mask = m & w
            out["gain_by_season"][f"{name}, {win}"] = {
                "A": float(loss["A"][mask].mean()),
                "CL": float(loss["CL"][mask].mean()),
                "MH": float(loss["MH"][mask].mean()),
                "gain_vs_CL": float(1 - loss["A"][mask].mean() / loss["CL"][mask].mean()),
                "gain_vs_MH": float(1 - loss["A"][mask].mean() / loss["MH"][mask].mean()),
                "n": int(mask.sum()),
            }
    common.write_json_atomic(path=common.REPORT_DIR / "weather_skill.json", content=out)
    print({k: round(v["r"], 3) for k, v in out["anomaly_correlation"].items()})
    print(
        {
            k: (round(v["gain_vs_CL"], 3), round(v["gain_vs_MH"], 3), v["n"])
            for k, v in out["gain_by_season"].items()
        }
    )
    print(out["ensemble_steps"])


if __name__ == "__main__":
    main()
