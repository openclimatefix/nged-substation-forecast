"""Step 0 of experiment 00230: the oracle ceiling with ERA5 as the weather at inference too.

Fits, for each farm and on valid times before 2025-06-17, (P) the physics core with its idle layer,
(S) a binned power curve, and (X) a flexible ERA5-trained XGBoost quantile model, each turned into
11 quantiles, and scores them on experiment 52's 53 Tuesday runs' cases with ERA5 wind as the
weather. Reports R squared of the median forecast per farm, the quantile score on all rows, rows
kept by experiment 210's mask, and rows above the core's rated speed, the idle-row coverage check,
and the gate: whether the best of P and X improves on S by at least 0.02 points.

Output: `work/step0.json`, `work/step0_cores.json`.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_step0.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00230_common as common  # noqa: E402
import e00230_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

N_ROUNDS = 500
XGBOOST_PARAMETERS = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
    "seed": 0,
}
SPEED_BIN_WIDTH_M_S = 0.5
GATE_POINTS = 0.02
BOOTSTRAP_RESAMPLES = 5_000


def binned_curve(*, speed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a non-decreasing power curve of the fraction on 100 m wind speed, by 0.5 m/s bin."""
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": target})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())


def weather_arrays(*, frame: pl.DataFrame) -> core_module.WeatherArrays:
    """Return the three columns the physics core reads."""
    return core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy(),
        speed_100m=frame["wind_speed_100m"].to_numpy(),
        temperature_c=frame["temperature_c"].to_numpy(),
    )


def month_block_gap(
    *, losses_a: np.ndarray, losses_b: np.ndarray, months: np.ndarray
) -> list[float]:
    """Return the mean of `a - b` and its 95% interval, resampling whole months."""
    labels = np.unique(months)
    sums = np.array([(losses_a - losses_b)[months == m].sum() for m in labels])
    counts = np.array([(months == m).sum() for m in labels])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, labels.size, size=(BOOTSTRAP_RESAMPLES, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    point = float(sums.sum() / counts.sum())
    return [point, *[float(x) for x in np.quantile(means, [0.025, 0.975])]]


def main() -> None:
    """Fit the three ERA5-input arms per farm, score them, and write the gate."""
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    train = common.training_table(sites=sites, era5=era5)
    cases = common.cases_table(sites=sites)
    rows = cases.join(era5, on=["time_series_id", "valid_time"], how="left", validate="m:1")
    if rows["wind_speed_100m"].null_count() or rows["temperature_c"].null_count():
        msg = "some validation cases have no ERA5 weather"
        raise ValueError(msg)
    arms = {"P": [], "S": [], "X": []}
    pieces, fits, rsquared = [], {}, {}
    for site_row in sites.iter_rows(named=True):
        series, label = int(site_row["time_series_id"]), site_row["site"]
        fit_rows = train.filter(pl.col("time_series_id") == series)
        part = rows.filter(pl.col("time_series_id") == series).sort(*common.KEY_COLUMNS)
        target = fit_rows["target"].to_numpy()
        core = core_module.fit_core(
            weather=weather_arrays(frame=fit_rows),
            target=target,
            with_idle_layer=label != "W2",
            seed=0,
        )
        expected_train = core_module.predict_expected_power(
            core=core, weather=weather_arrays(frame=fit_rows)
        )
        residuals = core_module.ResidualQuantiles.fit(expected=expected_train, observed=target)
        expected = core_module.predict_expected_power(core=core, weather=weather_arrays(frame=part))
        quantiles = {"P": residuals.quantiles(expected=expected)}
        centres, curve = binned_curve(speed=fit_rows["wind_speed_100m"].to_numpy(), target=target)
        mu_train = np.interp(fit_rows["wind_speed_100m"].to_numpy(), centres, curve)
        simple = core_module.ResidualQuantiles.fit(expected=mu_train, observed=target)
        mu = np.interp(part["wind_speed_100m"].to_numpy(), centres, curve)
        quantiles["S"] = simple.quantiles(expected=mu)
        features = list(common.XGBOOST_FEATURES)
        margin = np.tile(
            np.interp(fit_rows["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
        )
        booster = xgb.train(
            XGBOOST_PARAMETERS,
            xgb.DMatrix(fit_rows.select(features).to_numpy(), label=target, base_margin=margin),
            num_boost_round=N_ROUNDS,
        )
        part_margin = np.tile(mu[:, None], (1, 11))
        quantiles["X"] = np.sort(
            np.clip(
                booster.predict(
                    xgb.DMatrix(part.select(features).to_numpy(), base_margin=part_margin)
                ),
                0.0,
                1.0,
            ),
            axis=1,
        ).astype(np.float32)
        actual = part["actual_norm"].to_numpy().astype(np.float64)
        rated = float(core.rated_m_s.detach())
        above_rated = part["wind_speed_100m"].to_numpy() >= rated
        idle = part["wind_speed_100m"].to_numpy() < common.IDLE_WIND_M_S
        fits[label] = {
            "hub_height_m": float(core.hub_height_m.detach()),
            "cut_in_m_s": float(core.cut_in_m_s.detach()),
            "rated_m_s": rated,
            "idle_floor_fraction": float(
                torch_softplus(core.raw_idle_floor) if core.with_idle_layer else 0.0
            ),
            "share_above_rated": float(above_rated.mean()),
            "share_idle": float(idle.mean()),
            "idle_rows": int(idle.sum()),
            "idle_median_observed_fraction": float(np.median(actual[idle])),
            "idle_band_coverage": {
                name: float(common.covered(quantiles=q[idle], actual=actual[idle]).mean())
                for name, q in quantiles.items()
            },
            "idle_mean_median_forecast": {
                name: float(q[idle, common.MEDIAN_INDEX].mean()) for name, q in quantiles.items()
            },
        }
        rsquared[label] = {
            name: float(
                1
                - ((actual - q[:, common.MEDIAN_INDEX]) ** 2).sum()
                / ((actual - actual.mean()) ** 2).sum()
            )
            for name, q in quantiles.items()
        }
        for name, q in quantiles.items():
            arms[name].append(
                common.quantile_score(quantiles=q, actual=actual) * common.POINTS_PER_FRACTION
            )
        pieces.append(
            part.select("site", "valid_time", "masked").with_columns(
                above_rated=pl.Series(above_rated)
            )
        )
    frame = pl.concat(pieces)
    months = frame["valid_time"].dt.strftime("%Y-%m").to_numpy()
    masks = {
        "all rows": np.ones(frame.height, dtype=bool),
        "kept rows": ~frame["masked"].to_numpy(),
        "above rated": frame["above_rated"].to_numpy(),
    }
    scores = {name: np.concatenate(values) for name, values in arms.items()}
    report: dict = {"fits": fits, "r_squared": rsquared, "scores": {}, "gate": {}}
    for scope, keep in masks.items():
        report["scores"][scope] = {name: float(s[keep].mean()) for name, s in scores.items()}
    ceiling_name = min(("P", "X"), key=lambda n: report["scores"]["all rows"][n])
    gap = month_block_gap(losses_a=scores["S"], losses_b=scores[ceiling_name], months=months)
    report["gate"] = {
        "ceiling_arm": ceiling_name,
        "S_minus_ceiling_all_rows": gap,
        "threshold": GATE_POINTS,
        "skip_B3_B5_B6": bool(gap[0] < GATE_POINTS),
    }
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "step0.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def torch_softplus(raw: object) -> float:
    """Return softplus of a scalar tensor as a Python float."""
    import torch

    return float(torch.nn.functional.softplus(raw.detach()))


if __name__ == "__main__":
    main()
