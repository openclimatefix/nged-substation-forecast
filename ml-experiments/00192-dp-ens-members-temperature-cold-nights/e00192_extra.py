# ruff: noqa: E501
"""Checks beside the main scores: the gap to BG2 on sunny rows, the mean forecast's error, and tails.

- The gap to BG2 follows experiment 178: arm minus BG2 in MW on the same substation, run, and
  half-hour, over the sunniest third of daylight half-hours of the scored year, as a mean of the
  absolute error of the median and as a mean pinball loss over the 11 quantiles.
- The mean forecast of arms M and F is arm A's mean plus the mean over members of the response.
- Tail calibration (coverage and pinball loss at 0.01, 0.05, 0.95, 0.99) for daylight and all rows.

    uv run --no-sync python ml-experiments/00192-*/e00192_extra.py
"""

import json
from typing import Final

import e00192_common as common
import e00192_data as data
import e00192_fit as fit
import e00192_score as score
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BG2_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
GAP_ARMS: Final[tuple[str, ...]] = ("a", "m", "f", "s", "mr", "mp")


def mean_block_interval(*, month: np.ndarray, values: np.ndarray, seed: int = 184) -> dict:
    """Return the mean of `values` and a 95% interval from resampling whole months."""
    labels, inverse = np.unique(month, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(score.N_DRAWS, len(labels)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(sums.sum() / counts.sum()), "lower": float(low), "upper": float(high)}


def bg2_gap(*, frame: pl.DataFrame, arms: dict[str, np.ndarray]) -> dict:
    """Return each arm's gap to BG2 on the sunniest third of daylight half-hours, in MW."""
    run_dates = common.run_dates()
    origin = pl.Series(
        [common.origin_of(run=run_dates[int(r)]) for r in frame["run"].to_numpy()]
    ).dt.cast_time_unit("us")
    frame = frame.with_columns(
        power_fcst_init_time=origin.dt.replace_time_zone("UTC")
    ).with_row_index("row")
    bg2 = pl.read_parquet(BG2_PATH).filter(pl.col("site").is_in(list(common.SCORED_SITES)))
    capacity = bg2.group_by("site").agg(capacity_mw=pl.col("capacity_mw").first())
    keys = ["site", "power_fcst_init_time", "valid_time"]
    q_cols = list(data.QUANTILES)
    joined = frame.join(
        bg2.select(*keys, **{f"b_{c}": pl.col(c) for c in q_cols}),
        on=keys,
        how="inner",
        validate="1:1",
    ).join(capacity, on="site")
    rows = joined["row"].to_numpy()
    cap = joined["capacity_mw"].to_numpy()
    actual = joined["actual"].to_numpy().astype(np.float64) * cap / 100.0
    b = joined.select([f"b_{c}" for c in q_cols]).to_numpy().astype(np.float64) * cap[:, None]

    def pinball(q: np.ndarray) -> np.ndarray:
        error = actual[:, None] - q
        return np.mean(np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error), axis=1)

    bg2_pin = pinball(b)
    bg2_abs = np.abs(actual - b[:, 5])
    labelled = score.label_columns(frame=frame)
    usable = (
        labelled["usable"].to_numpy()[rows]
        & labelled["sunny"].to_numpy()[rows]
        & (labelled["kind"].to_numpy()[rows] == "daylight")
    )
    month = labelled["month"].to_numpy()[rows]
    lead = labelled["lead_hours"].to_numpy()[rows]
    site = labelled["site"].to_numpy()[rows]
    out: dict = {"n_joined": len(rows)}
    scopes = {
        "five substations, all leads": usable & np.isin(site, common.DECISION_SITES),
        "five substations, under 72 h": usable
        & np.isin(site, common.DECISION_SITES)
        & (lead < 72.0),
        "six substations, all leads": usable,
        "five substations without D5": usable
        & np.isin(site, common.DECISION_SITES)
        & (site != "D5"),
        "D5 only": usable & (site == "D5"),
    }
    for scope, mask in scopes.items():
        entry = {"rows": int(mask.sum())}
        for arm in GAP_ARMS:
            q = arms[arm][rows] * cap[:, None] / 100.0
            entry[arm] = {
                "gap_abs_mw": mean_block_interval(
                    month=month[mask], values=(np.abs(actual - q[:, 5]) - bg2_abs)[mask]
                ),
                "gap_pin_mw": mean_block_interval(
                    month=month[mask], values=(pinball(q) - bg2_pin)[mask]
                ),
            }
        out[scope] = entry
    return out


def mean_forecast_error(*, frame: pl.DataFrame, delta_m: np.ndarray, delta_f: np.ndarray) -> dict:
    """Return the mean forecast's absolute error of arms A, M, and F, and median minus mean."""
    labelled = score.label_columns(frame=frame)
    mean = frame["mean"].to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    shifts = {
        "a": np.zeros(len(mean)),
        "m": np.nanmean(delta_m, axis=1),
        "f": np.nanmean(delta_f, axis=1),
    }
    day = (labelled["kind"] == "daylight").to_numpy() & labelled["usable"].to_numpy()
    five = labelled["site"].is_in(list(common.DECISION_SITES)).to_numpy()
    month = labelled["month"].to_numpy()
    out: dict = {}
    for arm, shift in shifts.items():
        error = np.abs(actual - (mean + np.nan_to_num(shift)))
        base = np.abs(actual - mean)
        mask = day & five
        out[arm] = {
            "mae_daylight": float(error[mask].mean()),
            "change_pct": score.percent_change(
                month=month[mask], arm=error[mask], reference=base[mask], level=95.0, seed=184
            ),
            "mean_shift_daylight": float(np.nan_to_num(shift)[mask].mean()),
        }
    return out


def tails(*, frame: pl.DataFrame, arms: dict[str, np.ndarray]) -> dict:
    """Return tail coverage and pinball loss for daylight and all rows of the five substations."""
    labelled = score.label_columns(frame=frame)
    masks = score.subsets(frame=labelled)
    out: dict = {}
    for name in ("daylight", "all rows"):
        mask = masks[name]
        out[name] = {}
        for arm in ("a", "m", "f", "s"):
            rm = score.row_metrics(frame=frame, quantiles=arms[arm])
            actual = frame["actual"].to_numpy().astype(np.float64)
            q = arms[arm]
            out[name][arm] = {
                "cover01": float((actual <= q[:, 0])[mask].mean()),
                "cover05": float(rm["below05"][mask].mean()),
                "cover95": float(rm["below95"][mask].mean()),
                "cover99": float((actual <= q[:, 10])[mask].mean()),
                "pin01": float(rm["pin01"][mask].mean()),
                "pin05": float(rm["pin05"][mask].mean()),
                "pin95": float(rm["pin95"][mask].mean()),
                "pin99": float(rm["pin99"][mask].mean()),
                "width_q05_q95": float(rm["width"][mask].mean()),
            }
    return out


def median_minus_mean(*, frame: pl.DataFrame, arms: dict[str, np.ndarray]) -> dict:
    """Return the mean of (median forecast minus mean forecast) per substation at 12 to 36 hours."""
    lead = frame["lead_hours"].to_numpy()
    mask = (lead >= 12.0) & (lead < 36.0) & data.usable_rows(frame=frame)
    mean = frame["mean"].to_numpy().astype(np.float64)
    out = {}
    for arm in ("a", "m", "f", "s"):
        out[arm] = {
            site: float((arms[arm][:, 5] - mean)[mask & (frame["site"].to_numpy() == site)].mean())
            for site in common.SCORED_SITES
        }
    return out


def main() -> None:
    """Write `extra.json`."""
    frame = pl.read_parquet(fit.ARM_DIR / "scored_frame.parquet")
    arms = {arm: np.load(fit.ARM_DIR / f"scored_{arm}.npy").astype(np.float64) for arm in fit.ARMS}
    delta_m = np.load(fit.ARM_DIR / "scored_delta_m.npy")
    cache = common.load_cache()
    _, _, delta_f = data.load_period(period="scored", cache=cache)
    report = {
        "bg2_gap": bg2_gap(frame=frame, arms=arms),
        "mean_forecast_error": mean_forecast_error(frame=frame, delta_m=delta_m, delta_f=delta_f),
        "tails": tails(frame=frame, arms=arms),
        "median_minus_mean_12_36h": median_minus_mean(frame=frame, arms=arms),
    }
    common.write_json_atomic(path=score.SCORE_DIR / "extra.json", content=report)
    print(
        json.dumps(
            {k: report[k] for k in ("mean_forecast_error", "median_minus_mean_12_36h")},
            indent=1,
            default=float,
        )
    )
    for scope, entry in report["bg2_gap"].items():
        if scope == "n_joined":
            continue
        print(scope, entry["rows"], {a: round(entry[a]["gap_abs_mw"]["mean"], 4) for a in GAP_ARMS})


if __name__ == "__main__":
    main()
