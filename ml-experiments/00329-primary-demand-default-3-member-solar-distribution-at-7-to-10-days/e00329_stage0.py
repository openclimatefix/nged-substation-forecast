# ruff: noqa: E501
"""Stage 0 of experiment 329: what is left of the sunny-daylight weakness at 168 to 240 hours?

Zero-fit. Scores Primary demand research default 3 (experiment 322's `d1`) against the manual
heuristic and climatology on realised-sunny and forecast-sunny half-hours, by 24-hour lead band and
by Primary, with the probability-integral-transform histogram, the tail rates, and the q50's bias.
The reproduction gate (2.727 at 0.5 to 240 hours, 3.077 at 168 to 240 hours) uses experiment 322's
stored per-row scores.

    PYTHONPATH=the folders of the earlier experiments uv run --no-sync python ml-experiments/00329-*/e00329_stage0.py
"""

import json

import e00329_common as common
import numpy as np
import polars as pl
from e00317_score import read_reference  # ty: ignore[unresolved-import]

LOW_BOUND = 8.0
"""Pre-registered: below-q05 rate above this percentage on realised-sunny half-hours."""


def summary(*, actual: np.ndarray, q: np.ndarray, mask: np.ndarray) -> dict:
    """Return the score, the tail rates (percent), and the q50 bias on the masked rows.

    The bias is the q50 minus the realised demand, so a positive bias means the forecast is above it.
    """
    a, qq = actual[mask], q[mask]
    return {
        "n": int(mask.sum()),
        "score": float(common.pinball(actual=a, q=qq).mean()),
        "below_q01": float((a < qq[:, 0]).mean() * 100),
        "below_q05": float((a < qq[:, 1]).mean() * 100),
        "above_q95": float((a > qq[:, 9]).mean() * 100),
        "above_q99": float((a > qq[:, 10]).mean() * 100),
        "q50_bias": float((qq[:, 5] - a).mean()),
    }


def main() -> None:
    """Assemble the scored year's long-lead rows and write `stage0.json`."""
    frame, _ = common.load_default3(period="scored")
    frame = frame.filter(pl.col("lead_hours") <= 240.0)
    manual = read_reference(name="ref_manual_heuristic", tag="mh")
    clim = read_reference(name="ref_climatology", tag="cl")
    keys = common.s317.common.KEYS
    frame = frame.join(manual, on=keys, how="inner", validate="1:1").join(
        clim, on=keys, how="inner", validate="1:1"
    )
    frame = frame.filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    csi = common.forecast_clear_sky_index(frame=frame)
    actual = frame["actual"].to_numpy().astype(np.float64)
    arms = {
        "default": frame.select([f"d1_{q}" for q in common.QUANTILES])
        .to_numpy()
        .astype(np.float64),
        "manual_heuristic": frame.select([f"mh_{q}" for q in common.QUANTILES])
        .to_numpy()
        .astype(np.float64),
        "climatology": frame.select([f"cl_{q}" for q in common.QUANTILES])
        .to_numpy()
        .astype(np.float64),
    }
    lead = frame["lead_hours"].to_numpy()
    band = np.clip(
        np.searchsorted(np.array([168.0, 192.0, 216.0, 240.0]), lead, side="left") - 1, 0, 2
    )
    long = lead > 168.0
    sunny = frame["sunny_daylight"].to_numpy() & long
    fsunny = (
        (np.nan_to_num(csi, nan=0.0) >= common.CSI_FORECAST_SUNNY)
        & frame["daylight"].fill_null(False).to_numpy()
        & long
    )
    site = frame["site"].to_numpy()
    out: dict = {"n_long_rows": int(long.sum())}
    groups = {
        "realised-sunny daylight": sunny,
        "forecast-sunny daylight": fsunny,
        "all daylight": frame["daylight"].fill_null(False).to_numpy() & long,
        "all rows": long,
    }
    for g, gm in groups.items():
        out[g] = {a: summary(actual=actual, q=q, mask=gm) for a, q in arms.items()}
        out[g]["by_band"] = {
            common.BAND_LABELS[b]: {
                a: summary(actual=actual, q=q, mask=gm & (band == b)) for a, q in arms.items()
            }
            for b in range(3)
        }
        out[g]["by_site"] = {
            s: {a: summary(actual=actual, q=q, mask=gm & (site == s)) for a, q in arms.items()}
            for s in common.SITES
        }
    # PIT histogram on realised-sunny half-hours: the fraction of rows by the position of the actual among the 11 quantiles.
    for a, q in arms.items():
        position = (actual[sunny, None] > q[sunny]).sum(axis=1)
        out.setdefault("pit_sunny", {})[a] = (
            np.bincount(position, minlength=12) / sunny.sum()
        ).tolist()
    # Where the gap went: the chain (experiment 317), default 2 (319's x1), and default 3 on realised-sunny daylight.
    stages = {
        "chain (experiment 317)": frame.select(list(common.QUANTILES))
        .to_numpy()
        .astype(np.float64),
        "default 2 (experiment 319 x1)": frame.select([f"x1_{q}" for q in common.QUANTILES])
        .to_numpy()
        .astype(np.float64),
        "default 3": arms["default"],
        "manual_heuristic": arms["manual_heuristic"],
    }
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    loss = {
        k: common.pinball(actual=actual, q=v)
        for k, v in {**stages, "climatology": arms["climatology"]}.items()
    }
    out["sunny_by_stage"] = {
        k: {
            "score": float(loss[k][sunny].mean()),
            "mean_abs_median_error": float(np.abs(actual - v[:, 5])[sunny].mean()),
        }
        for k, v in stages.items()
    }
    from e00319_score import contrast  # ty: ignore[unresolved-import]

    for k in ("chain (experiment 317)", "default 2 (experiment 319 x1)", "default 3"):
        out["sunny_by_stage"][k]["vs_heuristic"] = contrast(
            treat=loss[k], ref=loss["manual_heuristic"], mask=sunny, blocks=month
        )
        out["sunny_by_stage"][k]["vs_heuristic_without_D5"] = contrast(
            treat=loss[k], ref=loss["manual_heuristic"], mask=sunny & (site != "D5"), blocks=month
        )
    out["median_minus_chain_mean_by_site"] = {
        s_: float((arms["default"][:, 5] - frame["mean"].to_numpy())[(site == s_) & long].mean())
        for s_ in common.SITES
    }
    # Reproduction gate from experiment 322's stored per-row scores.
    stored = pl.read_parquet(
        common.c322.REPORT_DIR / "scored_rows.parquet", columns=["lead_hours", "loss_d1"]
    )
    out["gate"] = {
        "default_0_240": float(stored["loss_d1"].to_numpy().mean()),
        "default_168_240": float(
            stored.filter(pl.col("lead_hours") > 168.0)["loss_d1"].to_numpy().mean()
        ),
        "expected": [2.727, 3.077],
        "recomputed_168_240_all_rows": float(
            common.pinball(actual=actual, q=arms["default"])[long].mean()
        ),
    }
    d, m = (
        out["realised-sunny daylight"]["default"],
        out["realised-sunny daylight"]["manual_heuristic"],
    )
    f_d, f_m = (
        out["forecast-sunny daylight"]["default"],
        out["forecast-sunny daylight"]["manual_heuristic"],
    )
    out["prediction_check"] = {
        "below_q05_over_8pct": d["below_q05"] > LOW_BOUND,
        "q50_bias_under_0.1": abs(d["q50_bias"]) < 0.1,
        "q50_bias_over_0.3": abs(d["q50_bias"]) > 0.3,
        "forecast_sunny_within_2pct_of_heuristic": f_d["score"] <= f_m["score"] * 1.02,
        "default_beats_heuristic_on_realised_sunny": d["score"] < m["score"],
        "realised_sunny_default_over_heuristic_ratio": d["score"] / m["score"],
    }
    common.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (common.REPORT_DIR / "stage0.json").write_text(json.dumps(out, indent=2, default=float))
    print(
        json.dumps(
            {k: out[k] for k in ("gate", "prediction_check", "sunny_by_stage")},
            indent=2,
            default=float,
        )
    )
    for g in ("realised-sunny daylight", "forecast-sunny daylight"):
        print(
            g,
            {
                a: (round(v["score"], 3), round(v["below_q05"], 1), round(v["q50_bias"], 3))
                for a, v in out[g].items()
                if a in arms
            },
        )


if __name__ == "__main__":
    main()
