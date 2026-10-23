"""Collect the numbers the write-up quotes: share explained by each correction, with intervals.

Reads `arms_scored.parquet` (step 3) and experiment 00261's scored cases (the ERA5 oracle's median),
and writes `report.json` in `work/`. Intervals resample the 12 months of the scored year (by the
month of the run) 5,000 times; the farms are not resampled.

    uv run --no-sync python ml-experiments/00307-*/e00307_report.py
"""

import json
from typing import Final

import e00307_common as common
import numpy as np
import polars as pl
from e00307_corrections import ARMS
from e00307_ensemble import PRIOR_261

RESAMPLES: Final[int] = 5_000
SEED: Final[int] = 307
LEAD_BANDS: Final[dict[str, tuple[int, int]]] = {
    "days 0-2": (0, 2),
    "days 3-6": (3, 6),
    "days 7+": (7, 99),
}


def monthly(*, frame: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-month rows below the median and rows, over the 12 run months."""
    grouped = (
        frame.with_columns(
            below=(pl.col("actual_norm") < pl.col(column)).cast(pl.Float64),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
        .group_by("month")
        .agg(below=pl.col("below").sum(), n=pl.len())
        .sort("month")
    )
    return grouped["below"].to_numpy(), grouped["n"].to_numpy().astype(float)


def interval(*, draws: np.ndarray) -> list[float]:
    """Return the 2.5th and 97.5th percentiles."""
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    """Write `report.json`."""
    arms = pl.read_parquet(common.WORK_DIR / "arms_scored.parquet")
    oracle = pl.read_parquet(PRIOR_261 / "cases_scored.parquet").select(
        *common.KEY_COLUMNS, "q50_oracle", "q50_mean_wind", "q50_stored_r1"
    )
    arms = arms.join(oracle, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
    rng = np.random.default_rng(SEED)
    out: dict = {"rows_all": arms.height, "rows_kept": arms.filter(~pl.col("masked")).height}
    sets = {"kept": arms.filter(~pl.col("masked")), "all": arms}
    columns = {
        **{arm: f"q50_{arm}" for arm in ARMS},
        "oracle": "q50_oracle",
        "mean_wind": "q50_mean_wind",
    }
    for set_name, frame in sets.items():
        stats = {name: monthly(frame=frame, column=col) for name, col in columns.items()}
        months = stats["raw"][1].size
        pick = rng.integers(0, months, size=(RESAMPLES, months))
        shares = {
            name: (below[pick].sum(axis=1) / n[pick].sum(axis=1)) * 100
            for name, (below, n) in stats.items()
        }
        point = {name: float(below.sum() / n.sum() * 100) for name, (below, n) in stats.items()}
        gap = shares["oracle"] - shares["raw"]
        out[set_name] = {
            "share": {
                n: {"point": point[n], "interval": interval(draws=shares[n])} for n in columns
            },
            "gain_points": {
                n: {
                    "point": point[n] - point["raw"],
                    "interval": interval(draws=shares[n] - shares["raw"]),
                }
                for n in columns
                if n != "raw"
            },
            "share_of_gap": {
                n: {
                    "point": (point[n] - point["raw"]) / (point["oracle"] - point["raw"]) * 100,
                    "interval": interval(draws=(shares[n] - shares["raw"]) / gap * 100),
                }
                for n in columns
                if n not in ("raw", "oracle")
            },
            "gap_points": point["oracle"] - point["raw"],
        }
        out[set_name]["contrasts_points"] = {
            f"{a} minus {b}": {
                "point": point[a] - point[b],
                "interval": interval(draws=shares[a] - shares[b]),
            }
            for a, b in (
                ("a_mean", "a_mean_era5"),
                ("e_lead", "e_lead_era5"),
                ("a_mean_era5_all", "a_mean_era5"),
                ("a_mean", "e_lead"),
                ("d_direction", "a_mean"),
            )
        }
        score_columns = {arm: f"score_{arm}" for arm in ARMS}
        out[set_name]["score_points"] = {
            arm: float(frame[col].to_numpy().mean() * common.POINTS_PER_FRACTION)
            for arm, col in score_columns.items()
        }
        by_farm, by_band = {}, {}
        for farm in sorted(frame["site"].unique().to_list()):
            part = frame.filter(pl.col("site") == farm)
            by_farm[farm] = {
                name: float((part["actual_norm"] < part[col]).to_numpy().mean() * 100)
                for name, col in columns.items()
            }
        for band, (low, high) in LEAD_BANDS.items():
            part = frame.filter(pl.col("lead_day").is_between(low, high))
            by_band[band] = {
                name: float((part["actual_norm"] < part[col]).to_numpy().mean() * 100)
                for name, col in columns.items()
            }
        out[set_name]["by_farm"] = by_farm
        out[set_name]["by_lead_band"] = by_band
        out[set_name]["median_mean_q50_points"] = {
            name: float(frame[col].to_numpy().mean() * common.POINTS_PER_FRACTION)
            for name, col in columns.items()
        }
    kept = sets["kept"]
    out["check_raw_vs_261_stored_r1_max_abs_median_gap"] = float(
        np.abs(kept["q50_raw"].to_numpy() - kept["q50_stored_r1"].to_numpy()).max()
    )
    (common.WORK_DIR / "report.json").write_text(json.dumps(out, indent=2))
    for set_name in ("kept", "all"):
        print(set_name, "gap to oracle points:", round(out[set_name]["gap_points"], 2))
        for name in columns:
            share = out[set_name]["share"][name]
            frac = out[set_name]["share_of_gap"].get(name)
            print(
                f"  {name:16s} share {share['point']:5.1f}"
                f" [{share['interval'][0]:.1f},{share['interval'][1]:.1f}]"
                + (
                    f"  closes {frac['point']:5.1f}%"
                    f" [{frac['interval'][0]:.1f},{frac['interval'][1]:.1f}]"
                    if frac
                    else ""
                )
                + (f"  score {out[set_name]['score_points'].get(name, float('nan')):.3f}")
            )
    print("check", out["check_raw_vs_261_stored_r1_max_abs_median_gap"])
    print(json.dumps(out["kept"]["by_lead_band"], indent=1))


if __name__ == "__main__":
    main()
