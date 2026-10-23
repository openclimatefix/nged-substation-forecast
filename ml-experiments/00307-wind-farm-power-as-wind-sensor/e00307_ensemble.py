"""Step 2: where the ensemble's 100 m wind departs from the wind the farms' power implies.

Experiment 00261's per-case tables (ensemble-mean, 10th, and 90th percentile 100 m speed, and
ERA5's 100 m speed, at every case of the hindcast and scored sets) are joined to the implied 100 m
wind of `e00307_implied.py`. ERA5 against the implied wind, on the same rows, is the control.

Output, in `work/`: `cases_<set>.parquet` (cases with the implied wind) and `ensemble.json`.

    systemd-run --user --scope -p MemoryMax=6G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00307-*/e00307_ensemble.py
"""

import json
from pathlib import Path
from typing import Final

import e00307_common as common
import numpy as np
import polars as pl
from e00307_implied import SPEED_BINS, direction_sector

PRIOR_261: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR / "00261-wind-low-bias-decomposition" / "work"
)
SETS: Final[tuple[str, ...]] = ("scored", "hindcast")
SHIFTS_HALF_HOURS: Final[tuple[int, ...]] = tuple(range(-6, 7))
Z_90: Final[float] = 2.5631
"""The 10th-to-90th percentile range of a normal distribution, in standard deviations."""


def load(*, name: str) -> pl.DataFrame:
    """Return the cases of one set with the implied 100 m wind joined on."""
    implied = pl.read_parquet(common.WORK_DIR / "implied.parquet").select(
        "time_series_id", "valid_time", "implied_100m", "outage", "sector", "era5_dir_sin"
    )
    cases = pl.read_parquet(PRIOR_261 / f"cases_{name}.parquet")
    return (
        cases.join(implied, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            era5_bin=pl.col("era5_speed_100m")
            .cut(list(SPEED_BINS[1:-1]), left_closed=True)
            .cast(pl.String),
            ens_bin=pl.col("ens_speed_100m")
            .cut(list(SPEED_BINS[1:-1]), left_closed=True)
            .cast(pl.String),
            spread=(pl.col("ens_speed_100m_p90") - pl.col("ens_speed_100m_p10")) / Z_90,
        )
        .with_columns(
            ens_sector=pl.struct("era5_dir_sin", "era5_dir_cos").map_batches(
                lambda s: pl.Series(
                    direction_sector(
                        sin=s.struct.field("era5_dir_sin").to_numpy(),
                        cos=s.struct.field("era5_dir_cos").to_numpy(),
                    )
                )
            )
        )
    )


def stats(*, frame: pl.DataFrame) -> dict:
    """Return the ensemble-against-implied statistics of the rows with an implied wind."""
    inv = frame.filter(pl.col("implied_100m").is_not_null())
    if inv.height == 0:
        return {"rows": 0}
    implied = inv["implied_100m"]
    ens = inv["ens_speed_100m"]
    era5 = inv["era5_speed_100m"]
    return {
        "rows": inv.height,
        "ens_minus_implied": float((ens - implied).to_numpy().mean()),
        "era5_minus_implied": float((era5 - implied).to_numpy().mean()),
        "ens_rmse": float(((ens - implied) ** 2).to_numpy().mean() ** 0.5),
        "era5_rmse": float(((era5 - implied) ** 2).to_numpy().mean() ** 0.5),
        "ens_p10_minus_implied": float((inv["ens_speed_100m_p10"] - implied).to_numpy().mean()),
        "ens_p90_minus_implied": float((inv["ens_speed_100m_p90"] - implied).to_numpy().mean()),
        "share_implied_below_p10": float((implied < inv["ens_speed_100m_p10"]).to_numpy().mean()),
        "share_implied_above_p90": float((implied > inv["ens_speed_100m_p90"]).to_numpy().mean()),
        "member_spread_mean": float(inv["spread"].to_numpy().mean()),
        "implied_minus_mean_std": float((implied - ens).to_numpy().std(ddof=1)),
        "era5_minus_implied_std": float((era5 - implied).to_numpy().std(ddof=1)),
    }


def timing(*, frame: pl.DataFrame) -> dict:
    """Return RMSE and mean difference of implied wind against the ensemble mean shifted in time."""
    ens = frame.select(
        "time_series_id", "power_fcst_init_time", "valid_time", shifted="ens_speed_100m"
    )
    base = frame.filter(pl.col("implied_100m").is_not_null() & ~pl.col("masked")).select(
        "time_series_id", "power_fcst_init_time", "valid_time", "implied_100m"
    )
    out = {}
    for shift in SHIFTS_HALF_HOURS:
        moved = ens.with_columns(valid_time=pl.col("valid_time") - pl.duration(minutes=30 * shift))
        joined = base.join(
            moved, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
        ).drop_nulls()
        diff = joined["shifted"] - joined["implied_100m"]
        out[str(shift * 0.5)] = {
            "rows": joined.height,
            "mean_diff": float(diff.to_numpy().mean()),
            "rmse": float((diff**2).to_numpy().mean() ** 0.5),
            "correlation": float(np.corrcoef(joined["shifted"], joined["implied_100m"])[0, 1]),
        }
    return out


def main() -> None:
    """Write the cases with implied wind and the grouped statistics."""
    out: dict = {}
    for name in SETS:
        cases = load(name=name)
        cases.write_parquet(common.WORK_DIR / f"cases_{name}.parquet")
        usable = cases.filter(~pl.col("masked"))
        entry: dict = {"all": stats(frame=usable)}
        entry["by_lead_day"] = {
            int(d): stats(frame=usable.filter(pl.col("lead_day") == d))
            for d in sorted(usable["lead_day"].unique().to_list())
            if d <= 9
        }
        entry["by_era5_bin"] = {
            b: stats(frame=usable.filter(pl.col("era5_bin") == b))
            for b in usable["era5_bin"].drop_nulls().unique().to_list()
        }
        entry["by_ens_bin"] = {
            b: stats(frame=usable.filter(pl.col("ens_bin") == b))
            for b in usable["ens_bin"].drop_nulls().unique().to_list()
        }
        entry["by_sector"] = {
            int(s): stats(frame=usable.filter(pl.col("sector") == s)) for s in range(8)
        }
        entry["by_farm"] = {
            f: stats(frame=usable.filter(pl.col("site") == f))
            for f in sorted(usable["site"].unique().to_list())
        }
        entry["timing"] = timing(frame=cases)
        out[name] = entry
        print(name, json.dumps(entry["all"], default=float))
        for d, v in entry["by_lead_day"].items():
            print("  day", d, {k: round(x, 3) for k, x in v.items()})
        for s, v in entry["timing"].items():
            print("  shift h", s, {k: round(x, 3) for k, x in v.items()})
    (common.WORK_DIR / "ensemble.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
