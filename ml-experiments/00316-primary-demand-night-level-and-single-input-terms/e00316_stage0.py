# ruff: noqa: PLR0915
"""Stage 0 of experiment 316: is the band heads' gain at D10 a night-time level error?

Zero-fit. Reads the stored forecasts of experiments 251 (Primary demand research default 1, the
rows), 311 (arm M, the mixture head) and 314 (arm P, the pinball head), splits D10's gain, and
scores an as-of night offset applied to the default with no head.

    uv run --no-sync python ml-experiments/00316-*/e00316_stage0.py
"""

from datetime import datetime, timedelta
from typing import Final

import e00316_common as common
import numpy as np
import polars as pl

SITE: Final[str] = "D10"
NIGHT_NIGHTS: Final[int] = 28
HEADS: Final[tuple[str, ...]] = ("P", "M")
"""Arm P is the pre-registered head; arm M is reported beside it."""


def night_mask(*, valid_time: pl.Series) -> np.ndarray:
    """Return True for half-hours from 21:00 to 07:30 UTC inclusive."""
    hh = (valid_time.dt.hour() * 2 + valid_time.dt.minute() // 30).to_numpy()
    return (hh >= 42) | (hh <= 15)


def as_of_offsets(*, rows: pl.DataFrame, default_q: np.ndarray) -> np.ndarray:
    """Return, per row, the median residual of the previous 28 nights as known at the run's init.

    Each half-hour's residual comes from its shortest-lead forecast of the default, so it uses only
    actuals observed before the run initialised. Night N runs from 21:00 UTC on day N to 07:30 UTC
    on day N + 1 and counts once day N + 1 has begun at the run's 09:00 UTC initialisation time.
    """
    site = (
        rows.with_columns(median=pl.Series(default_q[:, common.MEDIAN_INDEX]))
        .filter(pl.col("site") == SITE)
        .sort(["valid_time", "lead_hours"])
        .group_by("valid_time", maintain_order=True)
        .first()
    )
    night = site.filter(night_mask(valid_time=site["valid_time"])).with_columns(
        residual=pl.col("actual").cast(pl.Float64) - pl.col("median"),
        night=(pl.col("valid_time") - pl.duration(hours=21)).dt.date(),
    )
    per_night = {d: g["residual"].to_numpy() for (d,), g in night.group_by("night")}
    offset_by_run: dict[datetime, float] = {}
    for init in rows.filter(pl.col("site") == SITE)["power_fcst_init_time"].unique().to_list():
        pool = [
            per_night[d]
            for k in range(1, NIGHT_NIGHTS + 1)
            if (d := init.date() - timedelta(days=k)) in per_night
        ]
        offset_by_run[init] = float(np.median(np.concatenate(pool))) if pool else 0.0
    out = np.zeros(rows.height)
    is_site = (rows["site"] == SITE).to_numpy()
    init_all = rows["power_fcst_init_time"].to_list()
    for i in np.flatnonzero(is_site):
        out[i] = offset_by_run[init_all[i]]
    return out


def main() -> None:
    """Write the stage 0 report."""
    rows = pl.read_parquet(common.E251_SCORED / "rows.parquet")
    default_q = np.load(common.E251_SCORED / "arm1.npy").astype(np.float64)
    arms = {"default": default_q}
    arms["P"] = np.load(common.E314_SCORED / "P.npy").astype(np.float64)
    arms["M"] = np.load(common.E314_SCORED / "M.npy").astype(np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    d10 = site == SITE
    night = night_mask(valid_time=rows["valid_time"])
    blocks = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    qs = {name: common.qs_per_row(actual=actual, q=q) for name, q in arms.items()}
    report: dict = {
        "gate_default_six": float(qs["default"].mean()),
        "d10_rows": int(d10.sum()),
        "d10_night_rows": int((d10 & night).sum()),
    }

    # Counterfactual arms: only the head's centre shift, or only its width change.
    for head in HEADS:
        q = arms[head]
        base = default_q
        centre_shift = ((q[:, 1] + q[:, 9]) - (base[:, 1] + base[:, 9])) / 2.0
        width_change = (q[:, 9] - q[:, 1]) - (base[:, 9] - base[:, 1])
        arms[f"{head}_centre_only"] = base + centre_shift[:, None]
        scale = np.where(
            (base[:, 9] - base[:, 1]) > 0, (q[:, 9] - q[:, 1]) / (base[:, 9] - base[:, 1]), 1.0
        )
        centre = base[:, [5]]
        arms[f"{head}_width_only"] = centre + (base - centre) * scale[:, None]
        qs[f"{head}_centre_only"] = common.qs_per_row(actual=actual, q=arms[f"{head}_centre_only"])
        qs[f"{head}_width_only"] = common.qs_per_row(actual=actual, q=arms[f"{head}_width_only"])
        report[f"{head}_band"] = {
            "centre_shift_night": float(centre_shift[d10 & night].mean()),
            "centre_shift_day": float(centre_shift[d10 & ~night].mean()),
            "width_change_night": float(width_change[d10 & night].mean()),
            "width_change_day": float(width_change[d10 & ~night].mean()),
            "median_moved": bool(not np.array_equal(q[:, 5], base[:, 5])),
        }

    # As-of night offset on the default, no head.
    offset = as_of_offsets(rows=rows, default_q=default_q)
    applied = np.where((d10 & night)[:, None], offset[:, None], 0.0)
    arms["offset"] = default_q + applied
    qs["offset"] = common.qs_per_row(actual=actual, q=arms["offset"])
    report["offset"] = {
        "mean_night": float(offset[d10 & night].mean()),
        "sd_night": float(offset[d10 & night].std()),
    }

    # D10's gain per arm and the shares asked for.
    gain = {name: qs["default"] - v for name, v in qs.items() if name != "default"}
    for name, g in gain.items():
        report.setdefault("d10_gain", {})[name] = {
            "all": common.mean_interval(values=g[d10], blocks=blocks[d10], seed=1),
            "night": common.mean_interval(
                values=g[d10 & night], blocks=blocks[d10 & night], seed=2
            ),
            "day": common.mean_interval(
                values=g[d10 & ~night], blocks=blocks[d10 & ~night], seed=3
            ),
        }
        report.setdefault("other_five_gain", {})[name] = common.mean_interval(
            values=g[~d10], blocks=blocks[~d10], seed=4
        )
    for head in HEADS:
        g = gain[head]
        num_night = np.where(d10 & night, g, 0.0)
        all_d10 = np.where(d10, g, 0.0)
        report.setdefault("S0a_night_share_of_gain", {})[head] = common.ratio_interval(
            num=num_night, den=all_d10, blocks=blocks, seed=5
        )
        report.setdefault("S0b_offset_share_of_gain", {})[head] = common.ratio_interval(
            num=np.where(d10, gain["offset"], 0.0), den=all_d10, blocks=blocks, seed=6
        )
        for part in ("centre_only", "width_only"):
            report.setdefault("centre_width_share", {})[f"{head}_{part}"] = common.ratio_interval(
                num=np.where(d10, gain[f"{head}_{part}"], 0.0), den=all_d10, blocks=blocks, seed=7
            )
    # Splits by local hour and by quantile level for the pinball head.
    local_hour = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
    for head in HEADS:
        report.setdefault("by_local_hour", {})[head] = {
            int(h): float(gain[head][d10 & (local_hour == h)].mean()) for h in range(24)
        }
        pin_default = np.maximum(
            common.LEVELS[None, :] * (actual[:, None] - default_q),
            (common.LEVELS[None, :] - 1.0) * (actual[:, None] - default_q),
        )
        pin_head = np.maximum(
            common.LEVELS[None, :] * (actual[:, None] - arms[head]),
            (common.LEVELS[None, :] - 1.0) * (actual[:, None] - arms[head]),
        )
        diff = 2.0 * (pin_default - pin_head) / len(common.LEVELS)
        report.setdefault("by_level", {})[head] = {
            "night": {
                n: float(diff[d10 & night, k].mean()) for k, n in enumerate(common.QUANTILES)
            },
            "day": {n: float(diff[d10 & ~night, k].mean()) for k, n in enumerate(common.QUANTILES)},
        }
    # Night level error of the default at D10 by UTC hour, for the figure.
    resid = actual - default_q[:, common.MEDIAN_INDEX]
    hour_utc = rows["valid_time"].dt.hour().to_numpy()
    report["d10_default_residual_by_utc_hour"] = {
        int(h): float(resid[d10 & (hour_utc == h)].mean()) for h in range(24)
    }
    report["absolute_d10_qs"] = {name: float(v[d10].mean()) for name, v in qs.items()}
    common.write_json_atomic(path=common.REPORT_DIR / "stage0.json", content=report)
    np.save(common.WORK_DIR / "stage0_gain_P.npy", gain["P"])
    print(report["S0a_night_share_of_gain"], report["S0b_offset_share_of_gain"])
    print(report["absolute_d10_qs"], report["gate_default_six"])


if __name__ == "__main__":
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    main()
