# ruff: noqa: E501
r"""Stage 3: does the sunshine graft of experiment 316 replicate in April and June 2025?

Repeats experiment 316's level-matched graft. The refit is experiment 305's depth-1 XGBoost twin
refitted on 2025-04-01 and 2025-06-01 (`e00322_refit.py`); arm D is experiment 180's stored
unclipped arm D for the replicate period (0.5 to 168 hours, the longest lead stored). Each family's
contribution of the refit and of a depth-1 surrogate of arm D's median is averaged to zero, and the
difference is added to all 11 of arm D's quantiles, as in experiment 316. The share S is the part of
the refit's advantage over arm D that the graft recovers. The replicate period has no stored BG2, so
the worst quarter of daylight half-hours is chosen by arm D's own quantile score; the same
selection is applied to the scored year (experiment 316's stored frame) as a bridge.

    PYTHONPATH=<folders of experiments 305, 316, 322> OMP_NUM_THREADS=4 \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00322-*/e00322_graft.py
"""

from typing import Final

import e00305_analyse as an  # ty: ignore[unresolved-import]
import e00316_common as c316  # ty: ignore[unresolved-import]
import e00316_stage1 as s316  # ty: ignore[unresolved-import]
import e00322_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
TARGETS: Final[tuple[str, ...]] = ("D5", "D9")
MONTHS: Final[tuple[str, ...]] = ("2025-04", "2025-06")
KEYS: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
E180_DREPLICATE: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00180-dp-sunny-daylight-fix-d5-solar-shape" / "work" / "arms" / "c" / "dp_preds_s2"
)  # fmt: skip
ZENITH_DAYLIGHT_DEG: Final[float] = 85.0
"""Daylight means the sun is more than 5 degrees above the horizon."""
SHARE_FAMILIES: Final[tuple[str, ...]] = (*an.FAMILIES, "solar+lags+calendar", "all five")


def replicate_frame() -> pl.DataFrame:
    """Return arm D's quantiles, the refit's quantiles, and the inputs for April and June 2025."""
    pieces = []
    for site in SITES:
        arm = (
            pl.read_parquet(E180_DREPLICATE / f"d_replicate_{site}.parquet")
            .filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
            .select(*KEYS, "lead_hours", "actual", *common.c320.QUANTILES)
            .rename({q: f"d_{q}" for q in common.c320.QUANTILES})
        )
        for month in MONTHS:
            refit = pl.read_parquet(common.WORK_DIR / "refit" / "preds" / f"{site}_{month}.parquet")
            joined = arm.join(refit, on=KEYS, how="inner", validate="1:1").with_columns(
                site=pl.lit(site), init_month=pl.lit(month)
            )
            pieces.append(joined)
    frame = pl.concat(pieces).filter(pl.col("lead_hours") <= 168.0)
    q_d = frame.select([f"d_{q}" for q in common.c320.QUANTILES]).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        med_d=pl.Series(q_d[:, 5]),
        qs_d=pl.Series(c316.qs_per_row(actual=actual, q=q_d)),
        week=pl.col("power_fcst_init_time").dt.iso_year() * 100
        + pl.col("power_fcst_init_time").dt.week(),
    )


def worst_quarter(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return the quarter of a site's daylight half-hours where arm D's own quantile score is highest."""
    day = frame.filter((pl.col("site") == site) & (pl.col("solar_zenith") < ZENITH_DAYLIGHT_DEG))
    cut = day["qs_d"].quantile(0.75)
    return day.filter(pl.col("qs_d") >= cut)


def shares(*, rows: pl.DataFrame) -> dict:
    """Return the level-matched graft share of each family on `rows`, with 95% week-block intervals."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    q_d = rows.select([f"d_{q}" for q in common.c320.QUANTILES]).to_numpy().astype(np.float64)
    q_r = np.sort(
        rows.select([f"r_{q}" for q in common.c320.QUANTILES]).to_numpy().astype(np.float64), axis=1
    )
    qs_d, qs_r = c316.qs_per_row(actual=actual, q=q_d), c316.qs_per_row(actual=actual, q=q_r)
    blocks = rows["week"].to_numpy()
    entry: dict = {
        "rows": rows.height,
        "weeks": len(np.unique(blocks)),
        "qs_d": float(qs_d.mean()),
        "qs_refit": float(qs_r.mean()),
        "S_level_matched": {},
    }
    for graft in SHARE_FAMILIES:
        families = s316.GRAFTS[graft]
        delta = sum(rows[f"crc_{f}"].to_numpy() - rows[f"csc_{f}"].to_numpy() for f in families)
        qs_g = c316.qs_per_row(actual=actual, q=q_d + delta[:, None])
        entry["S_level_matched"][graft] = {
            **c316.ratio_interval(num=qs_d - qs_g, den=qs_d - qs_r, blocks=blocks, seed=11),
            "qs_graft": float(qs_g.mean()),
        }
    return entry


def subset_report(*, frame: pl.DataFrame) -> dict:
    """Return the share table for the pooled targets, each site, and the placebos."""
    out: dict = {}
    sites = [s for s in SITES if s in set(frame["site"].to_list())]
    parts = {s: worst_quarter(frame=frame, site=s) for s in sites}
    out["D5+D9 worst quarter"] = shares(rows=pl.concat([parts[s] for s in TARGETS]))
    for s in sites:
        out[f"{s} worst quarter"] = shares(rows=parts[s])
    return out


def main() -> None:
    """Write the stage 3 report: the replicate year, and the scored-year bridge."""
    common.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    c316.E305_DIR = common.WORK_DIR / "as305"  # models/additive_post_hoc/{site}_{month}.ubj
    c316.REPORT_DIR = common.REPORT_DIR / "stage3_surrogate"
    frame = s316.contributions(frame=replicate_frame())
    report: dict = {
        "replicate_year_2025": subset_report(frame=frame),
        "surrogate_stats_file": str(c316.REPORT_DIR / "stage1_surrogate.json"),
    }
    scored = pl.read_parquet(c316.WORK_DIR / "stage1_frame.parquet")
    report["scored_year_bridge_same_selection"] = subset_report(frame=scored)
    common.write_json_atomic(path=common.REPORT_DIR / "stage3.json", content=report)
    for name, section in report.items():
        if not isinstance(section, dict) or name == "surrogate_stats_file":
            continue
        for subset, entry in section.items():
            print(
                name, subset, entry["rows"], round(entry["qs_d"], 3), round(entry["qs_refit"], 3),
                {k: round(v["est"], 2) for k, v in entry["S_level_matched"].items()},
                flush=True,
            )  # fmt: skip


if __name__ == "__main__":
    main()
