"""Experiment 178: where the best GAM loses to BG2 on sunny daylight half-hours.

Joins the GAM's stored forecasts (experiment 164's fix O, arm A of experiment 169) with BG2's on the
same substation, run, and half-hour in the scored year, labels each row with its hour of day, the
clear-sky index of the ensemble-mean shortwave radiation, its lead band, and its substation, and
writes the gap tables to `work/report/report.json`.

uv run --no-sync python ml-experiments/00178-*/e00178_diagnose.py
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 178
SLUG: Final[str] = "dp-sunny-daylight-gap-diagnosis"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
E164_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00164-calibration-cold-night-tail-under-coverage"
    / "work"
    / "dp_preds"
)
E111_STORE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
    / "ens_store"
)
BG2_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
CSI_EDGES: Final[tuple[float, ...]] = (0.4, 0.7, 0.9)
CSI_LABELS: Final[tuple[str, ...]] = ("below 0.4", "0.4 to 0.7", "0.7 to 0.9", "0.9 and above")
LEAD_EDGES: Final[tuple[float, ...]] = (12.0, 36.0, 72.0)
LEAD_LABELS: Final[tuple[str, ...]] = ("0 to 12 h", "12 to 36 h", "36 to 72 h", "72 to 168 h")
N_DRAWS: Final[int] = 5000


def pinball(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles, in MW."""
    error = frame["actual"].to_numpy().astype(np.float64)[:, None] - frame.select(
        list(QUANTILES)
    ).to_numpy().astype(np.float64)
    return np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def one_site(*, site: str, cache, bg2: pl.DataFrame) -> pl.DataFrame:  # noqa: ANN001
    """Return the joined GAM and BG2 rows of one substation with their labels."""
    gam = pl.read_parquet(E164_DIR / f"o_scored_{site}.parquet").filter(
        ~pl.col("switching_mask") & pl.col("actual").is_finite()
    )
    key = ["power_fcst_init_time", "valid_time"]
    b = bg2.filter(pl.col("site") == site).select(
        *key, **{f"b_{c}": pl.col(c) for c in QUANTILES}, b_mean=pl.col("q50")
    )
    capacity = float(bg2.filter(pl.col("site") == site)["capacity_mw"][0])
    # The GAM stores percent of effective capacity and BG2 a fraction of it: convert both to MW.
    gam = gam.with_columns([pl.col(c) * capacity / 100.0 for c in (*QUANTILES, "actual")])
    b = b.with_columns([pl.col(c) * capacity for c in (*(f"b_{q}" for q in QUANTILES), "b_mean")])
    joined = gam.join(b, on=key, how="inner", validate="1:1")
    arrays = cache.sites[site]
    # `valid_time` is the end of the half-hour, as in the cache's `end_times`.
    index = np.searchsorted(
        cache.end_times, joined["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    if not (
        cache.end_times[index] == joined["valid_time"].dt.replace_time_zone(None).to_numpy()
    ).all():
        msg = f"{site}: valid_time is not on the cache's half-hour grid"
        raise ValueError(msg)
    elevation = arrays.elevation_deg[index]
    unit = arrays.pv_unit_shape[index]
    with np.load(E111_STORE / f"{site}.npz") as stored:
        origins = stored["origins"]
        shortwave = stored["shortwave"]
    run = joined["run"].to_numpy().astype(np.int64)
    origin = joined["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: run does not index the ensemble store"
        raise ValueError(msg)
    position = (
        (joined["valid_time"].dt.replace_time_zone(None).to_numpy() - origin)
        // np.timedelta64(30, "m")
    ).astype(np.int64) - 1
    if position.min() < 0 or position.max() >= shortwave.shape[2]:
        msg = f"{site}: a valid time lies outside the stored ensemble window"
        raise ValueError(msg)
    ens = np.nanmean(shortwave[run, :, position].astype(np.float64), axis=1)
    clear = haurwitz_w_m2(apparent_zenith_deg=90.0 - elevation)
    csi = np.where(clear > 20.0, ens / np.where(clear > 20.0, clear, 1.0), np.nan)
    local = joined["valid_time"].dt.convert_time_zone("Europe/London")
    gam_error = np.abs(joined["actual"].to_numpy() - joined["q50"].to_numpy())
    bg2_error = np.abs(joined["actual"].to_numpy() - joined["b_q50"].to_numpy())
    b_frame = joined.select("actual", **{c: pl.col(f"b_{c}") for c in QUANTILES})
    return joined.select("power_fcst_init_time", "valid_time", "lead_hours", "actual").with_columns(
        site=pl.lit(site),
        unit=pl.Series(unit),
        ghi=pl.Series(arrays.ghi_w_m2[index]),
        clear=pl.Series(clear),
        csi=pl.Series(csi),
        hour=((local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32) - 15) // 60),
        month=local.dt.strftime("%Y-%m"),
        gam_abs=pl.Series(gam_error),
        bg2_abs=pl.Series(bg2_error),
        gam_pin=pl.Series(pinball(frame=joined)),
        bg2_pin=pl.Series(pinball(frame=b_frame)),
        gam_signed=pl.Series(joined["q50"].to_numpy() - joined["actual"].to_numpy()),
        bg2_signed=pl.Series(joined["b_q50"].to_numpy() - joined["actual"].to_numpy()),
    )


def label_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the sunny-daylight flag and the level of each cut."""
    sunny_edge = {
        s: float(
            np.quantile(g.filter(pl.col("unit") > SUN_UP_UNIT_SHAPE)["unit"].to_numpy(), 2 / 3)
        )
        for (s,), g in frame.group_by("site")
    }
    edge = pl.col("site").replace_strict(sunny_edge, return_dtype=pl.Float64)
    csi_index = pl.col("csi").cut(list(CSI_EDGES), labels=list(CSI_LABELS), left_closed=True)
    lead_index = pl.col("lead_hours").cut(
        list(LEAD_EDGES), labels=list(LEAD_LABELS), left_closed=True
    )
    return frame.with_columns(
        sunny=pl.col("unit") >= edge,
        hour_bin=(pl.col("hour") // 2 * 2),
        csi_band=csi_index.cast(pl.String),
        lead_band=lead_index.cast(pl.String),
        gap_mae=pl.col("gam_abs") - pl.col("bg2_abs"),
        gap_pin=pl.col("gam_pin") - pl.col("bg2_pin"),
    )


def cut_table(*, frame: pl.DataFrame, column: str) -> list[dict]:
    """Return rows, mean scores, and share of the summed absolute-error gap per level of a cut."""
    total = frame["gap_mae"].sum()
    total_pin = frame["gap_pin"].sum()
    out = (
        frame.group_by(column)
        .agg(
            rows=pl.len(),
            gam_mae=pl.col("gam_abs").mean(),
            bg2_mae=pl.col("bg2_abs").mean(),
            gam_pin=pl.col("gam_pin").mean(),
            bg2_pin=pl.col("bg2_pin").mean(),
            gap_sum=pl.col("gap_mae").sum(),
            gap_pin_sum=pl.col("gap_pin").sum(),
            gam_bias=pl.col("gam_signed").mean(),
            bg2_bias=pl.col("bg2_signed").mean(),
        )
        .with_columns(
            row_share=pl.col("rows") / frame.height,
            gap_share=pl.col("gap_sum") / total,
            gap_pin_share=pl.col("gap_pin_sum") / total_pin,
        )
        .sort(column)
    )
    return out.to_dicts()


def month_block_interval(*, frame: pl.DataFrame, column: str, rng: np.random.Generator) -> dict:
    """Return the mean of a gap column and its 95% interval from resampling whole months."""
    monthly = frame.group_by("month").agg(total=pl.col(column).sum(), count=pl.len())
    totals = monthly["total"].to_numpy()
    counts = monthly["count"].to_numpy().astype(np.float64)
    picks = rng.integers(0, len(totals), size=(N_DRAWS, len(totals)))
    draws = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(totals.sum() / counts.sum()), "lower": float(low), "upper": float(high)}


def alternative_definitions(*, frame: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Return the all-lead gap under other definitions of a sunny daylight half-hour.

    Each definition takes the top third, per substation, of the daylight half-hours (unit shape
    above the sun-up limit) ranked by a different measure.
    """
    daylight = frame.filter(pl.col("unit") > SUN_UP_UNIT_SHAPE).with_columns(
        realised_csi=pl.col("ghi") / pl.col("clear")
    )
    measures = {
        "unit shape (the headline)": "unit",
        "realised irradiance (CAMS)": "ghi",
        "realised clear-sky index (CAMS over clear sky)": "realised_csi",
    }
    out = {}
    for label, column in measures.items():
        valid = daylight.filter(pl.col(column).is_finite() & (pl.col("clear") > 20.0))
        edge = pl.col(column).quantile(2 / 3).over("site")
        sunny = valid.filter(pl.col(column) >= edge)
        d5 = sunny.filter(pl.col("site") == "D5")
        rest = sunny.filter(pl.col("site") != "D5")
        out[label] = {
            "rows": sunny.height,
            "total": month_block_interval(frame=sunny, column="gap_mae", rng=rng),
            "d5_only": month_block_interval(frame=d5, column="gap_mae", rng=rng),
            "without_d5": month_block_interval(frame=rest, column="gap_mae", rng=rng),
            "d5_share": float(d5["gap_mae"].sum() / sunny["gap_mae"].sum()),
        }
    return out


def main() -> None:
    """Write the report."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    bg2 = pl.read_parquet(BG2_PATH).filter(pl.col("site").is_in(SITES))
    frame = label_rows(frame=pl.concat([one_site(site=s, cache=cache, bg2=bg2) for s in SITES]))
    rng = np.random.default_rng(178)
    report: dict = {"n_all_rows": frame.height}
    report["alternative_definitions"] = alternative_definitions(frame=frame, rng=rng)
    scopes = {
        "all_leads": frame,
        "lead_12_to_36": frame.filter(pl.col("lead_hours").is_between(12.0, 36.0, closed="left")),
    }
    for name, scope in scopes.items():
        for label, sites in (("five", SITES[:5]), ("six", SITES)):
            rows = scope.filter(pl.col("site").is_in(sites))
            sunny = rows.filter(pl.col("sunny"))
            entry = {
                "n_sunny": sunny.height,
                "n_other_daylight": rows.filter(
                    ~pl.col("sunny") & (pl.col("unit") > SUN_UP_UNIT_SHAPE)
                ).height,
                "sunny_gap_mae": month_block_interval(frame=sunny, column="gap_mae", rng=rng),
                "sunny_gap_pin": month_block_interval(frame=sunny, column="gap_pin", rng=rng),
                "sunny_gam_mae": float(sunny["gam_abs"].mean()),  # ty: ignore[invalid-argument-type]
                "sunny_bg2_mae": float(sunny["bg2_abs"].mean()),  # ty: ignore[invalid-argument-type]
                "sunny_gam_pin": float(sunny["gam_pin"].mean()),  # ty: ignore[invalid-argument-type]
                "sunny_bg2_pin": float(sunny["bg2_pin"].mean()),  # ty: ignore[invalid-argument-type]
                "other_daylight_gap_mae": float(
                    rows.filter(~pl.col("sunny") & (pl.col("unit") > SUN_UP_UNIT_SHAPE))[  # ty: ignore[invalid-argument-type]
                        "gap_mae"
                    ].mean()
                ),
                "night_gap_mae": float(
                    rows.filter(pl.col("unit") <= SUN_UP_UNIT_SHAPE)["gap_mae"].mean()  # ty: ignore[invalid-argument-type]
                ),
            }
            for cut in ("hour_bin", "csi_band", "lead_band", "site"):
                entry[cut] = cut_table(frame=sunny.drop_nulls(cut), column=cut)
            entry["without_d5"] = {
                "n": sunny.filter(pl.col("site") != "D5").height,
                "gap_mae": month_block_interval(
                    frame=sunny.filter(pl.col("site") != "D5"), column="gap_mae", rng=rng
                ),
                "gap_pin": month_block_interval(
                    frame=sunny.filter(pl.col("site") != "D5"), column="gap_pin", rng=rng
                ),
            }
            entry["d5_only"] = {
                "gap_mae": month_block_interval(
                    frame=sunny.filter(pl.col("site") == "D5"), column="gap_mae", rng=rng
                ),
                "gap_pin": month_block_interval(
                    frame=sunny.filter(pl.col("site") == "D5"), column="gap_pin", rng=rng
                ),
            }
            for cut in ("hour_bin", "csi_band", "lead_band"):
                entry[f"d5_{cut}"] = cut_table(
                    frame=sunny.filter(pl.col("site") == "D5").drop_nulls(cut), column=cut
                )
                entry[f"not_d5_{cut}"] = cut_table(
                    frame=sunny.filter(pl.col("site") != "D5").drop_nulls(cut), column=cut
                )
            report[f"{name}_{label}"] = entry
    # The gap per site beside the mean sunny-noon fall: a proxy for the PV share.
    shares = (
        frame.filter(pl.col("sunny"))
        .group_by("site")
        .agg(mean_actual=pl.col("actual").mean())
        .join(
            frame.filter(pl.col("unit") <= SUN_UP_UNIT_SHAPE)
            .group_by("site")
            .agg(night_actual=pl.col("actual").mean()),
            on="site",
        )
        .with_columns(sunny_minus_night=pl.col("mean_actual") - pl.col("night_actual"))
    )
    report["site_actuals"] = shares.to_dicts()
    # Distribution of the ensemble clear-sky index on sunny daylight rows.
    report["csi_quantiles"] = {
        str(q): float(np.nanquantile(frame.filter(pl.col("sunny"))["csi"].to_numpy(), q))
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }
    out = WORK_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    frame.write_parquet(out / "joined_rows.parquet")
    for name in ("all_leads_five", "all_leads_six", "lead_12_to_36_five", "lead_12_to_36_six"):
        e = report[name]
        print(
            name,
            e["n_sunny"],
            e["sunny_gam_mae"],
            e["sunny_bg2_mae"],
            e["sunny_gap_mae"],
            e["sunny_gap_pin"],
        )


if __name__ == "__main__":
    main()
