# ruff: noqa: E501, PLR0915
"""Experiment 210: idle floor, dip, H0 test, outage rows, and the ERA5 wind-speed detector.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python -I ml-experiments/00210-wind-idle-noise-mva-artefact/e00210_analysis.py

Reads `joined.parquet` (written by `e00210_data.py`) and writes tables and the per-row mask under
`data/ml-experiments/00210-wind-idle-noise-mva-artefact/`.
"""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

OUT_DIR: Final[Path] = Path("data/ml-experiments/00210-wind-idle-noise-mva-artefact")
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
UNITS: Final[dict[str, str]] = {"W1": "MVA", "W2": "MW", "W3": "MVA"}
BIN_WIDTH: Final[float] = 0.5
IDLE_WIND_MAX: Final[float] = 2.0
DIP_BINS: Final[tuple[float, float]] = (2.0, 5.0)
DEFAULT_CUT_IN: Final[float] = 3.0
CUT_IN_RANGE: Final[tuple[float, float]] = (2.5, 4.5)
OUTAGE_MIN_HALF_HOURS: Final[int] = 24
MAX_FALSE_DROP_RATE: Final[float] = 0.05
FALSE_DROP_MARGIN_OF_CAPACITY: Final[float] = 0.02
NAN: Final[float] = float("nan")
THRESHOLD_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 8.01, 0.1), 1)


def to_float(value: Any) -> float:
    """Convert a Polars or NumPy scalar to a Python float."""
    return float(value)


def with_bins(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the 0.5 m/s wind bin's lower edge as `bin_lo`."""
    return frame.with_columns(bin_lo=(pl.col("wind_speed") / BIN_WIDTH).floor() * BIN_WIDTH)


def idle_floor(*, frame: pl.DataFrame) -> dict[str, float]:
    """Summarise power at ERA5 wind below 2 m/s."""
    idle = frame.filter(pl.col("wind_speed") < IDLE_WIND_MAX).sort("valid_time")
    power = idle["power"]
    contiguous = idle["valid_time"].diff().dt.total_minutes() == 30
    repeated = (power.diff() == 0) & contiguous
    return {
        "n": idle.height,
        "median": to_float(power.median()),
        "p05": to_float(power.quantile(0.05)),
        "p95": to_float(power.quantile(0.95)),
        "share_exact_zero": to_float((power == 0).mean()),
        "share_repeated_value": to_float(repeated.mean()),
        "share_negative": to_float((power < 0).mean()),
        "min": to_float(power.min()),
    }


def bin_table(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Median, 5% and 95% of power per 0.5 m/s wind bin."""
    return (
        with_bins(frame=frame)
        .group_by("bin_lo")
        .agg(
            pl.len().alias("n"),
            pl.col("power").median().alias("median"),
            pl.col("power").quantile(0.05).alias("p05"),
            pl.col("power").quantile(0.95).alias("p95"),
        )
        .filter(pl.col("n") >= 30)
        .sort("bin_lo")
    )


def estimate_cut_in(*, bins: pl.DataFrame, idle_p95: float) -> tuple[float, float]:
    """Return (cut-in used, empirical estimate): lowest bin whose median exceeds idle p95."""
    above = bins.filter((pl.col("bin_lo") >= IDLE_WIND_MAX) & (pl.col("median") > idle_p95))
    estimate = to_float(above["bin_lo"].min()) if above.height else NAN
    if not (CUT_IN_RANGE[0] <= estimate <= CUT_IN_RANGE[1]):
        return DEFAULT_CUT_IN, estimate
    return estimate, estimate


def outage_runs(*, frame: pl.DataFrame, wind_min: float, idle_p95: float) -> pl.DataFrame:
    """Return runs of at least 24 contiguous half-hours at wind > wind_min and power <= idle p95."""
    flagged = frame.sort("valid_time").with_columns(
        flag=(pl.col("wind_speed") > wind_min) & (pl.col("power") <= idle_p95),
        contiguous=pl.col("valid_time").diff().dt.total_minutes() == 30,
    )
    # A run breaks when the flag drops or the readings are not contiguous.
    flagged = flagged.with_columns(
        run_id=((~pl.col("flag")) | (~pl.col("contiguous").fill_null(False))).cum_sum()
    )
    runs = (
        flagged.filter(pl.col("flag"))
        .group_by("run_id")
        .agg(
            pl.len().alias("n_half_hours"),
            pl.col("valid_time").min().alias("start"),
            pl.col("valid_time").max().alias("end"),
            pl.col("wind_speed").median().alias("median_wind"),
        )
        .filter(pl.col("n_half_hours") >= OUTAGE_MIN_HALF_HOURS)
        .sort("start")
    )
    return runs  # noqa: RET504


def detector_rates(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    capacity: float,
) -> dict[str, float]:
    """Choose a wind threshold on `train`, score it on `test`."""
    idle_train = idle_floor(frame=train)
    idle_hi = idle_train["p95"]
    well_above = idle_hi + FALSE_DROP_MARGIN_OF_CAPACITY * capacity
    wind = train["wind_speed"].to_numpy()
    power = train["power"].to_numpy()
    chosen = to_float(THRESHOLD_GRID[0])
    for threshold in THRESHOLD_GRID:
        dropped = wind < threshold
        if dropped.sum() == 0:
            chosen = to_float(threshold)
            continue
        false_drop = to_float((power[dropped] > well_above).mean())
        if false_drop <= MAX_FALSE_DROP_RATE:
            chosen = to_float(threshold)
    return {"threshold": chosen, "idle_hi": idle_hi, "well_above": well_above}


def score_detector(
    *, test: pl.DataFrame, threshold: float, idle_hi: float, well_above: float, outage: np.ndarray
) -> dict[str, float]:
    """Share dropped, false-drop rate, and miss rate of a wind threshold on `test`."""
    wind = test["wind_speed"].to_numpy()
    power = test["power"].to_numpy()
    dropped = wind < threshold
    idle_rows = (power <= idle_hi) & ~outage
    return {
        "share_dropped": to_float(dropped.mean()),
        "false_drop_rate": to_float((power[dropped] > well_above).mean()) if dropped.any() else 0.0,
        "miss_rate": to_float((~dropped & idle_rows).sum() / max(idle_rows.sum(), 1)),
        "idle_rows_share": to_float(idle_rows.mean()),
    }


def main() -> None:
    """Run every step for the three farms and write the tables and the mask."""
    joined = pl.read_parquet(OUT_DIR / "joined.parquet")
    report: dict[str, dict] = {}
    masks: list[pl.DataFrame] = []
    bin_tables: list[pl.DataFrame] = []
    outage_tables: list[pl.DataFrame] = []
    month_tables: list[pl.DataFrame] = []
    year_bin_tables: list[pl.DataFrame] = []
    for site in SITES:
        frame = joined.filter(pl.col("site") == site).sort("valid_time")
        capacity = to_float(frame["capacity_mw"][0])
        floor = idle_floor(frame=frame)
        bins = bin_table(frame=frame)
        cut_in, cut_in_estimate = estimate_cut_in(bins=bins, idle_p95=floor["p95"])
        high_wind = 1.5 * cut_in

        # Dip: bin medians from 2 to 5 m/s against the idle floor.
        dip_bins = bins.filter(
            (pl.col("bin_lo") >= DIP_BINS[0]) & (pl.col("bin_lo") < DIP_BINS[1])
        ).with_columns(below_idle_median=pl.col("median") - floor["median"])
        dip_row = dip_bins.sort("below_idle_median").head(1)
        dip_depth = to_float(dip_row["below_idle_median"][0])
        dip = {
            "lowest_bin_median_minus_idle_median": dip_depth,
            "bin": to_float(dip_row["bin_lo"][0]),
            "share_of_capacity": dip_depth / capacity,
            "bin_median_below_idle_p05": bool(to_float(dip_row["median"][0]) < floor["p05"]),
            "bins": dip_bins.select("bin_lo", "n", "median", "below_idle_median").to_dicts(),
        }

        # H0: spread per bin above 1.5 x cut-in against the residual spread from cut-in to there.
        binned = (
            with_bins(frame=frame)
            .join(
                bins.select("bin_lo", pl.col("median").alias("bin_median")), on="bin_lo", how="left"
            )
            .with_columns(residual=pl.col("power") - pl.col("bin_median"))
        )
        reference = binned.filter(
            (pl.col("wind_speed") >= cut_in) & (pl.col("wind_speed") < high_wind)
        )["residual"]
        reference_width = to_float(reference.quantile(0.95)) - to_float(reference.quantile(0.05))
        above = bins.filter(pl.col("bin_lo") >= high_wind).with_columns(
            width=pl.col("p95") - pl.col("p05")
        )
        plateau = bins.filter(pl.col("bin_lo") >= 12.0).with_columns(
            width=pl.col("p95") - pl.col("p05")
        )
        above_rows = frame.filter(pl.col("wind_speed") > high_wind)
        idle_share_above = to_float((above_rows["power"] <= floor["p95"]).mean())
        per_bin_idle = (
            with_bins(frame=frame)
            .group_by("bin_lo")
            .agg(
                pl.len().alias("n"),
                (pl.col("power") <= floor["p95"]).mean().alias("idle_share"),
            )
            .filter(pl.col("n") >= 30)
            .sort("bin_lo")
        )

        # Outage rows.
        runs = outage_runs(frame=frame, wind_min=high_wind, idle_p95=floor["p95"])
        outage_flag = np.zeros(frame.height, dtype=bool)
        if runs.height:
            times = frame["valid_time"].to_numpy()
            for start, end in zip(runs["start"].to_numpy(), runs["end"].to_numpy(), strict=True):
                outage_flag |= (times >= start) & (times <= end)
        n_outage = int(outage_flag.sum())
        idle_share_above_excl_outage = to_float(
            (
                (above_rows["power"] <= floor["p95"]).sum()
                - int(
                    (
                        (frame["wind_speed"].to_numpy() > high_wind)
                        & (frame["power"].to_numpy() <= floor["p95"])
                        & outage_flag
                    ).sum()
                )
            )
            / max(above_rows.height, 1)
        )

        # Detector: odd years -> even years, and even -> odd.
        years = frame["valid_time"].dt.year().to_numpy()
        detector: dict[str, dict] = {}
        for label, train_mask in (("odd_to_even", years % 2 == 1), ("even_to_odd", years % 2 == 0)):
            train = frame.filter(pl.Series(train_mask))
            test = frame.filter(pl.Series(~train_mask))
            fit = detector_rates(train=train, test=test, capacity=capacity)
            scores = score_detector(
                test=test,
                threshold=fit["threshold"],
                idle_hi=fit["idle_hi"],
                well_above=fit["well_above"],
                outage=outage_flag[~train_mask],
            )
            detector[label] = {**fit, **scores}
        # Final rule: fit on all years.
        fit_all = detector_rates(train=frame, test=frame, capacity=capacity)
        final_scores = score_detector(
            test=frame,
            threshold=fit_all["threshold"],
            idle_hi=fit_all["idle_hi"],
            well_above=fit_all["well_above"],
            outage=outage_flag,
        )
        detector["all_years"] = {**fit_all, **final_scores}
        threshold = fit_all["threshold"]

        # Mask and retained-row summary.
        low_wind = (frame["wind_speed"].to_numpy() < threshold) & ~outage_flag
        dropped = low_wind | outage_flag
        reason = np.where(outage_flag, "outage_suspected", "low_wind_idle")
        masks.append(
            frame.filter(pl.Series(dropped))
            .select("time_series_id", "valid_time")
            .with_columns(drop_reason=pl.Series(reason[dropped]))
        )
        kept = frame.filter(pl.Series(~dropped))
        retained = {
            "n_all": frame.height,
            "n_dropped": int(dropped.sum()),
            "n_low_wind_idle": int(low_wind.sum()),
            "n_outage_suspected": int(outage_flag.sum()),
            "share_dropped": to_float(dropped.mean()),
            "all_median": to_float(frame["power"].median()),
            "all_p05": to_float(frame["power"].quantile(0.05)),
            "all_p95": to_float(frame["power"].quantile(0.95)),
            "kept_median": to_float(kept["power"].median()),
            "kept_p05": to_float(kept["power"].quantile(0.05)),
            "kept_p95": to_float(kept["power"].quantile(0.95)),
        }
        month_tables.append(
            frame.with_columns(dropped=pl.Series(dropped), month=pl.col("valid_time").dt.month())
            .group_by("month")
            .agg(pl.col("dropped").mean().alias("share_dropped"), pl.len().alias("n"))
            .with_columns(site=pl.lit(site))
            .sort("month")
        )

        report[site] = {
            "unit": UNITS[site],
            "capacity": capacity,
            "idle_floor": floor,
            "cut_in_used": cut_in,
            "cut_in_empirical": cut_in_estimate,
            "dip": dip,
            "h0": {
                "high_wind_from": high_wind,
                "reference_width": reference_width,
                "median_bin_width_above": to_float(above["width"].median()),
                "max_bin_width_above": to_float(above["width"].max()),
                "share_bins_wider_than_reference": to_float(
                    (above["width"] > reference_width).mean()
                ),
                "median_bin_width_plateau_12ms_up": to_float(plateau["width"].median())
                if plateau.height
                else None,
                "n_rows_above": above_rows.height,
                "idle_share_above": idle_share_above,
                "idle_share_above_excluding_outage_rows": idle_share_above_excl_outage,
                "idle_share_above_by_year": above_rows.group_by(
                    pl.col("valid_time").dt.year().alias("year")
                )
                .agg(
                    (pl.col("power") <= floor["p95"]).mean().alias("idle_share"),
                    pl.len().alias("n"),
                )
                .sort("year")
                .to_dicts(),
                "idle_share_above_10ms": to_float(
                    (frame.filter(pl.col("wind_speed") > 10.0)["power"] <= floor["p95"]).mean()
                ),
                "idle_share_above_10ms_from_2021": to_float(
                    (
                        frame.filter(
                            (pl.col("wind_speed") > 10.0) & (pl.col("valid_time").dt.year() >= 2021)
                        )["power"]
                        <= floor["p95"]
                    ).mean()
                ),
                "idle_share_by_bin": per_bin_idle.to_dicts(),
            },
            "outage": {
                "n_runs": runs.height,
                "n_half_hours": n_outage,
                "share_of_all_rows": n_outage / frame.height,
                "run_lengths_hours": (runs["n_half_hours"] / 2).to_list(),
                "start_months": runs["start"].dt.strftime("%Y-%m").to_list(),
                "median_wind_in_runs": runs["median_wind"].to_list(),
            },
            "detector": detector,
            "retained": retained,
        }
        bin_tables.append(bins.with_columns(site=pl.lit(site)))
        outage_tables.append(runs.with_columns(site=pl.lit(site)))
        year_bin_tables.append(
            with_bins(frame=frame.with_columns(year=pl.col("valid_time").dt.year()))
            .group_by("year", "bin_lo")
            .agg(
                pl.len().alias("n"),
                pl.col("power").median().alias("median"),
                pl.col("power").quantile(0.05).alias("p05"),
                pl.col("power").quantile(0.95).alias("p95"),
            )
            .filter(pl.col("n") >= 30)
            .with_columns(site=pl.lit(site))
            .sort("year", "bin_lo")
        )

    pl.concat(masks).write_parquet(OUT_DIR / "mask.parquet")
    pl.concat(bin_tables).write_parquet(OUT_DIR / "bins.parquet")
    pl.concat(year_bin_tables).write_parquet(OUT_DIR / "bins_by_year.parquet")
    pl.concat(outage_tables).write_parquet(OUT_DIR / "outage_runs.parquet")
    pl.concat(month_tables).write_parquet(OUT_DIR / "dropped_by_month.parquet")
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for site, entry in report.items():
        brief = {k: v for k, v in entry.items() if k not in ("dip", "h0")}
        brief["dip"] = {k: v for k, v in entry["dip"].items() if k != "bins"}
        brief["h0"] = {k: v for k, v in entry["h0"].items() if k != "idle_share_by_bin"}
        print(site, json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
