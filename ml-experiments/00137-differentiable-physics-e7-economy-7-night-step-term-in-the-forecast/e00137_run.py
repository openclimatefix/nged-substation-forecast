"""Run experiment 137: controls first, then the forecasts of every arm.

    uv run --no-sync python ml-experiments/00137-*/e00137_run.py prep
    uv run --no-sync python ml-experiments/00137-*/e00137_run.py controls
    uv run --no-sync python ml-experiments/00137-*/e00137_run.py forecast

`prep` annotates the base rows with night features. `controls` runs the ridge-penalty choice, the
leakage control, and the injected-step control. `forecast` writes the arms' prediction sets.
"""

import argparse
from datetime import UTC, datetime
from typing import Any, Final

import e00137_common as common
import e00137_term as term
import numpy as np
import polars as pl
from studies.experiment_store import write_predictions

NIGHT_DIR: Final = common.WORK_DIR / "night"
TUNING_ODD_MONTHS: Final = (
    datetime(2024, 9, 1, tzinfo=UTC),
    datetime(2024, 11, 1, tzinfo=UTC),
    datetime(2025, 1, 1, tzinfo=UTC),
    datetime(2025, 3, 1, tzinfo=UTC),
    datetime(2025, 5, 1, tzinfo=UTC),
)
INJECT_LEVEL: Final[float] = 8.0
INJECT_SLOPE_PER_C: Final[float] = 0.5
INJECT_TRUTH_AT_3C: Final[float] = INJECT_LEVEL + INJECT_SLOPE_PER_C * 12.5
READ_SLOTS: Final[tuple[int, ...]] = (8, 9, 10, 11)
"""Night slots ending 01:30, 02:00, 02:30, and 03:00 UTC."""
INJECT_TOLERANCE: Final[float] = 0.25
CORRUPTION: Final[float] = 100.0
LEAKAGE_CUTS: Final = (
    datetime(2025, 11, 1, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
    datetime(2026, 2, 1, tzinfo=UTC),
)
PLACEBO_SEED: Final[int] = 137


def read_groups() -> dict[str, pl.DataFrame]:
    """Return each site's night-annotated base rows written by `prep`."""
    return {s: pl.read_parquet(NIGHT_DIR / f"{s}.parquet") for s in common.SITES}


def prep() -> None:
    """Annotate every site's base rows and write them."""
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    for site in common.SITES:
        frame = term.add_night_columns(site=site, frame=common.load_base(site=site))
        frame.write_parquet(NIGHT_DIR / f"{site}.parquet")
        night = frame.filter(pl.col("slot") >= 0)
        print(
            site,
            frame.height,
            "night rows",
            night.height,
            "T missing",
            night["temp_c"].is_nan().sum(),
        )


def quantile_score(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the quantile score (twice the mean pinball loss over the 11 levels) per row."""
    error = frame["actual"].to_numpy().astype(np.float64)[:, None] - frame.select(
        list(common.QUANTILES)
    ).to_numpy().astype(np.float64)
    return 2.0 * np.mean(
        np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error), axis=1
    )


def day_ahead(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep rows at leads above 12 and at most 36 hours."""
    return frame.filter((pl.col("lead_hours") > 12.0) & (pl.col("lead_hours") <= 36.0))


def select_ridge(*, groups: dict[str, pl.DataFrame]) -> dict:
    """Pick the ridge penalty with the lowest window-row score on the odd tuning months."""
    scores = {}
    for ridge in term.LAMBDA_GRID:
        fitted = term.monthly_fit_forecast(groups=groups, ridge=ridge, pooled=False)
        total, count = 0.0, 0
        for frame in fitted.values():
            rows = day_ahead(
                frame=frame.filter(
                    pl.col("window")
                    & pl.col("power_fcst_init_time")
                    .dt.truncate("1mo")
                    .is_in(list(TUNING_ODD_MONTHS))
                )
            )
            total += float(quantile_score(frame=rows).sum())
            count += rows.height
        scores[ridge] = total / count
    base_total, base_count = 0.0, 0
    for frame in groups.values():
        rows = day_ahead(
            frame=frame.filter(
                pl.col("window")
                & pl.col("power_fcst_init_time").dt.truncate("1mo").is_in(list(TUNING_ODD_MONTHS))
            )
        )
        base_total += float(quantile_score(frame=rows).sum())
        base_count += rows.height
    best = min(scores, key=lambda ridge: scores[ridge])
    return {"scores": scores, "base_score": base_total / base_count, "ridge": best}


def inject(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the known synthetic step to `actual` on every night row with a finite `T`."""
    minute = term.minute_of_day(frame=frame)
    ramp = np.where((minute >= 90) & (minute <= 540), 1.0, np.where(minute == 60, 0.5, 0.0))
    cold = frame["cold"].to_numpy()
    step = np.where(
        (frame["slot"].to_numpy() >= 0) & np.isfinite(cold),
        (INJECT_LEVEL + INJECT_SLOPE_PER_C * cold) * ramp,
        0.0,
    )
    return frame.with_columns((pl.col("actual") + pl.Series(step).cast(pl.Float32)).alias("actual"))


def recovered(*, beta: np.ndarray) -> float:
    """Return the fitted correction at 3 C on a weekday, averaged over the read slots."""
    return float(np.mean([beta[3 * s] + beta[3 * s + 1] * 1.25 for s in READ_SLOTS]))


def controls() -> dict:
    """Run the ridge choice, the leakage control, and the injected-step control."""
    groups = read_groups()
    chosen = select_ridge(groups=groups)
    ridge = chosen["ridge"]
    print("ridge", chosen)
    # Leakage: corrupt `actual` from a cut on, rerun the whole monthly fit, and compare the
    # corrected quantiles of that cut's month. Only rows before the cut may enter its fit.
    clean_fit = term.monthly_fit_forecast(groups=groups, ridge=ridge, pooled=False)
    worst = 0.0
    for cut in LEAKAGE_CUTS:
        corrupt = {
            site: frame.with_columns(
                pl.when(pl.col("valid_time") >= cut)
                .then(pl.col("actual") + CORRUPTION)
                .otherwise(pl.col("actual"))
                .alias("actual")
            )
            for site, frame in groups.items()
        }
        corrupt_fit = term.monthly_fit_forecast(groups=corrupt, ridge=ridge, pooled=False)
        for site in groups:
            month = pl.col("power_fcst_init_time").dt.truncate("1mo") == cut
            a = clean_fit[site].filter(month)["q50"].to_numpy()
            b = corrupt_fit[site].filter(month)["q50"].to_numpy()
            worst = max(worst, float(np.abs(a - b).max()))
    chosen["leakage_max_q50_change"] = worst
    # Injected step: refit on injected rows; recovery is the fitted step minus the clean fit's.
    injected = {s: inject(frame=f) for s, f in groups.items()}
    _, clean_site = term.monthly_fit_with_betas(groups=groups, ridge=ridge, pooled=False)
    _, injected_site = term.monthly_fit_with_betas(groups=injected, ridge=ridge, pooled=False)
    _, clean_pool = term.monthly_fit_with_betas(groups=groups, ridge=ridge, pooled=True)
    _, injected_pool = term.monthly_fit_with_betas(groups=injected, ridge=ridge, pooled=True)
    rows: list[dict[str, Any]] = []
    cuts = [
        c for c in term.month_starts(frame=groups["D5"]) if c >= datetime(2025, 11, 1, tzinfo=UTC)
    ]
    cuts = [c for c in cuts if c <= datetime(2026, 3, 1, tzinfo=UTC)]
    for cut in cuts:
        for site in common.SITES:
            rec = recovered(beta=injected_site[(site, cut)]) - recovered(
                beta=clean_site[(site, cut)]
            )
            rows.append({"arm": "t_site", "site": site, "cut": str(cut.date()), "recovered": rec})
        rec = recovered(beta=injected_pool[("pool", cut)]) - recovered(
            beta=clean_pool[("pool", cut)]
        )
        rows.append({"arm": "t_pool", "site": "pool", "cut": str(cut.date()), "recovered": rec})
    for row in rows:
        row["relative_error"] = abs(row["recovered"] - INJECT_TRUTH_AT_3C) / INJECT_TRUTH_AT_3C
    chosen["injected"] = rows
    chosen["injected_pass_by_cut"] = {
        str(cut.date()): int(
            sum(
                r["relative_error"] <= INJECT_TOLERANCE
                for r in rows
                if r["arm"] == "t_site" and r["cut"] == str(cut.date())
            )
        )
        for cut in cuts
    }
    common.write_json_atomic(path=common.WORK_DIR / "controls.json", content=chosen)
    print("leakage max q50 change", worst)
    print("injected: sites within 25% by cut", chosen["injected_pass_by_cut"])
    return chosen


def ramp_123(*, minute: np.ndarray) -> np.ndarray:
    """Return experiment 123's ramp by the half-hour's end: 0 to 00:00, 0.3 at 00:30, then 1."""
    in_night = (minute >= common.NIGHT_START_MINUTE) | (minute <= 540)
    ramp = np.where((minute >= 60) & (minute <= 540), 1.0, np.where(minute == 30, 0.3, 0.0))
    return np.where(in_night, ramp, 0.0)


def fixed_123(*, site: str, frame: pl.DataFrame, summer_adjusted: bool = False) -> pl.DataFrame:
    """Insert experiment 123's step at the forecast temperature, minus the step the base forecasts.

    Experiment 123's step is a winter night's rise minus the same rise on June to August nights.
    With `summer_adjusted` (post hoc), the base's own rise on its July and August 2024 nights is
    subtracted from the base's rise on the night, so the two steps are measured the same way.
    """
    row = (
        pl.read_parquet(common.FIT_HEADLINE)
        .filter((pl.col("site") == site) & pl.col("winter_set").str.starts_with("headline"))
        .row(0, named=True)
    )
    minute = term.minute_of_day(frame=frame)
    window = pl.col("window")
    evening = term.MINUTE
    base_step = (
        frame.with_columns(
            in_window=window,
            in_baseline=evening.is_in([22 * 60 + 30, 23 * 60, 23 * 60 + 30, 0]),
        )
        .group_by(["run", "evening"])
        .agg(
            night=pl.col("mean").filter(pl.col("in_window")).mean(),
            before=pl.col("mean").filter(pl.col("in_baseline")).mean(),
        )
        .with_columns(base_step=(pl.col("night") - pl.col("before")).fill_null(0.0))
        .select(["run", "evening", "base_step"])
    )
    joined = frame.join(base_step, on=["run", "evening"], how="left")
    if summer_adjusted:
        summer = base_step.join(
            frame.select(["run", "evening", "power_fcst_init_time"]).unique(["run", "evening"]),
            on=["run", "evening"],
        ).filter(
            pl.col("power_fcst_init_time")
            .dt.truncate("1mo")
            .is_in([datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 8, 1, tzinfo=UTC)])
        )
        joined = joined.with_columns(
            base_step=pl.col("base_step").fill_null(0.0)
            - float(np.mean(summer["base_step"].to_numpy()))
        )
    step = row["flat"] + row["slope"] * joined["cold"].to_numpy()
    shift = ramp_123(minute=minute) * (step - joined["base_step"].fill_null(0.0).to_numpy())
    shift = np.where(np.isfinite(shift) & (frame["slot"].to_numpy() >= 0), shift, 0.0)
    return term.shift_quantiles(frame=frame, shift=shift)


def placebo(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Permute each evening's `cold` across the evenings of one substation."""
    rng = np.random.default_rng(PLACEBO_SEED)
    per_evening = frame.filter(pl.col("slot") >= 0).group_by("evening").agg(pl.col("cold").mean())
    evenings = per_evening["evening"].to_list()
    values = per_evening["cold"].to_numpy()
    mapping = dict(zip(evenings, rng.permutation(values), strict=True))
    permuted = frame["evening"].map_elements(
        lambda d: mapping.get(d, np.nan), return_dtype=pl.Float64
    )
    return frame.with_columns(
        cold=pl.when(pl.col("slot") >= 0).then(permuted).otherwise(pl.col("cold"))
    )


def scored(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the scored year's rows, `run`-ordered."""
    return frame.filter(pl.col("power_fcst_init_time") >= common.SCORED_START).sort(
        ["time_series_id", "power_fcst_init_time", "valid_time"]
    )


def forecast(*, ridge: float, only: str | None = None) -> None:
    """Write the prediction sets of every arm for the scored year."""
    groups = read_groups()
    arms: dict[str, dict[str, pl.DataFrame]] = {"base": groups}
    arms["t_site"] = term.monthly_fit_forecast(groups=groups, ridge=ridge, pooled=False)
    arms["t_pool"] = term.monthly_fit_forecast(groups=groups, ridge=ridge, pooled=True)
    arms["placebo"] = term.monthly_fit_forecast(
        groups={s: placebo(frame=f) for s, f in groups.items()}, ridge=ridge, pooled=False
    )
    arms["fixed123"] = {s: fixed_123(site=s, frame=f) for s, f in groups.items()}
    oracle = {}
    for site, frame in groups.items():
        rows = scored(frame=frame)
        beta = term.fit_ridge(frames=[rows], ridge=ridge)
        oracle[site] = term.shift_quantiles(
            frame=rows, shift=term.correction(frame=rows, beta=beta)
        )
    arms["oracle"] = oracle
    constant_only = {
        s: f.with_columns(
            cold=pl.when(pl.col("slot") >= 0).then(0.0).otherwise(pl.col("cold")),
            weekend=pl.lit(0.0),
        )
        for s, f in groups.items()
    }
    arms["t_const"] = term.monthly_fit_forecast(groups=constant_only, ridge=ridge, pooled=False)
    arms["fixed123b"] = {
        s: fixed_123(site=s, frame=f, summer_adjusted=True) for s, f in groups.items()
    }
    for name, by_site in arms.items():
        if only is not None and name != only:
            continue
        frame = pl.concat([scored(frame=f) for f in by_site.values()], how="diagonal")
        columns = [*common.KEEP, "site", "slot", "evening", "temp_c", "cold", "weekend", "window"]
        write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=frame.select(columns),
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        print(name, frame.height)


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prep", "controls", "forecast"])
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.stage == "prep":
        prep()
    elif args.stage == "controls":
        controls()
    else:
        import json

        ridge = json.loads((common.WORK_DIR / "controls.json").read_text())["ridge"]
        forecast(ridge=float(ridge), only=args.only)


if __name__ == "__main__":
    main()
