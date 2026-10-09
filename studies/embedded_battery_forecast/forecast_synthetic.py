"""Rung A0: does the instrument find a price effect that is there, and stay silent when none is?

Forty synthetic batteries, all issued at `DA-late`. Each battery is scored from the scoring start
on the half-hours where its own series and the inputs of both XGBoost arms are present:

- 20 **price-driven** batteries follow `lp_schedule` on the actual N2EX day-ahead price (energy
  durations of 1, 2, and 4 hours, one-way efficiencies of 0.90, 0.92, and 0.94), plus Gaussian noise
  of 10% of the noise-free output's 99th-percentile absolute value. The 20 batteries cover only 9
  distinct schedules (3 durations by 3 efficiencies), so batteries that share a schedule differ
  only in their noise, and the pooled positive control rests on fewer independent schedules than
  20.
- 20 **price-blind** batteries repeat the trailing 28-day time-of-day mean of a real embedded
  battery BMU's output, plus the same noise. They carry a battery-like daily shape and no
  information about the target day's price.

Each is forecast by an XGBoost quantile model given the actual price and by the same model given the
`shuffled` price. The positive control is the price-driven batteries' pooled CRPS difference
(`actual` minus `shuffled`) being negative and statistically significant at the 5% level; the
false-alarm control is at most 2 of the 20 price-blind batteries showing the same on their own.

Run: `OMP_NUM_THREADS=2 uv run python
studies/embedded_battery_forecast/forecast_synthetic.py [primary|sensitivity]`. Writes the frames
under `inputs/synthetic/`, the fits under `fits_as_written/<setting>/A0_DA-late/`, and
`a0_report.md`.
"""

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from forecast_arms import issue_cutoff_lines
from forecast_fit import (
    LEVELS,
    Q_COLUMNS,
    ArmDefinition,
    SettingType,
    arm_file,
    link_unchanged_fits,
    run_in_pool,
    run_job,
)
from forecast_inputs import (
    DAY_AHEAD_ARMS,
    Battery,
    build_issue_frame,
    half_hour_grid,
    load_physical_notifications,
    load_testbed_batteries,
    price_sources,
    shuffled_time,
    testbed,
)
from forecast_results import load_losses
from studies.battery_dispatch import lp_schedule
from studies.battery_forecast import band_coverage_and_width
from studies.battery_market import HALF_HOURS_PER_DAY
from studies.bootstrap import BootstrapInterval, bootstrap_difference
from studies.sources import EMBEDDED_BATTERY_FORECAST_DIR, EMBEDDED_BATTERY_FORECAST_INPUTS_DIR

SYNTHETIC_DIR: Final[Path] = EMBEDDED_BATTERY_FORECAST_INPUTS_DIR / "synthetic"
A0_ISSUE: Final[str] = "A0_DA-late"
N_PER_KIND: Final[int] = 20
NOISE_SHARE_OF_P99: Final[float] = 0.10
PROFILE_DAYS: Final[int] = 28
DURATIONS_HOURS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)
EFFICIENCIES: Final[tuple[float, ...]] = (0.90, 0.92, 0.94)
SEED: Final[int] = 20261008
ACTUAL_ARM: Final[str] = "xgb_quantile__price_actual"
SHUFFLED_ARM: Final[str] = "xgb_quantile__price_shuffled"
COVERAGE_TOLERANCE: Final[float] = 0.05


def synthetic_ids() -> tuple[list[str], list[str]]:
    """Return the identifiers of the price-driven and the price-blind batteries."""
    driven = [f"SYN_PD_{i:02d}" for i in range(1, N_PER_KIND + 1)]
    blind = [f"SYN_PB_{i:02d}" for i in range(1, N_PER_KIND + 1)]
    return driven, blind


def price_driven_output(*, prices: np.ndarray, index: int) -> np.ndarray:
    """Return one price-driven battery's noise-free output on the grid, in per-unit power."""
    return lp_schedule(
        prices=prices,
        energy_hours=DURATIONS_HOURS[index % len(DURATIONS_HOURS)],
        eta_one_way=EFFICIENCIES[(index // len(DURATIONS_HOURS)) % len(EFFICIENCIES)],
    )


def trailing_profile(*, output: np.ndarray, days: int = PROFILE_DAYS) -> np.ndarray:
    """Return, for each half-hour, the mean output at the same half-hour of day over earlier days.

    Args:
        output: The output on the half-hour grid in whole days, NaN where missing.
        days: How many earlier days the mean spans. The target's own day is never included.

    Returns:
        An array of the same shape; NaN where no earlier day has a value.
    """
    matrix = output.reshape(-1, HALF_HOURS_PER_DAY)
    profile = np.full_like(matrix, np.nan)
    for day in range(1, matrix.shape[0]):
        window = matrix[max(day - days, 0) : day]
        valid = ~np.isnan(window)
        counts = valid.sum(axis=0)
        sums = np.where(valid, window, 0.0).sum(axis=0)
        profile[day] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return profile.ravel()


def build_synthetic_series() -> dict[str, np.ndarray]:
    """Return the 40 synthetic batteries' outputs on the half-hour grid, in per-unit power."""
    grid = half_hour_grid()
    prices = price_sources(grid=grid)["price_actual"].to_numpy()
    rng = np.random.default_rng(SEED)
    driven_ids, blind_ids = synthetic_ids()
    series: dict[str, np.ndarray] = {}
    for index, battery_id in enumerate(driven_ids):
        clean = price_driven_output(prices=prices, index=index)
        scale = float(np.quantile(np.abs(clean), 0.99))
        series[battery_id] = clean + rng.normal(0.0, NOISE_SHARE_OF_P99 * scale, clean.size)
    pn = load_physical_notifications()
    members = testbed(pn=pn)
    real = load_testbed_batteries(members=members)
    chosen = list(rng.permutation(sorted(real))[:N_PER_KIND])
    for battery_id, real_id in zip(blind_ids, chosen, strict=True):
        observed = (
            grid.join(real[str(real_id)].output, on="time", how="left")["output_mw"]
            .cast(pl.Float64)
            .to_numpy()
        )
        profile = trailing_profile(output=observed)
        scale = float(np.nanquantile(np.abs(profile), 0.99))
        series[battery_id] = profile + rng.normal(0.0, NOISE_SHARE_OF_P99 * scale, profile.size)
    return series


def write_frames() -> list[str]:
    """Build and save the `DA-late` input frame of every synthetic battery."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    grid = half_hour_grid()
    prices = price_sources(grid=grid)
    shuffle = shuffled_time(grid=grid)
    empty_pn = pl.DataFrame(
        schema={
            "bmu_id": pl.String,
            "time": pl.Datetime("us", "UTC"),
            "fpn_mw": pl.Float64,
        }
    )
    series = build_synthetic_series()
    for battery_id, values in series.items():
        output = grid.with_columns(output_mw=pl.Series(values).fill_nan(None)).drop_nulls(
            "output_mw"
        )
        battery = Battery(
            battery_id=battery_id,
            output=output,
            p99_mw=float(np.quantile(np.abs(output["output_mw"].to_numpy()), 0.99)),
            lead_party=None,
            has_fpn=False,
        )
        base = build_issue_frame(
            battery=battery,
            issue="DA-late",
            grid=grid,
            prices=prices,
            pn=empty_pn,
            batteries={},
            shuffle=shuffle,
        )
        base.write_parquet(SYNTHETIC_DIR / f"{battery_id}.parquet")
    return list(series)


def control_arms() -> list[ArmDefinition]:
    """Return the arms of rung A0: the climatology and the two XGBoost quantile models."""
    return [
        ArmDefinition(name="clim", method="clim", spec=None),
        ArmDefinition(name=ACTUAL_ARM, method="xgb_quantile", spec=DAY_AHEAD_ARMS["price_actual"]),
        ArmDefinition(
            name=SHUFFLED_ARM, method="xgb_quantile", spec=DAY_AHEAD_ARMS["price_shuffled"]
        ),
    ]


def job(task: tuple[str, SettingType]) -> list[str]:
    """Run rung A0's arms for one synthetic battery."""
    battery_id, setting = task
    base = pl.read_parquet(SYNTHETIC_DIR / f"{battery_id}.parquet")
    scoring = {n: a for n, a in DAY_AHEAD_ARMS.items() if a.price_source in ("actual", "shuffled")}
    return run_job(
        battery_id=battery_id,
        issue=A0_ISSUE,
        setting=setting,
        base=base,
        arms=control_arms(),
        scoring_arms=scoring,
    )


def group_row(*, label: str, result: BootstrapInterval, hits: int) -> str:
    """Return one row of the pooled-contrast table."""
    return (
        f"| {label} | {result['difference']:+.3f} | "
        f"[{result['lower_95']:+.3f}, {result['upper_95']:+.3f}] | {hits} of 20 |"
    )


def analyse(*, setting: SettingType) -> tuple[list[str], bool]:
    """Return rung A0's report lines and whether the positive control passed."""
    driven, blind = synthetic_ids()
    losses = load_losses(
        setting=setting,
        issue=A0_ISSUE,
        arms=[ACTUAL_ARM, SHUFFLED_ARM, "clim"],
        batteries=[*driven, *blind],
    )

    def contrast(batteries: list[str]) -> BootstrapInterval:
        subset = losses.filter(pl.col("site").is_in(batteries))
        return bootstrap_difference(
            losses=subset, treatment=ACTUAL_ARM, reference=SHUFFLED_ARM, metric="crps_pct"
        )

    lines = [f"## Rung A0 at the {setting} hyperparameter setting", ""]
    pooled_driven = contrast(driven)
    pooled_blind = contrast(blind)
    per_driven = {b: contrast([b]) for b in driven}
    per_blind = {b: contrast([b]) for b in blind}
    hit_driven = sum(r["upper_95"] < 0 for r in per_driven.values())
    hit_blind = sum(r["upper_95"] < 0 for r in per_blind.values())
    lines += [
        (
            "| Group | CRPS difference, `actual` minus `shuffled` (points of p99) | 95% interval | "
            "Batteries individually significant at the 5% level |"
        ),
        "|---|---|---|---|",
        group_row(label="Price-driven (20)", result=pooled_driven, hits=hit_driven),
        group_row(label="Price-blind (20)", result=pooled_blind, hits=hit_blind),
        "",
    ]
    positive = pooled_driven["upper_95"] < 0
    false_alarm_ok = hit_blind <= 2
    lines += [
        (
            "- Positive control (price-driven pooled difference negative and statistically "
            f"significant at the 5% level): **{'passed' if positive else 'FAILED'}**."
        ),
        (
            "- False-alarm control (at most 2 of 20 price-blind batteries significant): "
            f"**{'passed' if false_alarm_ok else 'FAILED'}** ({hit_blind} of 20)."
        ),
        "",
        "| Battery | Difference | Lower 95 | Upper 95 |",
        "|---|---|---|---|",
    ]
    for battery, result in {**per_driven, **per_blind}.items():
        lines.append(
            f"| {battery} | {result['difference']:+.3f} | {result['lower_95']:+.3f} | "
            f"{result['upper_95']:+.3f} |"
        )
    # Calibration of the price-aware model on the price-driven batteries.
    coverages = []
    for battery in driven:
        frame = pl.read_parquet(
            arm_file(setting=setting, issue=A0_ISSUE, battery_id=battery, arm=ACTUAL_ARM),
            columns=["seed", "truth_mw", *Q_COLUMNS],
        ).filter(pl.col("seed") == 0)
        bands = band_coverage_and_width(
            truth=frame["truth_mw"].to_numpy(),
            quantiles=frame.select(Q_COLUMNS).to_numpy(),
            levels=LEVELS,
        )
        coverages.append(bands.filter(pl.col("lower_level") == 0.1)["coverage"].item())
    mean_coverage = float(np.mean(coverages))
    calibrated = abs(mean_coverage - 0.8) <= COVERAGE_TOLERANCE
    lines += [
        "",
        (
            f"- Mean p10-p90 coverage of `{ACTUAL_ARM}` on the 20 price-driven batteries: "
            f"{mean_coverage:.3f} (nominal 0.800; target within {COVERAGE_TOLERANCE:.2f}): "
            f"**{'within' if calibrated else 'OUTSIDE'}**."
        ),
        "",
    ]
    return lines, positive


def main() -> None:
    """Build the frames, fit rung A0, and write `a0_report.md`."""
    setting: SettingType = "sensitivity" if "sensitivity" in sys.argv[1:] else "primary"
    write_frames()
    link_unchanged_fits()
    driven, blind = synthetic_ids()
    run_in_pool(function=job, tasks=[(b, setting) for b in [*driven, *blind]])
    lines, positive = analyse(setting=setting)
    path = EMBEDDED_BATTERY_FORECAST_DIR / f"a0_report_{setting}.md"
    cutoffs = ["## Issue-time cut-offs used", "", *issue_cutoff_lines(), ""]
    lines = [*cutoffs, *lines]
    path.write_text("\n".join(["# Rung A0: the known-answer check", "", *lines]))
    print("\n".join(lines))
    print(f"positive control passed: {positive}")


if __name__ == "__main__":
    main()
