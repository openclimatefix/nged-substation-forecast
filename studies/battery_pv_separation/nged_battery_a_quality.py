"""Step 1 for NGED battery A: is the series usable? Prints aggregate statistics only.

Checks the unit, resolution, coverage of the study's window, gaps, exact zeros and stuck runs, the
sign convention (a battery charges when the day-ahead price is low), and whether the series is net
of other equipment (a solar-shaped component would correlate with the CAMS irradiance after the
time of day and the price are accounted for). Every power value is a fraction of the series' own
99th percentile absolute output. Writes `report_nged_battery_a_quality.md` under the study's data
folder.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/nged_battery_a_quality.py`.
"""

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import HALF_HOURS_PER_DAY, OUTPUT_DIR, WINDOW_START
from studies.nged_battery_a import (
    ALIAS,
    WINDOW_DAYS,
    battery_a_frame,
    battery_a_raw,
    battery_a_units,
    market_full,
    nearest_cams_hourly,
    window_filter,
)

STUCK_HALF_HOURS: Final[int] = 4
"""A run of this many identical consecutive readings (two hours) counts as stuck."""
LONG_STUCK_HALF_HOURS: Final[int] = 48
MIN_CLEAR_SKY_W_M2: Final[float] = 50.0
"""The half-hours used in the solar check have at least this clear-sky irradiance."""


def run_lengths(*, values: np.ndarray, grid_ok: np.ndarray) -> list[tuple[float, int]]:
    """Return the value and length of every run of identical consecutive readings.

    Args:
        values: Readings on a regular half-hourly grid.
        grid_ok: Whether each reading is present and follows the previous one by 30 minutes.

    Returns:
        One pair per run of at least `STUCK_HALF_HOURS`.
    """
    runs = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start] or not grid_ok[index]:
            if index - start >= STUCK_HALF_HOURS:
                runs.append((float(values[start]), index - start))
            start = index
    return runs


def solar_check(*, frame: pl.DataFrame) -> list[str]:
    """Test whether the output follows the sun after the time of day and the price are removed.

    Args:
        frame: The window frame from `battery_a_frame`.

    Returns:
        Report lines.
    """
    cams = nearest_cams_hourly()
    joined = (
        frame.with_columns(hour_start=pl.col("time").dt.truncate("1h"))
        .join(cams, on="hour_start", how="inner")
        .filter(pl.col("clear_sky_ghi_w_m2") > MIN_CLEAR_SKY_W_M2)
    )
    tod_mean = joined.group_by("tod").agg(tod_mean=pl.col("output_mw").mean())
    joined = joined.join(tod_mean, on="tod").with_columns(
        residual=pl.col("output_mw") - pl.col("tod_mean"),
        clearness=pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2"),
    )
    daylight = joined["time"].n_unique()
    raw_correlation = np.corrcoef(joined["output_mw"], joined["ghi_w_m2"])[0, 1]
    residual_correlation = np.corrcoef(joined["residual"], joined["clearness"])[0, 1]
    lines = [
        (
            f"- Daylight half-hours with CAMS irradiance (clear-sky above "
            f"{MIN_CLEAR_SKY_W_M2:.0f} W/m2): {daylight}."
        ),
        (
            f"- Correlation of output with irradiance (W/m2): {raw_correlation:.3f}; of the "
            "output's deviation from its time-of-day mean with the clearness index "
            f"(irradiance / clear-sky): {residual_correlation:.3f}."
        ),
    ]
    # Least squares on time-of-day indicators, price rank, price level, and irradiance.
    tod = joined["tod"].to_numpy()
    design = np.hstack(
        [
            (tod[:, None] == np.arange(HALF_HOURS_PER_DAY)[None, :]).astype(float),
            joined.select("rank_pct", "day_ahead_gbp_per_mwh").to_numpy() / [1.0, 100.0],
            joined["clearness"].to_numpy()[:, None],
        ]
    )
    target = joined["output_mw"].to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    sigma2 = residual @ residual / (len(target) - design.shape[1])
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(covariance[-1, -1])
    lines.append(
        "- Output (fraction of p99) per unit of clearness index, controlling for the half-hour of "
        "day, the day-ahead price rank and level: "
        f"{coefficients[-1]:.4f} (naive standard error {standard_error:.4f}). "
        "A solar array behind the meter would give a clearly positive coefficient of about the "
        "array's size."
    )
    # Midday against the rest of the same day, sunny against dull.
    daily = (
        joined.filter(pl.col("tod").is_between(20, 27))
        .group_by("date")
        .agg(
            midday=pl.col("residual").mean(),
            clearness=pl.col("clearness").mean(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= 6)
    )
    lines.append(
        f"- Over {daily.height} days, the correlation of the midday (10:00 to 14:00 UTC) "
        "deviation from the time-of-day mean with the day's clearness index: "
        f"{np.corrcoef(daily['midday'], daily['clearness'])[0, 1]:.3f}."
    )
    return lines


def main() -> None:
    """Print and write the data-quality check."""
    raw = battery_a_raw()
    window = raw.filter(window_filter())
    expected = WINDOW_DAYS * HALF_HOURS_PER_DAY
    lines = [f"## {ALIAS}: data-quality check", ""]
    lines.append(f"- Unit: {battery_a_units()}. Rows in the whole series: {raw.height}.")
    steps = raw["time"].diff().drop_nulls().value_counts().sort("count", descending=True)
    lines.append(
        "- Most common spacing between readings: "
        f"{steps['time'][0]} ({steps['count'][0] / (raw.height - 1):.1%} of consecutive pairs)."
    )
    span_days = raw.select((pl.col("time").max() - pl.col("time").min()).dt.total_days()).item()
    lines.append(
        f"- Series length: {span_days / 365.25:.1f} years. "
        f"Rows in the study window of {WINDOW_DAYS} days: {window.height} of {expected} "
        f"({window.height / expected:.1%} present)."
    )
    market = market_full()
    covered = (
        window.join(market.drop_nulls("day_ahead_gbp_per_mwh"), on="time", how="inner").height
        / expected
    )
    lines.append(f"- Window rows that also have a day-ahead price: {covered:.1%} of the window.")
    # Gaps in the window.
    grid = pl.DataFrame(
        {
            "time": pl.datetime_range(
                WINDOW_START,
                WINDOW_START + timedelta(days=WINDOW_DAYS) - timedelta(minutes=30),
                interval="30m",
                time_zone="UTC",
                time_unit="us",
                eager=True,
            )
        }
    ).join(window, on="time", how="left")
    missing = grid["power"].is_null().to_numpy()
    gap_lengths = []
    length = 0
    for flag in missing:
        if flag:
            length += 1
        elif length:
            gap_lengths.append(length)
            length = 0
    if length:
        gap_lengths.append(length)
    median_gap = np.median(gap_lengths) if gap_lengths else 0
    lines.append(
        f"- Gaps in the window: {len(gap_lengths)} gaps, {int(missing.sum())} missing half-hours "
        f"({missing.mean():.2%}); gap length median {median_gap:.0f} "
        f"half-hours, longest {max(gap_lengths, default=0)} half-hours "
        f"({max(gap_lengths, default=0) / 2:.0f} hours); gaps longer than a day: "
        f"{sum(g > 48 for g in gap_lengths)}."
    )
    power = window["power"].to_numpy()
    scale = float(np.quantile(np.abs(power), 0.99))
    normalised = power / scale
    lines.append(
        f"- Exact zeros: {(power == 0).mean():.1%} of window readings. Readings within 0.5% of "
        f"p99 of zero: {(np.abs(normalised) < 0.005).mean():.1%}."
    )
    ok = np.concatenate([[False], np.diff(window["time"].dt.epoch("s").to_numpy()) == 1800])
    runs = run_lengths(values=power, grid_ok=ok)
    nonzero = [r for r in runs if r[0] != 0.0]
    zero = [r for r in runs if r[0] == 0.0]
    lines.append(
        f"- Runs of at least {STUCK_HALF_HOURS} identical consecutive readings: "
        f"{len(nonzero)} nonzero runs (longest {max((r[1] for r in nonzero), default=0)} "
        f"half-hours; {sum(r[1] >= LONG_STUCK_HALF_HOURS for r in nonzero)} of at least "
        f"{LONG_STUCK_HALF_HOURS}) covering {sum(r[1] for r in nonzero) / window.height:.2%} of "
        f"readings, and {len(zero)} zero runs (longest {max((r[1] for r in zero), default=0)}) "
        f"covering {sum(r[1] for r in zero) / window.height:.2%}."
    )
    lines.append(
        "- Distribution of output as a fraction of p99 absolute output: "
        + ", ".join(
            f"p{q}: {np.quantile(normalised, q / 100):.3f}" for q in (0.5, 1, 5, 25, 50, 75, 95, 99)
        )
        + f"; largest absolute value {np.abs(normalised).max():.2f}."
    )
    lines.append(
        f"- Share of readings beyond +/-0.05 of p99: charging (negative) "
        f"{(normalised < -0.05).mean():.1%}, exporting (positive) {(normalised > 0.05).mean():.1%}."
    )
    # Sign convention.
    frame, _ = battery_a_frame()
    by_tercile = frame.group_by("tercile").agg(mean=pl.col("output_mw").mean()).sort("tercile")
    correlation = float(np.corrcoef(frame["output_mw"], frame["rank_pct"])[0, 1])
    lines.append(
        f"- Sign check, as metered: mean output (fraction of p99) in the cheapest, middle, and "
        f"dearest third of each day: {', '.join(f'{v:.3f}' for v in by_tercile['mean'])}; "
        f"correlation with the day-ahead price rank within the day: {correlation:.3f}. "
        + (
            "A battery charges when the price is low, so a positive correlation means positive "
            "is export, the convention the public batteries use."
            if correlation > 0
            else "The correlation is negative, so the series is import-positive."
        )
    )
    exported = float(frame.filter(pl.col("output_mwh") > 0)["output_mwh"].sum())
    imported = float(-frame.filter(pl.col("output_mwh") < 0)["output_mwh"].sum())
    lines.append(
        f"- Energy balance: exported / imported = {exported / imported:.3f}, so the implied "
        f"one-way efficiency sqrt(ratio) is {np.sqrt(exported / imported):.3f}."
    )
    lines += solar_check(frame=frame)
    lines.append("")
    (OUTPUT_DIR / "report_nged_battery_a_quality.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
