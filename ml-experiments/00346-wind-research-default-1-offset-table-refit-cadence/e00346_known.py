"""Known-answer test for experiment 00346: do the trailing windows track a drifting bias?

Synthetic half-hourly cases for one farm, ten lead days each: observed power is a fixed Beta(2, 5)
draw plus a bias that grows linearly from 0 to 6 points of capacity between 2024-04 and 2026-06,
and the forecast is the Beta distribution's true quantiles, which know nothing of the bias. Each
arm's table is fitted with the same window and the same cutoff as `e00346_final.py`, and the
offset at the median level is averaged over the lead days. A table that tracks the drift returns
the bias averaged over its own window (the window's centre, 1.5 months back for the 3-month window
and 3 months back for the 6-month window, ignoring the cutoff). The test passes if the 3-month and
6-month tables are each within 0.5 points of that expected value at every month from 2025-01, and
the expanding window lags the drift at 2026-06 by more than 1 point.

    uv run --no-sync python e00346_known.py   (PYTHONPATH: see e00346_common)
"""

import json

import e00346_common as common
import e00346_final as final
import numpy as np
import polars as pl

DRIFT_POINTS = 0.06
FIRST = final.month_start(ym=(2024, 4))
LAST = final.month_start(ym=(2026, 7))


def bias_at(*, valid: np.ndarray) -> np.ndarray:
    """Return the injected bias (fraction of capacity) at each valid time."""
    span = (LAST - FIRST).total_seconds()
    seconds = (
        (valid - np.datetime64(FIRST.replace(tzinfo=None))).astype("timedelta64[s]").astype(float)
    )
    return DRIFT_POINTS * np.clip(seconds / span, 0.0, 1.0)


def main() -> None:
    """Run the test and write `work/final/known_answer.json`."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    days = int((LAST - FIRST).days)
    half_hours = np.arange(days * 48).astype("timedelta64[30m]")
    valid = np.repeat(np.datetime64(FIRST.replace(tzinfo=None)) + half_hours, 10)
    lead_day = np.tile(np.arange(1, 11), days * 48)
    raw = rng.beta(2.0, 5.0, size=valid.size)
    truth = np.quantile(raw, common.c303.LEVELS)
    actual = raw + bias_at(valid=valid)
    cases = pl.DataFrame(
        {
            "site": "W1",
            "lead_day": lead_day,
            "actual_norm": actual,
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        }
    )
    forecast = np.tile(truth, (valid.size, 1))
    result: dict = {"months": {}}
    errors: dict[str, list[float]] = {"x1": [], "x2": [], "x3": []}
    for ym in final.months_between(first=(2025, 1), last=(2026, 6)):
        entry = {}
        for arm, window in final.ARMS_MONTHLY.items():
            select, _ = final.window_select(cases=cases, ym=ym, months=window)
            table, _ = final.fit_table(cases=cases, adjusted=forecast, select=select)
            recovered = float(table[:, common.c303.MEDIAN_INDEX].mean() + 0.0)
            # The table's offset at the median is the median of (actual - forecast median).
            start = final.month_start(ym=ym)
            lowest = (
                FIRST
                if window is None
                else final.month_start(ym=final.add_months(year=ym[0], month=ym[1], delta=-window))
            )
            centre = lowest + (start - lowest) / 2
            expected = float(
                bias_at(valid=np.array([np.datetime64(centre.replace(tzinfo=None))]))[0]
            )
            entry[arm] = {"recovered": recovered, "expected_window_mean": expected}
            errors[arm].append(abs(recovered - expected) * 100.0)
        result["months"][final.label(ym=ym)] = entry
    lag_x1 = (
        abs(
            result["months"]["2026-06"]["x1"]["recovered"]
            - bias_at(
                valid=np.array(
                    [np.datetime64(final.month_start(ym=(2026, 6)).replace(tzinfo=None))]
                )
            )[0]
        )
        * 100.0
    )
    result["max_error_points"] = {arm: float(max(v)) for arm, v in errors.items()}
    result["x1_lag_points_at_2026_06"] = float(lag_x1)
    result["passed"] = bool(max(errors["x3"]) <= 0.5 and max(errors["x2"]) <= 0.5 and lag_x1 > 1.0)
    (common.FINAL_DIR / "known_answer.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "months"}, indent=1))


if __name__ == "__main__":
    main()
