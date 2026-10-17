"""POST HOC: the change-point monitor with a shorter minimum segment.

**Not pre-registered.** The pre-registered monitor needs 14 valid days in every segment, and a
30-day loss that starts from April to September holds only about 17 valid days, so it can hardly
raise an alarm for one. This script changes only that minimum (to `POSTHOC_MIN_SEGMENT`), re-tunes
the penalty multiplier on the tuning placebo to the same 0.35 false alarms per farm-year, and scores
the same planted losses and the same scored placebo. The pre-registered contrasts and verdict do
not change.
"""

import json
from typing import Final

import polars as pl
from e00072_detectors import change_point_alarms
from e00072_inject import tuned_parameter
from e00072_score import OUTPUT_DIR, SCORED_DRAWS, SCORED_SEED_BASE, injected_rows
from e00072_tune import (
    GRIDS,
    TARGET_FALSE_ALARMS_PER_FARM_YEAR,
    TUNING_SEED_BASE,
    TUNING_SHUFFLES,
    placebo_signals,
)
from studies.guards import refuse_to_overwrite

POSTHOC_MIN_SEGMENT: Final[int] = 7
NAME: Final[str] = "changepoint_short_segment"


def main() -> None:
    """Tune the short-segment monitor, score it, and write its tables."""
    paths = {name: OUTPUT_DIR / f"posthoc_{name}.parquet" for name in ("events", "placebo")}
    config_path = OUTPUT_DIR / "posthoc_config.json"
    refuse_to_overwrite(paths=[*paths.values(), config_path])
    tuning = [placebo_signals(seed=TUNING_SEED_BASE + k) for k in range(TUNING_SHUFFLES)]
    years = sum(span for signals in tuning for _, span in signals)
    rates = {}
    for scale in GRIDS["changepoint"]:
        count = sum(
            len(
                change_point_alarms(
                    signal=signal, penalty_scale=scale, min_segment=POSTHOC_MIN_SEGMENT
                )
            )
            for signals in tuning
            for signal, _ in signals
        )
        rates[scale] = count / years
    chosen = tuned_parameter(rates=rates, target=TARGET_FALSE_ALARMS_PER_FARM_YEAR)
    print("chosen", chosen, {k: round(v, 3) for k, v in rates.items()})
    config_path.write_text(
        json.dumps(
            {
                "min_segment": POSTHOC_MIN_SEGMENT,
                "penalty_scale": chosen,
                "rates": {str(k): v for k, v in rates.items()},
            },
            indent=2,
        )
        + "\n"
    )

    def alarm_function(signal):  # noqa: ANN001, ANN202
        return change_point_alarms(
            signal=signal, penalty_scale=chosen, min_segment=POSTHOC_MIN_SEGMENT
        )

    events, _ = injected_rows(
        parameters={}, extra_alarm_functions={NAME: alarm_function}, only_extra=True
    )
    events.write_parquet(paths["events"])
    rows = []
    for draw in range(SCORED_DRAWS):
        for site, (signal, span) in zip(
            ["A", "B", "C", "D", "F"], placebo_signals(seed=SCORED_SEED_BASE + draw), strict=True
        ):
            rows.append(
                {
                    "draw": draw,
                    "site": site,
                    "method": NAME,
                    "alarms": len(alarm_function(signal)),
                    "farm_years": span,
                }
            )
    pl.DataFrame(rows).write_parquet(paths["placebo"])


if __name__ == "__main__":
    main()
