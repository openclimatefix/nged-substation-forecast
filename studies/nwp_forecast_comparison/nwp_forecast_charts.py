"""Chart the matched-lead weather-forecast comparison's results.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. **Skeleton only: every
function below is a stub that raises `NotImplementedError`.** No chart is drawn in this pass, and
none should be until `nwp_forecast_comparison.py` has actually fitted every arm and
`studies/nwp_forecast_comparison/report.md` holds real numbers — drawing a chart against a stub
report would risk a chart nobody checks against the numbers it is meant to illustrate, which the
plan's "printed-number guard" step exists to prevent.

When these are implemented, each chart follows the plan's "Charts" section and both the `dataviz`
skill and the bundled `dataviz` skill it supplements: the OCF-brand palette
(`plotting.ocf_theme`), `aria=False` on every data mark, SVG export through `svgo`, generators
labelled `A`-`F` and `W1`-`W3` only, values as a percentage of capacity, and every printed number
checked against `report.md` before the page is written.

1. `leaderboard_chart`: the day-1 leaderboard per technology, each arm's error with its interval,
   smaller is better, family and lead band marked.
2. `contrasts_chart`: the planned contrasts P1 to P4, with the two sides of each bracket drawn
   together.
3. `predictions_chart`: the XGBoost models' work — day-1 out-of-fold forecasts against measured
   output, one week per generator chosen by a stated rule, and per-generator errors per arm.
4. `error_by_day_chart`: error against day (0 to 3) for every product, with ENS's day-0 and day-1
   errors drawn so the brackets are visible.
5. `blends_chart`: each blend against ENS alone and against its control, at both bounds.

Run it with `uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py --report-dir
<dir> --output-dir <dir>`, once a real report exists.
"""

import argparse
import sys
from pathlib import Path

import altair as alt
import polars as pl


def leaderboard_chart(*, losses: pl.DataFrame, domain: str) -> alt.Chart:
    """Draw the day-1 leaderboard: every arm's own error with its interval, smaller is better.

    Args:
        losses: Per-row losses for one technology, day 1 only.
        domain: `solar` or `wind`.

    Returns:
        The chart.

    Raises:
        NotImplementedError: Always; see the module docstring.
    """
    raise NotImplementedError("nwp_forecast_charts is a skeleton; no report exists to chart yet.")


def contrasts_chart(*, contrasts: pl.DataFrame) -> alt.Chart:
    """Draw the planned contrasts P1 to P4, both sides of each bracket together.

    Args:
        contrasts: One row per planned contrast, its interval and its bracket side.

    Returns:
        The chart.

    Raises:
        NotImplementedError: Always; see the module docstring.
    """
    raise NotImplementedError("nwp_forecast_charts is a skeleton; no report exists to chart yet.")


def predictions_chart(*, predictions: pl.DataFrame, site: str) -> alt.Chart:
    """Draw one generator's day-1 out-of-fold forecasts against its measured output, one week.

    Args:
        predictions: Saved per-row predictions for every arm.
        site: The anonymised generator label (`A`-`F` or `W1`-`W3`) to chart.

    Returns:
        The chart.

    Raises:
        NotImplementedError: Always; see the module docstring.
    """
    raise NotImplementedError("nwp_forecast_charts is a skeleton; no report exists to chart yet.")


def error_by_day_chart(*, losses: pl.DataFrame, domain: str) -> alt.Chart:
    """Draw error against day (0 to 3) for every product, ENS's day-0 and day-1 errors marked.

    Args:
        losses: Per-row losses for one technology, every day.
        domain: `solar` or `wind`.

    Returns:
        The chart.

    Raises:
        NotImplementedError: Always; see the module docstring.
    """
    raise NotImplementedError("nwp_forecast_charts is a skeleton; no report exists to chart yet.")


def blends_chart(*, blend_verdicts: pl.DataFrame) -> alt.Chart:
    """Draw each blend against ENS alone and against its control, at both bounds.

    Args:
        blend_verdicts: One row per blend per bound, its interval against ENS and its guard.

    Returns:
        The chart.

    Raises:
        NotImplementedError: Always; see the module docstring.
    """
    raise NotImplementedError("nwp_forecast_charts is a skeleton; no report exists to chart yet.")


def main() -> int:
    """Parse arguments and report that no chart is implemented yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, required=True, help="Where report.md was written."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where SVGs are written.")
    parser.parse_args()
    print(
        "nwp_forecast_charts.py is a skeleton (see the module docstring); "
        "no chart has been implemented or drawn in this pass."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
