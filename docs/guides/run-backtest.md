# Run a Cross-Validation Backtest

The backtesting system uses **expanding-window cross-fold validation** to evaluate model performance in a way that faithfully mirrors how the system will behave in production.

## Cross-Validation Fold Structure

We use 5 folds with an expanding training window and a fixed one-year validation window:

| Fold | Train       | Validate |
|------|-------------|----------|
| 1    | 2020 – 2021 | 2022     |
| 2    | 2020 – 2022 | 2023     |
| 3    | 2020 – 2023 | 2024     |
| 4    | 2020 – 2024 | 2025     |
| 5    | 2020 – 2025 | 2026     |

A time series is only included in a given fold if it has valid data for the entire validation year and at least 6 months of training data. (For example, a time series with data only from early 2024 onwards would not be included in folds 1, 2, or 3.)

We use an expanding (not sliding) window to maximise data available for data-hungry algorithms like neural nets. This means fold-to-fold comparisons confound "more data" with "algorithmic improvement", which is fine — we aggregate across folds to report a single leaderboard figure.

## Running the Backtest

1. Ensure your environment is set up: `uv sync`
2. Start the Dagster UI: `uv run dg dev`
3. Materialise the `cv_power_forecasts` asset for the desired fold(s).
4. Materialise the `cv_metrics` asset to compute evaluation metrics.
5. Results are tracked in MLflow; open the MLflow UI to compare experiments.

## Metrics

Metrics are computed across multiple time-slice categories and reported to MLflow:

**Deterministic metrics:**
- Normalised mean bias error (MBE)
- Normalised mean absolute error (MAE)
- Root mean squared error (RMSE) — the most critical metric for power grids, as it penalises large misses heavily

**Probabilistic metrics:**
- Pinball loss (quantile loss), averaged across target quantiles
- PICP (Prediction Interval Coverage Probability) — e.g. 80% of observations should fall within the P10–P90 band
- CRPS (Continuous Ranked Probability Score) — the probabilistic equivalent of MAE
- Spread-Skill Ratio — compares ensemble spread to ensemble-mean RMSE; should be 1.0 for a well-calibrated ensemble

**Time-slice filters** (to emphasise the periods that matter most for grid management):
- 0–6 h (nowcasting / intraday)
- 6–36 h (day-ahead)
- Day 2–7 (short/medium range)
- Day 8–14 (extended range)
- Peak events — top 5% highest-demand half-hours
