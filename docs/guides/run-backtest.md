# Run a Backtest

To run a backtest, follow these steps:

1.  Ensure your environment is set up with `uv sync`.
2.  Run the Dagster UI: `uv run dagster dev`.
3.  Trigger the backtest pipeline for the desired time range.
4.  Monitor the progress in the Dagster UI.
5.  The backtest results will be saved to the configured storage location and tracked in MLflow.
