# Add a New Forecasting Model

To add a new forecasting model, follow these steps:

1.  Create a new package in `packages/` for the model (e.g., `packages/new_model_forecaster/`).
2.  Implement the model, ensuring it follows the Universal Model Interface.
3.  Register the model in the Hydra configuration (`conf/model/`).
4.  Update the Dagster pipeline to include the new model.
5.  Run the pipeline to train and evaluate the model.
