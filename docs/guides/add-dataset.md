# Add a New Dataset

To add a new dataset, follow these steps:

1.  Create a new package in `packages/` for the dataset (e.g., `packages/new_dataset_data/`).
2.  Implement the ingestion logic, ensuring it follows the project's data contracts.
3.  Register the new asset in `src/nged_substation_forecast/defs/`.
4.  Update the Dagster pipeline to include the new asset.
5.  Run the pipeline to verify the data ingestion.
