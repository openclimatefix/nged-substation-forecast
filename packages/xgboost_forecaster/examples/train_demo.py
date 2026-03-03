"""Demo script to train XGBoost on a few substations."""

import logging
import random

import polars as pl
from xgboost_forecaster import (
    XGBoostForecaster,
    get_substation_metadata,
    prepare_data_for_substation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def run_demo(num_substations: int = 5):
    log.info("Fetching substation metadata...")
    metadata = get_substation_metadata()

    # Select a few substations with available power data
    available_subs = metadata.filter(pl.col("parquet_filename").is_not_null())

    if available_subs.is_empty():
        log.error("No substations with power data found!")
        return

    sample_subs = random.sample(
        available_subs["substation_name_in_location_table"].to_list(),
        min(num_substations, len(available_subs)),
    )

    log.info(f"Selected substations for demo: {sample_subs}")

    for sub_name in sample_subs:
        log.info(f"--- Processing {sub_name} ---")
        try:
            data = prepare_data_for_substation(sub_name, metadata)

            if data.is_empty():
                log.warning(f"Skipping {sub_name} due to lack of overlapping data.")
                continue

            # Train model
            forecaster = XGBoostForecaster()
            forecaster.train(data)

            # Predict (on the same data for demo)
            preds = forecaster.predict(data)

            # Calculate simple MAE
            mae = (data["power_mw"] - preds).abs().mean()
            log.info(f"Finished {sub_name}. MAE on training data: {mae:.4f}")

        except Exception:
            log.exception(f"Failed to train model for {sub_name}")


if __name__ == "__main__":
    run_demo()
