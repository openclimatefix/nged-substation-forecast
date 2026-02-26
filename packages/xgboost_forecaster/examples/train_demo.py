"""Demo script to train XGBoost on a few substations."""

import logging
import random
import polars as pl
from xgboost_forecaster.data import get_substation_metadata, prepare_training_data
from xgboost_forecaster.model import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def run_demo(num_substations: int = 5):
    log.info("Fetching substation metadata...")
    metadata = get_substation_metadata()

    # Select a few substations with available power data
    # Filter for those that have a parquet file (we know they are in the directory)
    available_subs = metadata.filter(pl.col("parquet_filename").is_not_null())

    if available_subs.is_empty():
        log.error("No substations with power data found!")
        return

    sample_subs = random.sample(
        available_subs["substation_name_in_location_table"].to_list(),
        min(num_substations, len(available_subs)),
    )

    log.info(f"Selected substations for demo: {sample_subs}")

    results = {}

    for sub_name in sample_subs:
        log.info(f"--- Processing {sub_name} ---")
        try:
            data = prepare_training_data(sub_name, metadata)

            if data.is_empty():
                log.warning(f"Skipping {sub_name} due to lack of overlapping data.")
                continue

            # Sanity checks
            log.info(f"Data sanity check for {sub_name}:")
            log.info(f"  Rows: {len(data)}")
            log.info(f"  Power range: {data['power_mw'].min()} to {data['power_mw'].max()} MW")
            log.info(f"  Timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")

            if data["power_mw"].null_count() > 0:
                log.warning(f"  Found {data['power_mw'].null_count()} null power values.")

            if (data["power_mw"] > 1000).any() or (data["power_mw"] < -1000).any():
                log.warning("  Power values outside expected range (-1000, 1000) MW!")

            # Train model
            model, metrics = train_model(data)
            results[sub_name] = metrics

        except Exception as e:
            log.exception(f"Failed to train model for {sub_name}: {e}")

    log.info("--- Demo Results Summary ---")
    for sub, metrics in results.items():
        log.info(f"{sub}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    run_demo()
