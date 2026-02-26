"""Plot forecast vs ground truth for a substation."""

import logging
import matplotlib.pyplot as plt
from xgboost_forecaster.data import get_substation_metadata, prepare_training_data
from xgboost_forecaster.model import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def plot_substation_forecast(sub_name: str = "Lawford 33 11kv S Stn"):
    log.info(f"Fetching metadata and preparing data for {sub_name}...")
    metadata = get_substation_metadata()

    try:
        data = prepare_training_data(sub_name, metadata)
        if data.is_empty():
            log.error(f"No overlapping data found for {sub_name}.")
            return

        log.info(f"Training model for {sub_name} using time-based split...")
        # Use time_split=True so we test on the most recent data
        model, metrics = train_model(data, time_split=True, test_size=0.2)

        test_timestamps = metrics["test_timestamps"]
        y_test = metrics["y_test"]
        y_pred = metrics["y_pred"]

        log.info("Generating plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(test_timestamps, y_test, label="Actual Power (MW)", marker=".", alpha=0.7)
        plt.plot(test_timestamps, y_pred, label="Predicted Power (MW)", marker=".", alpha=0.7)

        plt.title(f"XGBoost Forecast vs Actual: {sub_name}")
        plt.xlabel("Timestamp")
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"forecast_{sub_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename)
        log.info(f"Plot saved to {filename}")

    except Exception as e:
        log.exception(f"Failed to plot forecast for {sub_name}: {e}")


if __name__ == "__main__":
    # You can change the substation name here
    plot_substation_forecast("Lawford 33 11kv S Stn")
