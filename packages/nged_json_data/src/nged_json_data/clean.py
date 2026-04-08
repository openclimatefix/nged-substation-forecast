import polars as pl
import patito as pt
from contracts.data_schemas import NgedJsonPowerFlows


def clean_power_data(
    df: pl.DataFrame, variance_threshold: float = 0.1
) -> pt.DataFrame[NgedJsonPowerFlows]:
    """
    Cleans power data by sorting, dropping nulls, and removing low-variance time slices.

    Rationale for variance threshold:
    The variance threshold is used to identify 'bad pre-amble data' or stuck sensors.
    If a sensor's output has very low variance over a period (e.g., a day), it is likely
    not reflecting actual power fluctuations and is instead reporting a constant,
    potentially erroneous value. A threshold of 0.1 is chosen as a heuristic to
    distinguish between legitimate low-load periods and sensor failure.

    Args:
        df: The input DataFrame containing power data.
        variance_threshold: The variance threshold below which data is considered 'bad'.

    Returns:
        A cleaned Polars DataFrame.

    Raises:
        ValueError: If all rows are removed by the filtering process.
    """
    # 1. Sort by 'end_time'
    df = df.sort("end_time")

    # 2. Drop any rows where 'value' is null
    df = df.drop_nulls(subset=["value"])

    if df.is_empty():
        raise ValueError("All rows were removed after dropping nulls.")

    # 3. Identify 'bad pre-amble data' by throwing away contiguous time slices
    # (e.g., days) where the variance is <= 'variance_threshold'.
    # Stop throwing away days when the variance exceeds the threshold.

    # Assuming 'end_time' is a datetime column.
    # We need to group by day.

    # Create a 'date' column for grouping
    df = df.with_columns(date=pl.col("end_time").dt.truncate("1d"))

    # Calculate variance per day
    daily_variance = df.group_by("date").agg(variance=pl.col("value").var())

    # Identify days to keep (variance > threshold)
    days_to_keep = daily_variance.filter(pl.col("variance") > variance_threshold).select("date")

    # Filter the original dataframe
    cleaned_df = df.join(days_to_keep, on="date", how="inner").drop("date")

    # 5. Ensure the function fails loudly if all rows are removed
    if cleaned_df.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")

    return NgedJsonPowerFlows.validate(cleaned_df)
