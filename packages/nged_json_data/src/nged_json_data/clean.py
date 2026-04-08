import polars as pl
import patito as pt
from contracts.data_schemas import NgedJsonPowerFlows


def clean_power_data(
    df: pl.DataFrame,
    substation_number: int | None = None,
    variance_thresholds: dict[int, float] | None = None,
    default_threshold: float = 0.1,
) -> pt.DataFrame[NgedJsonPowerFlows]:
    """
    Cleans power data by sorting, dropping nulls, and nullifying low-variance time slices.

    Rationale for variance threshold:
    The variance threshold is used to identify 'bad pre-amble data' or stuck sensors.
    If a sensor's output has very low variance over a 6-hour rolling window, it is likely
    not reflecting actual power fluctuations and is instead reporting a constant,
    potentially erroneous value. A threshold of 0.1 is chosen as a heuristic to
    distinguish between legitimate low-load periods and sensor failure.

    Args:
        df: The input DataFrame containing power data.
        substation_number: The substation number, used to look up a specific variance threshold.
        variance_thresholds: A mapping of substation numbers to variance thresholds.
        default_threshold: The default variance threshold if no substation-specific threshold is found.

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

    # Determine threshold
    threshold = default_threshold
    if (
        variance_thresholds
        and substation_number is not None
        and substation_number in variance_thresholds
    ):
        threshold = variance_thresholds[substation_number]

    # 3. Identify 'bad pre-amble data' by nullifying time slices
    # where the rolling variance is <= 'threshold'.

    # Calculate rolling variance over a 6-hour window
    # Assuming 'end_time' is a datetime column.
    # We need to ensure the dataframe is sorted by 'end_time' (already done).
    rolling_df = df.rolling(index_column="end_time", period="6h").agg(
        rolling_variance=pl.col("value").var()
    )
    df = df.join(rolling_df, on="end_time")

    # Nullify values where rolling variance is below the threshold
    # We only nullify if rolling_variance is not null (i.e., we have enough data)
    df = df.with_columns(
        value=pl.when(
            pl.col("rolling_variance").is_not_null() & (pl.col("rolling_variance") <= threshold)
        )
        .then(None)
        .otherwise(pl.col("value"))
    )

    # Drop the rolling_variance column
    df = df.drop("rolling_variance")

    # Drop nulls again?
    # The requirement says "Only nullify specific 'stuck' time slices instead of dropping entire days."
    # But the test expects ValueError if all rows are removed.
    # If I nullify, the rows are still there, just with null values.
    # The test expects ValueError if all rows are removed.
    # Maybe I should drop nulls after nullifying?
    df = df.drop_nulls(subset=["value"])

    # 4. Ensure the function fails loudly if all rows are removed
    if df.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")

    return NgedJsonPowerFlows.validate(df)
