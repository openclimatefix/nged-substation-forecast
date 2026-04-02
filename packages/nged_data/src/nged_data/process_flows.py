import math

import patito as pt
import polars as pl
from contracts.data_schemas import UTC_DATETIME_DTYPE, SubstationFlows


def process_live_primary_substation_flows(csv_data: bytes) -> pt.DataFrame[SubstationFlows]:
    """Read a primary substation CSV and validate it against the schema."""
    df = pl.read_csv(csv_data)
    first_orig_rows = df.head()

    # The CSV column names vary between NGED license areas:
    # East Midlands (e.g. Abington)  : ValueDate,                       MVA,                     Volts
    # West Midlands (e.g. Albrighton): ValueDate, Amps,                 MVA, MVAr,      MW,      Volts
    # South Wales   (e.g. Aberaeron) : ValueDate, Current Inst, Derived MVA, MVAr Inst, MW Inst, Volts Inst
    # South West    (e.g. Filton Dc) : ValueDate, Current Inst,              MVAr Inst, MW Inst, Volts Inst
    # New format    (e.g. Regent St) : site, time, unit, value
    if "unit" in df.columns and "value" in df.columns:
        if (df["unit"] == "MVA").all():
            # Handle Regent Street primary substation (in the East Midlands), which uses a completely
            # different CSV structure. See `example_csv_data/regent-street.csv`.
            df = df.rename({"value": "MVA"}, strict=False)
        elif (df["unit"] == "MW").all():
            # Handle milford-haven-grid.csv.
            df = df.rename({"value": "MW"}, strict=False)
        else:
            raise ValueError(f"Unexpected unit in CSV: {df['unit'].unique().to_list()}")

    df = df.rename(
        {
            "time": "timestamp",
            "ValueDate": "timestamp",
            "Timestamp": "timestamp",
            "MW Inst": "MW",
            "MVAr Inst": "MVAr",
            "MVA Inst": "MVA",
            "Derived MVA": "MVA",
        },
        strict=False,
    )

    # If we don't have MVA or MW, but we have Current and Voltage, we can compute MVA.
    # Apparent Power (MVA) = sqrt(3) * Voltage (kV) * Current (A) / 1000
    if "MVA" not in df.columns and "MW" not in df.columns:
        # Check for current and voltage columns
        current_col = None
        volts_col = None

        for col in ["Current Inst", "Amps"]:
            if col in df.columns:
                current_col = col
                break

        for col in ["Volts Inst", "Volts"]:
            if col in df.columns:
                volts_col = col
                break

        if current_col and volts_col:
            # Check if voltage is likely in Volts rather than kV
            # If max voltage > 1000, assume it's in Volts
            max_volts = df.select(pl.col(volts_col).max()).item()
            divisor = 1_000_000 if max_volts and max_volts > 1000 else 1000

            df = df.with_columns(
                (math.sqrt(3) * pl.col(volts_col) * pl.col(current_col) / divisor).alias("MVA")
            )

    # Ensure MW, MVA, MVAr, and ingested_at are present before validation.
    # If they are missing from the source CSV, fill them with null.
    for col in ["MW", "MVA", "MVAr"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

    if "ingested_at" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(UTC_DATETIME_DTYPE).alias("ingested_at"))

    columns = [col for col in SubstationFlows.columns if col in df.columns]
    df = df.select(columns)
    df = df.cast({col: SubstationFlows.dtypes[col] for col in columns})
    df = df.sort("timestamp")

    try:
        return SubstationFlows.validate(df, allow_missing_columns=True)
    except Exception as e:
        e.add_note(f"First rows of CSV data, before processing: {first_orig_rows}")
        raise
