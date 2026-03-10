import math

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows


class MissingCorePowerVariablesError(ValueError):
    """Raised when a substation CSV lacks both MW and MVA data."""

    pass


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
            df = df.with_columns(
                (math.sqrt(3) * pl.col(volts_col) * pl.col(current_col) / 1000).alias("MVA")
            )

    columns = [col for col in SubstationFlows.columns if col in df.columns]
    df = df.select(columns)
    df = df.cast({col: SubstationFlows.dtypes[col] for col in columns})
    df = df.sort("timestamp")

    try:
        return SubstationFlows.validate(df, allow_missing_columns=True)
    except Exception as e:
        # If the error is about missing MW or MVA, raise a specific exception
        # so we can catch it and ignore it gracefully.
        if "must contain at least one of 'MW' or 'MVA'" in str(e):
            raise MissingCorePowerVariablesError("Missing core power variables (MW/MVA)") from e
        raise RuntimeError(f"First rows of CSV data, before processing: {first_orig_rows}") from e
