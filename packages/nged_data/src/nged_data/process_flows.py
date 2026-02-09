from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows


def process_live_primary_substation_flows(csv_path: Path) -> pt.DataFrame[SubstationFlows]:
    """Read a primary substation CSV and validate it against the schema."""
    df: pl.DataFrame = pl.read_csv(csv_path)
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
            "Derived MVA": "MVA",
        },
        strict=False,
    )

    columns = [col for col in SubstationFlows.columns if col in df.columns]
    df = df.select(columns)
    df = df.cast({col: SubstationFlows.dtypes[col] for col in columns})
    df = df.sort("timestamp")

    try:
        return SubstationFlows.validate(df, allow_missing_columns=True)
    except Exception as e:
        raise RuntimeError(f"First rows in {csv_path}, before processing: {first_orig_rows}") from e
