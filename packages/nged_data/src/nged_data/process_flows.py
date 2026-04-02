import math
from typing import Protocol

import patito as pt
import polars as pl
from contracts.data_schemas import UTC_DATETIME_DTYPE, SubstationFlows


class ParserStrategy(Protocol):
    """Protocol for CSV parsing strategies."""

    def can_parse(self, df: pl.DataFrame) -> bool:
        """Return True if this strategy can parse the given DataFrame."""
        ...

    def parse(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse the DataFrame and return it with standardized column names."""
        ...


class RegentStreetParser:
    """Parser for Regent Street format (unit=MVA)."""

    def can_parse(self, df: pl.DataFrame) -> bool:
        return "unit" in df.columns and "value" in df.columns and (df["unit"] == "MVA").all()

    def parse(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({"value": "MVA", "time": "timestamp"}, strict=False)


class MilfordHavenParser:
    """Parser for Milford Haven format (unit=MW)."""

    def can_parse(self, df: pl.DataFrame) -> bool:
        return "unit" in df.columns and "value" in df.columns and (df["unit"] == "MW").all()

    def parse(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({"value": "MW", "time": "timestamp"}, strict=False)


class StandardFormatParser:
    """Parser for standard NGED formats with various column names."""

    def can_parse(self, df: pl.DataFrame) -> bool:
        standard_cols = [
            "ValueDate",
            "Timestamp",
            "MW Inst",
            "MVA Inst",
            "Derived MVA",
            "Amps",
            "Volts",
        ]
        return any(col in df.columns for col in standard_cols)

    def parse(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(
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


def _compute_missing_mva(df: pl.DataFrame) -> pl.DataFrame:
    """Compute MVA from Current and Voltage if MW and MVA are missing.

    Apparent Power (MVA) = sqrt(3) * Voltage (kV) * Current (A) / 1000
    """
    if "MVA" in df.columns or "MW" in df.columns:
        return df

    # Check for current and voltage columns
    current_col = next((c for c in ["Current Inst", "Amps"] if c in df.columns), None)
    volts_col = next((c for c in ["Volts Inst", "Volts"] if c in df.columns), None)

    if current_col and volts_col:
        # Check if voltage is likely in Volts rather than kV
        # If max voltage > 1000, assume it's in Volts
        max_volts = df.select(pl.col(volts_col).max()).item()
        divisor = 1_000_000 if max_volts and max_volts > 1000 else 1000

        return df.with_columns(
            (math.sqrt(3) * pl.col(volts_col) * pl.col(current_col) / divisor).alias("MVA")
        )

    return df


def process_live_primary_substation_flows(
    csv_data: bytes,
) -> pt.DataFrame[SubstationFlows]:
    """Read a primary substation CSV and validate it against the schema."""
    df = pl.read_csv(csv_data)
    first_orig_rows = df.head()

    # Define available parsing strategies
    parsers: list[ParserStrategy] = [
        RegentStreetParser(),
        MilfordHavenParser(),
        StandardFormatParser(),
    ]

    # Try each parser until one succeeds
    parsed = False
    for parser in parsers:
        if parser.can_parse(df):
            df = parser.parse(df)
            parsed = True
            break

    if not parsed and ("unit" in df.columns and "value" in df.columns):
        raise ValueError(f"Unexpected unit in CSV: {df['unit'].unique().to_list()}")

    # Compute MVA if missing
    df = _compute_missing_mva(df)

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
