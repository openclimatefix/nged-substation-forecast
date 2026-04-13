"""Domain-specific exceptions for the NGED substation forecast project.

This module centralizes custom exception classes to provide more granular
error handling and debugging context. These exceptions are designed to be
caught by Dagster's execution engine or higher-level pipeline logic to
handle data quality issues or operational failures gracefully.
"""


class NGEDDataValidationError(Exception):
    """Raised when raw NGED data fails Patito schema validation."""
