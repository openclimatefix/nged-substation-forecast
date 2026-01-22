"""Dagster definitions for the NGED substation forecast project."""

from pathlib import Path

from dagster import Definitions, definitions, load_from_defs_folder


@definitions
def defs() -> Definitions:
    """Load all Dagster definitions from the defs folder.

    Returns:
        Definitions: The combined Dagster definitions.
    """
    return load_from_defs_folder(path_within_project=Path(__file__).parent)
