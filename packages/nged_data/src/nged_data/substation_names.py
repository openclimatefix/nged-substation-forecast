from collections import namedtuple

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationLocations
from rapidfuzz import fuzz, process


def process_substation_names_from_substation_locations(
    locations: pt.DataFrame[SubstationLocations],
) -> pl.DataFrame:
    return locations.with_columns(
        name_filtered=(
            pl.col("substation_name")
            .str.replace_all(r"\d{2,}(?i)kv", "")
            .str.replace_all("/", "", literal=True)
            .str.replace_all("132", "")
            .str.replace_all("66", "")
            .str.replace_all("33", "")
            .str.replace_all("11", "")
            .str.replace_all(r"6\.6(?i)kv", "")
            .str.replace_all("Primary", "")
            .str.replace_all("S Stn", "")
            .str.replace_all("  ", " ")
            .str.strip_chars()
            .str.strip_chars(".")
        )
    ).sort("name_filtered")


def shorten_substation_name(name: str) -> str:
    name = name.replace(" Transformer Flows", "")
    name = name.replace(" 11kV", "").replace(" 11Kv", "")  # Historical data
    name = name.replace(" Primary", "")  # Live data
    return name


Match = namedtuple("Match", field_names=["canonical_name", "score"])


def fuzzy_match_two_lists_of_strings(
    canonical_strings: list[str], test_strings: list[str]
) -> dict[str, list[Match]]:
    return {
        test_string: fuzzy_match_test_string(canonical_strings, test_string)
        for test_string in test_strings
    }


def fuzzy_match_test_string(canonical_strings: list[str], test_string: str) -> list[Match]:
    matches: list = process.extract(
        test_string,
        choices=canonical_strings,
        scorer=fuzz.WRatio,
        limit=3,
    )
    return [Match(canonical_name=m[0], score=m[1]) for m in matches]
