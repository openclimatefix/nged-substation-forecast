from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationLocations

from nged_data.schemas import CkanResource


def simplify_substation_name(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .str.replace_all(" Primary Transformer Flows", "")
        .str.replace_all(r"\d{2,}(?i)kv", "")  # the (?i) means "ignore case"
        .str.replace_all("S/Stn", "", literal=True)  # e.g. "Sheepbridge 11/6 6kv S/Stn"
        .str.replace_all("/", "", literal=True)
        .str.replace_all("132", "")
        .str.replace_all("66", "")
        .str.replace_all("33", "")
        .str.replace_all("11", "")
        .str.replace_all(r"6\.6(?i)kv", "")
        # e.g. "Sheffield Road 33 11 6 6kv S Stn", "Sheepbridge 11/6 6kv S/Stn", "Sandy Lane 33 6 6kv S Stn"
        .str.replace_all("6 6kv", "")
        # Live primary flows tend to end the name with just "Primary".
        .str.replace_all("Primary", "")
        .str.replace_all("S Stn", "")  # Assuming "S Stn" is short for substation.
        .str.replace_all(" Kv", "")  # e.g. "Infinity Park 33 11 Kv S Stn"
        .str.replace_all("  ", " ")
        .str.strip_chars()
        .str.strip_chars(".")
    )


def join_location_table_to_live_primaries(
    locations: pt.DataFrame[SubstationLocations],
    live_primaries: list[CkanResource],
) -> pl.DataFrame:
    live_primaries_df = pl.DataFrame(live_primaries)

    # Append a "simple_name" column to each dataframe:
    live_primaries_df = live_primaries_df.with_columns(simple_name=simplify_substation_name("name"))
    locations = locations.with_columns(simple_name=simplify_substation_name("substation_name"))

    # Load manual mapping
    csv_filename = (
        Path(__file__).parent
        / "map_substation_names_in_live_primary_flows_to_substation_names_in_location_table.csv"
    )
    name_mapping = pl.read_csv(csv_filename)

    # Replace any simple names in the live primaries dataframe with the name from the manual
    # mapping, if such a mapping exists.
    live_primaries_df = live_primaries_df.join(
        name_mapping,
        how="left",
        left_on="simple_name",
        right_on="simplified_substation_name_in_live_flows",
    )
    live_primaries_df = live_primaries_df.with_columns(
        simple_name=pl.coalesce("simplified_substation_name_in_location_table", "simple_name")
    )

    # Rename the name columns
    live_primaries_df = live_primaries_df.rename({"name": "substation_name_in_live_primaries"})
    locations_renamed_col = locations.rename(
        {"substation_name": "substation_name_in_location_table"}
    )
    return live_primaries_df.join(locations_renamed_col, on="simple_name").sort(by="simple_name")
