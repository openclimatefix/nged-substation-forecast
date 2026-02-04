import polars as pl


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
