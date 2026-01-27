from typing import Callable

import polars as pl


def find_one_match[T](predicate: Callable[[T], bool], haystack: list[T]) -> T:
    """Find exactly one match in haystack. Raise a ValueError if haystack is empty, or if there is
    more than 1 match."""
    if len(haystack) == 0:
        raise ValueError("haystack is empty!")
    filtered = list(filter(predicate, haystack))
    if len(filtered) != 1:
        raise ValueError(f"Found {len(filtered)} matches when we were expecting exactly 1!")
    return filtered[0]


def change_dataframe_column_names_to_snake_case(df: pl.DataFrame) -> pl.DataFrame:
    new_col_names = {col: to_snake_case(col) for col in df.columns}
    return df.rename(new_col_names)


def to_snake_case(s: str) -> str:
    return s.lower().replace(" ", "_")
