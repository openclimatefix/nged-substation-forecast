from pathlib import Path

import pytest

from studies import era5_grid


def test_the_first_year_starts_in_the_first_usable_month():
    assert era5_grid.first_date_of(year=era5_grid.FIRST_YEAR) == (
        f"{era5_grid.FIRST_YEAR}-{era5_grid.FIRST_MONTH_OF_FIRST_YEAR:02d}-01"
    )


def test_a_later_year_starts_in_january():
    assert era5_grid.first_date_of(year=era5_grid.FIRST_YEAR + 1) == (
        f"{era5_grid.FIRST_YEAR + 1}-01-01"
    )


def test_the_suffix_goes_before_the_extension(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(era5_grid, "OUTPUT_SUFFIX", "_refresh")

    assert era5_grid.suffixed(Path("a/beam_diffuse_era5.parquet")) == Path(
        "a/beam_diffuse_era5_refresh.parquet"
    )


def test_an_empty_suffix_leaves_the_path_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(era5_grid, "OUTPUT_SUFFIX", "")

    assert era5_grid.suffixed(Path("a/x.parquet")) == Path("a/x.parquet")
