from pathlib import Path

import polars as pl
import pytest
from studies.guards import check_no_missing, refuse_to_overwrite


def test_an_existing_output_is_not_overwritten(tmp_path: Path):
    existing = tmp_path / "losses.parquet"
    existing.write_text("published")

    with pytest.raises(FileExistsError, match="superseded"):
        refuse_to_overwrite(paths=[tmp_path / "report.md", existing])


def test_absent_outputs_pass(tmp_path: Path):
    refuse_to_overwrite(paths=[tmp_path / "report.md", tmp_path / "losses.parquet"])


def test_a_null_in_an_arms_column_raises():
    frame = pl.DataFrame({"ghi_a": [1.0, None], "ghi_b": [1.0, 2.0]})

    with pytest.raises(ValueError, match="ghi_a"):
        check_no_missing(frame=frame, columns=["ghi_a", "ghi_b"])


def test_a_not_a_number_in_an_arms_column_raises():
    frame = pl.DataFrame({"ghi_a": [1.0, float("nan")]})

    with pytest.raises(ValueError, match="ghi_a"):
        check_no_missing(frame=frame, columns=["ghi_a"])


def test_complete_columns_pass_and_unnamed_columns_are_not_checked():
    frame = pl.DataFrame({"ghi_a": [1.0, 2.0], "note": [None, "x"]})

    check_no_missing(frame=frame, columns=["ghi_a"])
