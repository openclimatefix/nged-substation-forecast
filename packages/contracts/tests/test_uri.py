"""Unit tests for the local-or-remote URI helpers in ``contracts._uri``.

Covers the local-filesystem behaviour (the remote ``s3://`` behaviour is exercised end-to-end
against a moto server in ``tests/test_s3_data_paths.py``).
"""

from pathlib import Path

import polars as pl
from contracts._uri import (
    delta_table_exists,
    ensure_local_parent,
    is_remote_uri,
    object_exists,
    uri_join,
)


def test_is_remote_uri():
    assert is_remote_uri("s3://bucket/key")
    assert not is_remote_uri("/home/jack/data/NWP")
    assert not is_remote_uri("relative/path")


def test_uri_join_local_and_remote():
    # Remote joins stay scheme-preserving and posix-style.
    assert uri_join("s3://bucket/data", "NGED", "metadata.parquet") == (
        "s3://bucket/data/NGED/metadata.parquet"
    )
    assert uri_join("s3://bucket/data/", "NWP") == "s3://bucket/data/NWP"
    # Local joins go through pathlib.
    assert uri_join("/srv/data", "NWP") == "/srv/data/NWP"


def test_ensure_local_parent_creates_local_dir(tmp_path: Path):
    target = tmp_path / "nested" / "dir" / "table"
    assert not target.parent.exists()
    ensure_local_parent(str(target))
    assert target.parent.is_dir()


def test_ensure_local_parent_noop_for_remote():
    # Must not raise or touch the filesystem for a remote URI.
    ensure_local_parent("s3://bucket/data/table")


def test_object_exists_local(tmp_path: Path):
    path = tmp_path / "metadata.parquet"
    assert not object_exists(str(path))
    pl.DataFrame({"x": [1]}).write_parquet(path)
    assert object_exists(str(path))


def test_delta_table_exists_local(tmp_path: Path):
    path = str(tmp_path / "table")
    assert not delta_table_exists(path)
    pl.DataFrame({"x": [1, 2]}).write_delta(path)
    assert delta_table_exists(path)
