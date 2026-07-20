"""Unit test for ``geo.great_britain.load.load_gb_boundary``.

The shipped GB coastline GeoJSON takes ~30s to buffer, so we monkeypatch a tiny stand-in polygon
in its place. This still exercises the real read -> parse -> buffer path end to end, just on a
trivially small geometry, keeping the suite fast.
"""

from pathlib import Path

import pytest
from geo.great_britain import load

_SMALL_GEOJSON: str = (
    '{"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}'
)


def test_load_gb_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda self: _SMALL_GEOJSON)

    boundary = load.load_gb_boundary()

    assert boundary.is_valid
    assert not boundary.is_empty
    # The 0.25-degree buffer expands the unit square (area 1.0) outward.
    assert boundary.area > 1.0
