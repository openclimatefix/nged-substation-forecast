import logging
from pathlib import Path

import shapely
from shapely.geometry.base import BaseGeometry

_LOG = logging.getLogger(__name__)


def load_gb_boundary() -> BaseGeometry:
    """Loads the boundary geometry for Great Britain from a local GeoJSON file.

    The boundary is buffered to ensure that coastal substations and nearby islands
    are included in the resulting H3 grid without spatial distortion.
    """
    geojson_path = Path(__file__).parent / "england_scotland_wales.geojson"
    _LOG.info(f"Loading GB boundary from {geojson_path}")
    file_contents = geojson_path.read_text()
    shape: BaseGeometry = shapely.from_geojson(file_contents)
    return shape.buffer(distance=0.25)  # This takes about 30 seconds.
