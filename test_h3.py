import h3
from shapely.geometry import Polygon

# Create a simple polygon
poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])

print(f"h3 version: {h3.__version__}")

try:
    cells = h3.geo_to_cells(poly, res=5)
    print(f"geo_to_cells success! Found {len(cells)} cells.")
except Exception as e:
    print(f"geo_to_cells failed: {e}")

try:
    cells = h3.polygon_to_cells(poly, res=5)
    print(f"polygon_to_cells success! Found {len(cells)} cells.")
except Exception as e:
    print(f"polygon_to_cells failed: {e}")
