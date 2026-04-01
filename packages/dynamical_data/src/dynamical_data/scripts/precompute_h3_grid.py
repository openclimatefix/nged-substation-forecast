import h3.api.numpy_int as h3
import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry
from pathlib import Path
from geo.h3 import compute_h3_grid_weights

# Constants matching dynamical_data.processing
H3_RES = 5
GRID_SIZE = 0.25


def main():
    """Pre-computes the H3 grid for Great Britain and saves it as a Parquet file.

    This script avoids the 30-second penalty of generating the grid at runtime.
    """
    # Path resolution relative to this script
    dynamical_assets_path = Path(__file__).resolve().parent.parent.parent.parent / "assets"
    geo_assets_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "geo" / "assets"
    geojson_path = geo_assets_path / "england_scotland_wales.geojson"
    output_path = dynamical_assets_path / "gb_h3_grid.parquet"

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")

    print(f"Reading GeoJSON from {geojson_path}...")
    with open(geojson_path) as f:
        file_contents = f.read()

    shape: BaseGeometry = shapely.from_geojson(file_contents)

    print("Buffering shape (this may take ~30s)...")
    # Buffer by 0.25 degrees to catch islands/coasts
    shape = shape.buffer(0.25)

    print(f"Generating H3 cells at resolution {H3_RES}...")
    cells = h3.geo_to_cells(shape, res=H3_RES)
    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")

    print("Computing H3 grid weights...")
    df_with_counts = compute_h3_grid_weights(df, grid_size=GRID_SIZE, child_res=7)

    print(f"Saving pre-computed H3 grid to {output_path}...")
    df_with_counts.write_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
