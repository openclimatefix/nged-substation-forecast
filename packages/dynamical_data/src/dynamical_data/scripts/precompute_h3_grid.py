import h3.api.numpy_int as h3
import polars as pl
import polars_h3 as plh3
import shapely
from shapely.geometry.base import BaseGeometry
from pathlib import Path

# Constants matching dynamical_data.processing
H3_RES = 5
GRID_SIZE = 0.25


def compute_h3_grid_weights(df: pl.DataFrame) -> pl.DataFrame:
    """Computes the proportion mapping for H3 grid cells.

    This function is duplicated from dynamical_data.processing to avoid circular imports
    during the pre-computation step.
    """
    return (
        df.with_columns(h3_res7=plh3.cell_to_children("h3_index", 7))
        .explode("h3_res7")
        .with_columns(
            nwp_lat=((plh3.cell_to_lat("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
            nwp_lng=((plh3.cell_to_lng("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lng"])
        .len()
        .with_columns(total=pl.col("len").sum().over("h3_index"))
        .with_columns(proportion=pl.col("len") / pl.col("total"))
    )


def main():
    """Pre-computes the H3 grid for Great Britain and saves it as a Parquet file.

    This script avoids the 30-second penalty of generating the grid at runtime.
    """
    # Path resolution relative to this script
    assets_path = Path(__file__).resolve().parent.parent.parent.parent / "assets"
    geojson_path = assets_path / "england_scotland_wales.geojson"
    output_path = assets_path / "gb_h3_grid.parquet"

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
    df_with_counts = compute_h3_grid_weights(df)

    print(f"Saving pre-computed H3 grid to {output_path}...")
    df_with_counts.write_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
