import importlib.resources
from functools import lru_cache
from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import ScalingParams

# The scaling parameters are stored in the dynamical_data package assets.
# We use importlib.resources for robust, environment-agnostic path resolution.
scaling_params_resource = importlib.resources.files("dynamical_data.assets").joinpath(
    "ecmwf_scaling_params.csv"
)


@lru_cache(maxsize=1)
def load_scaling_params(path: Path | None = None) -> pt.DataFrame[ScalingParams]:
    if path is not None:
        df = pl.read_csv(path).with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return ScalingParams.validate(df, drop_superfluous_columns=True)

    with importlib.resources.as_file(scaling_params_resource) as path:
        if not path.exists():
            # Fallback for if we are running from root or package
            alt_path = Path("packages/xgboost_forecaster/scaling_params.csv")
            if not alt_path.exists():
                # If it still doesn't exist, we might be in trouble, but let's try to find it
                raise FileNotFoundError(
                    f"Scaling params not found at {path}. Please set the XGBOOST_SCALING_PARAMS_PATH environment variable."
                )
            df = pl.read_csv(alt_path).with_columns(pl.col(pl.Float64).cast(pl.Float32))
            return ScalingParams.validate(df, drop_superfluous_columns=True)
        df = pl.read_csv(path).with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return ScalingParams.validate(df, drop_superfluous_columns=True)
