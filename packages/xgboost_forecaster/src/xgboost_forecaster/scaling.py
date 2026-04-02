from functools import lru_cache
from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import ScalingParams
from contracts.settings import PROJECT_ROOT

# The scaling parameters are stored in the dynamical_data package assets.
scaling_params_path = PROJECT_ROOT / "packages/dynamical_data/assets/ecmwf_scaling_params.csv"


@lru_cache(maxsize=1)
def load_scaling_params(path: Path | None = None) -> pt.DataFrame[ScalingParams]:
    if path is not None:
        df = pl.read_csv(path).with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return ScalingParams.validate(df, drop_superfluous_columns=True)

    if not scaling_params_path.exists():
        raise FileNotFoundError(
            f"Scaling params not found at {scaling_params_path}. Please check the path."
        )
    df = pl.read_csv(scaling_params_path).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    return ScalingParams.validate(df, drop_superfluous_columns=True)
