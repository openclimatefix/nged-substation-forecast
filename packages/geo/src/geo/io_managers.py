import polars as pl
from dagster import InputContext, OutputContext
from dagster._core.storage.upath_io_manager import UPathIOManager
from typing import Any, TYPE_CHECKING
import pickle

if TYPE_CHECKING:
    from upath import UPath


class CompositeIOManager(UPathIOManager):
    extension = ""

    def dump_to_path(self, context: OutputContext, obj: Any, path: "UPath"):
        if isinstance(obj, pl.DataFrame):
            obj.write_parquet(str(path.with_suffix(".parquet")))
        else:
            with path.with_suffix(".pkl").open("wb") as f:
                pickle.dump(obj, f)

    def load_from_path(self, context: InputContext, path: "UPath") -> Any:
        if path.with_suffix(".parquet").exists():
            return pl.read_parquet(str(path.with_suffix(".parquet")))
        else:
            with path.with_suffix(".pkl").open("rb") as f:
                return pickle.load(f)
