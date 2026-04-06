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
            # Ensure the directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            obj.write_parquet(str(path.with_suffix(".parquet")))
        else:
            # Ensure the directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".pkl").open("wb") as f:
                pickle.dump(obj, f)

    def load_from_path(self, context: InputContext, path: "UPath") -> Any:
        parquet_path = path.with_suffix(".parquet")
        pkl_path = path.with_suffix(".pkl")

        if parquet_path.exists():
            return pl.read_parquet(str(parquet_path))
        elif pkl_path.exists():
            with pkl_path.open("rb") as f:
                return pickle.load(f)
        else:
            # Fallback for existing files that might not have an extension
            if path.exists() and not path.is_dir():
                try:
                    return pl.read_parquet(str(path))
                except Exception:
                    with path.open("rb") as f:
                        return pickle.load(f)
            raise FileNotFoundError(
                f"No file found for {path} (checked {parquet_path} and {pkl_path})"
            )
