"""Streaming a Polars ``LazyFrame`` into XGBoost one row-batch at a time.

This is the memory-bounded training path: instead of collecting the whole ``AllFeatures``
frame (and letting XGBoost build a second, uncompressed ``Float32`` ``DMatrix`` copy),
``LazyFrameBatchIter`` feeds fixed-size row batches to an ``xgb.QuantileDMatrix``, which keeps
the data as compressed 8-bit quantile bins. Only one batch is ever resident in memory.
"""

from collections.abc import Callable
from typing import Any

import polars as pl
import xgboost as xgb


def _prepare_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Return a Float32 DataFrame containing only the feature columns.

    String, Categorical, and Enum columns are encoded as integer codes before casting,
    so XGBoost treats them as ordinal numerics. Nulls are preserved as NaN, which XGBoost
    handles natively as missing values. The Patito model is stripped from the result (zero-copy)
    so XGBoost's data iterator sees a plain ``pl.DataFrame``, which it accepts.
    """
    exprs = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            exprs.append(pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Float32).alias(col))
        else:
            exprs.append(pl.col(col).cast(pl.Float32).alias(col))
    return pl.DataFrame._from_pydf(df.select(exprs)._df)


class LazyFrameBatchIter(xgb.DataIter):
    """Streams a Polars ``LazyFrame`` to XGBoost one row-batch at a time.

    Grouping-agnostic: it batches whatever ``LazyFrame`` it is given. The caller decides which
    rows feed which booster (per ``time_series_id`` today; per time-series-*type* in a future
    model that trains one booster across many series) by passing an already-filtered ``LazyFrame``.
    Only one batch is collected into memory at a time.

    The caller is responsible for filtering out rows with a null label before constructing the
    iterator, so batch boundaries fall on clean rows.

    ``xgb.QuantileDMatrix`` iterates this object several full passes to sketch the quantiles, so
    every pass re-executes ``lf.slice(...).collect()`` on the lazy plan — bounded memory is bought
    with repeated computation. See ``docs/architecture/overview.md``.
    """

    def __init__(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        *,
        label_col: str = "power",
        batch_size: int,
    ) -> None:
        self._lf = lf
        self._feature_cols = feature_cols
        self._label_col = label_col
        self._batch_size = batch_size
        self._offset = 0
        super().__init__()

    def reset(self) -> None:
        """Rewind to the first batch (XGBoost calls this between sketch passes)."""
        self._offset = 0

    def next(self, input_data: Callable[..., Any]) -> bool:
        """Hand the next row-batch to XGBoost; return False once the frame is exhausted."""
        batch = self._lf.slice(self._offset, self._batch_size).collect()
        if batch.is_empty():
            return False
        features = _prepare_features(batch, self._feature_cols)
        label = batch[self._label_col].cast(pl.Float32)
        input_data(data=features, label=label)
        self._offset += self._batch_size
        return True
