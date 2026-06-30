"""Streaming a Polars ``LazyFrame`` into XGBoost one row-batch at a time.

``_prepare_features`` is the shared feature-prep helper used by ``forecaster.py``.

``LazyFrameBatchIter`` is **not currently used.** ``XGBoostForecaster`` bounds memory by
collecting one ``time_series_id`` at a time (a predicate Polars pushes down into the scans), which
is enough while training on a single NWP ensemble member. The iterator is kept for the **future**
work of training across many ensemble members (or many series in one booster), where even a single
booster's data will not fit in memory and must be streamed.

When that day comes, feed it a **cheap-to-slice source** — a ``pl.scan_parquet``/``scan_delta``
scan, or a frame already narrowed by a push-down predicate (e.g. one ``ensemble_member``). Do
**not** point it at the full feature-engineering lazy join: a row ``slice`` does not push through a
join, so each batch would re-run the entire join (and ``QuantileDMatrix`` sketches over several
passes), which neither bounds memory nor performs. See the Polars join push-down notes in
``docs/architecture/overview.md``.
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

    Not currently wired into ``XGBoostForecaster`` — see the module docstring for when and how to
    use it (the future multi-ensemble-member path), and what to feed it.

    Grouping-agnostic: it batches whatever ``LazyFrame`` it is given. The caller decides which rows
    feed which booster by passing an already-filtered ``LazyFrame``, and is responsible for filtering
    out null-label rows first so batch boundaries fall on clean rows. Only one batch is collected at
    a time.

    ``xgb.QuantileDMatrix`` iterates this object several full passes to sketch the quantiles, so
    every pass re-executes ``lf.slice(...).collect()``. That is cheap only when ``lf`` is a
    cheap-to-slice source (a parquet/Delta scan, where ``slice`` pushes into the scan); on a
    complex lazy join the slice cannot push through, so each pass re-runs the whole join.
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
