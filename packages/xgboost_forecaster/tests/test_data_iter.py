import patito as pt
import polars as pl
import xgboost as xgb

from xgboost_forecaster._data_iter import LazyFrameBatchIter

_FEATURE_COLS = ["a", "b"]


def _frame(n: int, ts_ids: list[int] | None = None) -> pl.DataFrame:
    """Minimal feature frame with two float features and a power label."""
    ts_ids = ts_ids or [1]
    rows = [
        {"time_series_id": ts, "a": float(i), "b": float(n - i), "power": float(i % 7)}
        for ts in ts_ids
        for i in range(n)
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("a").cast(pl.Float32),
        pl.col("b").cast(pl.Float32),
        pl.col("power").cast(pl.Float32),
    )


def test_iter_yields_every_row_across_multiple_batches() -> None:
    df = _frame(250)
    data_iter = LazyFrameBatchIter(df.lazy(), _FEATURE_COLS, batch_size=100)
    dmat = xgb.QuantileDMatrix(data_iter)
    assert dmat.num_row() == 250
    assert dmat.num_col() == len(_FEATURE_COLS)
    assert dmat.feature_names == _FEATURE_COLS


def test_iter_single_batch_when_batch_size_exceeds_rows() -> None:
    df = _frame(40)
    data_iter = LazyFrameBatchIter(df.lazy(), _FEATURE_COLS, batch_size=100_000)
    dmat = xgb.QuantileDMatrix(data_iter)
    assert dmat.num_row() == 40


def test_iter_spans_multiple_time_series_ids_into_one_matrix() -> None:
    """Forward-looking: a future model trains one booster across many series via one iterator."""
    df = _frame(30, ts_ids=[1, 2, 3])
    data_iter = LazyFrameBatchIter(df.lazy(), _FEATURE_COLS, batch_size=25)
    dmat = xgb.QuantileDMatrix(data_iter)
    assert dmat.num_row() == 90


def test_iter_accepts_patito_lazyframe() -> None:
    """The iterator must handle the Patito-modelled LazyFrame that train() passes in."""
    df = _frame(60)
    lf = pt.LazyFrame.from_existing(df.lazy())
    data_iter = LazyFrameBatchIter(lf, _FEATURE_COLS, batch_size=25)
    dmat = xgb.QuantileDMatrix(data_iter)
    assert dmat.num_row() == 60
