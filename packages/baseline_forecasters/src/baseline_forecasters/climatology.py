"""The climatology baseline: quantiles of past power at the same calendar month, time, and day type.

Climatology answers one question: at long lead times, does the weather ensemble know more than the
distribution of past power at that time of year, time of day, and day type? The forecaster stores
51 empirical quantiles of training power for each calendar cell, and emits them as 51 ensemble
members. See the package README for what each member means and for the caveats of the comparison.
"""

import itertools
import logging
from pathlib import Path
from typing import ClassVar, Final, Self

import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from ml_core.features import FeatureEngineer
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE

from baseline_forecasters._saved_model import clear_directory_and_write_meta, read_meta
from baseline_forecasters.nwp_run_rows import NwpRunRowsWithoutWeatherFeatureEngineer

logger = logging.getLogger(__name__)

CLIMATOLOGY_MEMBER_COUNT: Final[int] = 51
"""The number of ensemble members, and so of quantiles stored per calendar cell."""

CLIMATOLOGY_QUANTILE_LEVELS: Final[tuple[float, ...]] = tuple(
    (member + 0.5) / CLIMATOLOGY_MEMBER_COUNT for member in range(CLIMATOLOGY_MEMBER_COUNT)
)
"""The equiprobable quantile level of each member: ``(k + 0.5) / 51`` for member k from 0 to 50.

The metrics layer treats the members as an equiprobable sample, so the levels are equally spaced,
and member 25 is exactly the median.
"""

CLIMATOLOGY_QUANTILE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"power_quantile_member_{member:02d}" for member in range(CLIMATOLOGY_MEMBER_COUNT)
)
"""The lookup's quantile column names, in member order."""

_QUANTILE_COLUMN_TO_MEMBER: Final[dict[str, int]] = {
    column: member for member, column in enumerate(CLIMATOLOGY_QUANTILE_COLUMNS)
}

CLIMATOLOGY_POOLING_OFFSETS: Final[pl.DataFrame] = pl.DataFrame(
    {
        "month_offset": [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        "half_hour_offset": [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    },
    schema={"month_offset": pl.Int8, "half_hour_offset": pl.Int8},
)
"""The nine ``(month_offset, half_hour_offset)`` pairs that define a cell's neighbourhood.

Each training sample counts towards the cell it falls in and the eight cells one local month and one
local half-hour of day away, all with the same day type.
"""

_POOLING_SERIES_PER_BATCH: Final[int] = 100
"""How many series ``train`` pools at once.

Every pooled group sits inside one series, so batching changes no group's samples. The pooled
group-by holds about 65 bytes per pooled row, and pooling copies each sample nine times.
"""

_CELL_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "local_month",
    "local_half_hour_of_day",
    "local_is_weekend",
)

_LOOKUP_SCHEMA: Final[dict[str, pl.DataType | type[pl.DataType]]] = {
    "time_series_id": pl.Int32,
    "local_month": pl.Int8,
    "local_half_hour_of_day": pl.Int8,
    "local_is_weekend": pl.Boolean,
    **dict.fromkeys(CLIMATOLOGY_QUANTILE_COLUMNS, pl.Float32),
}

_LOOKUP_FILENAME: Final[str] = "climatology_quantiles.parquet"
"""Differs from ``time_series_metadata.parquet``, which ``save_to_mlflow`` adds to the directory."""


def _local_calendar_cell_keys(valid_time: pl.Expr, *, time_zone: str) -> dict[str, pl.Expr]:
    """The three calendar keys of a cell, derived from ``valid_time`` in local time.

    Both ``train`` and ``predict`` call this function, so the two cannot disagree on a cell. The
    keys are local because the substation load follows the local clock.

    Args:
        valid_time: A UTC datetime expression.
        time_zone: The IANA time zone the keys are computed in.

    Returns:
        Expressions named ``local_month`` (``Int8``, 1 to 12), ``local_half_hour_of_day`` (``Int8``,
        0 to 47), and ``local_is_weekend`` (``Boolean``, local Saturday or Sunday), ready to pass
        to ``with_columns`` as keyword arguments.
    """
    local = valid_time.dt.convert_time_zone(time_zone)
    return {
        "local_month": local.dt.month().cast(pl.Int8),
        "local_half_hour_of_day": (local.dt.hour() * 2 + local.dt.minute() // 30).cast(pl.Int8),
        "local_is_weekend": local.dt.weekday() >= 6,
    }


def _pooled_quantile_lookup(samples: pl.DataFrame) -> pl.DataFrame:
    """Reduce deduplicated samples to one row per populated cell, with its pooled sample count.

    Args:
        samples: Columns ``time_series_id``, ``valid_time``, and ``power``, one row per series and
            valid time.

    Returns:
        The ``_LOOKUP_SCHEMA`` columns plus ``pooled_sample_count``, the number of samples pooled
        into the cell.
    """
    keyed = samples.select(
        "time_series_id",
        "power",
        **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE),
    )
    # Polars' integer `%` is floor modulo, so December and January are neighbours, and so are
    # half-hours 47 and 0.
    pooled = keyed.join(CLIMATOLOGY_POOLING_OFFSETS, how="cross").select(
        "time_series_id",
        "power",
        "local_is_weekend",
        local_month=((pl.col("local_month") - 1 + pl.col("month_offset")) % 12 + 1).cast(pl.Int8),
        local_half_hour_of_day=(
            (pl.col("local_half_hour_of_day") + pl.col("half_hour_offset")) % 48
        ).cast(pl.Int8),
    )
    # "linear" is passed explicitly because Polars' default is "nearest". Polars rejects a list of
    # levels inside `group_by().agg`, so each level is its own expression.
    quantiles = [
        pl.col("power").quantile(level, "linear").cast(pl.Float32).alias(column)
        for level, column in zip(
            CLIMATOLOGY_QUANTILE_LEVELS, CLIMATOLOGY_QUANTILE_COLUMNS, strict=True
        )
    ]
    return pooled.group_by(*_CELL_KEY_COLUMNS).agg(*quantiles, pooled_sample_count=pl.len())


class ClimatologyForecaster(BaseForecaster):
    """Forecasts each target time as 51 quantiles of past power in the same calendar cell.

    A cell is ``(time_series_id, local month, local half-hour of day, local is-weekend)``. Each
    training sample counts towards nine cells: its own, and the cells one month and one half-hour
    either side, with the same weekend flag. ``train`` stores the 51 equiprobable quantiles of each
    cell's pooled samples, at levels ``(k + 0.5) / 51``. ``predict`` emits the quantiles as the
    ensemble members 0 to 50, with member 0 the lowest quantile. ``nwp_init_time`` is null on every
    row, because the forecaster consumes no weather.

    Bank holidays are ordinary days. A forecast row in a cell with no training sample anywhere in
    its neighbourhood is dropped, and ``predict`` logs a warning with the count.
    """

    MODEL_NAME = "climatology"
    MODEL_VERSION = 1
    CONFIG_CLASS: ClassVar[type[BaseForecasterConfig]] = BaseForecasterConfig

    feature_engineer: ClassVar[FeatureEngineer] = NwpRunRowsWithoutWeatherFeatureEngineer()

    def __init__(self, model_params: BaseForecasterConfig) -> None:
        """Check that the config selects no feature.

        Args:
            model_params: The config, whose ``selected_features`` must be empty.

        Raises:
            ValueError: ``model_params.selected_features`` is not empty. Climatology reads no
                feature, so a non-empty list would describe an experiment that is not the one
                running, and a power lag would widen the power scan for nothing.
        """
        super().__init__(model_params)
        if model_params.selected_features:
            raise ValueError(
                "ClimatologyForecaster reads no feature, so selected_features must be empty, but "
                f"got {sorted(model_params.selected_features)}."
            )
        self._trained_ids: list[int] = []
        self._lookup: pl.DataFrame = pl.DataFrame(schema=_LOOKUP_SCHEMA)

    @property
    def trained_time_series_ids(self) -> list[int]:
        """The sorted ``time_series_id``s that have at least one cell in the lookup."""
        return self._trained_ids

    def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None:
        """Store the 51 pooled quantiles of power for every populated calendar cell.

        The engineer repeats each target once per NWP run covering it, so the targets are
        deduplicated on ``(time_series_id, valid_time)`` before any quantile is taken. The pooling
        then runs over batches of series, so the nine copies of the samples exist for one batch at
        a time.

        Args:
            data: Features engineered by ``NwpRunRowsWithoutWeatherFeatureEngineer``. Only
                ``time_series_id``, ``valid_time``, and ``power`` are read.
            time_series_ids: The series to train. A series with no non-null ``power`` gets no
                lookup row and is absent from ``trained_time_series_ids``.
        """
        # Strip the Patito model: a model-bearing frame loses it on some operations and not others.
        plain: pl.LazyFrame = pl.LazyFrame._from_pyldf(data._ldf)
        samples = (
            plain.select("time_series_id", "valid_time", "power")
            .filter(pl.col("time_series_id").is_in(time_series_ids), pl.col("power").is_not_null())
            .unique(subset=["time_series_id", "valid_time"], keep="any")
            .collect(engine="streaming")
        )
        series_ids = sorted(samples["time_series_id"].unique().to_list())
        batches = [
            _pooled_quantile_lookup(samples.filter(pl.col("time_series_id").is_in(batch)))
            for batch in itertools.batched(series_ids, _POOLING_SERIES_PER_BATCH, strict=False)
        ]
        pooled = (pl.concat(batches) if batches else pl.DataFrame(schema=_LOOKUP_SCHEMA)).sort(
            _CELL_KEY_COLUMNS
        )
        self._lookup = pooled.select(list(_LOOKUP_SCHEMA))
        self._trained_ids = series_ids
        if pooled.height == 0:
            logger.warning("Climatology found no training samples for the requested series.")
            return
        sample_counts = pooled["pooled_sample_count"]
        logger.info(
            "Trained climatology on %d series and %d cells. Pooled samples per cell: minimum %d, "
            "median %.0f. Training valid times run from %s to %s.",
            len(series_ids),
            pooled.height,
            sample_counts.min(),
            sample_counts.median(),
            samples["valid_time"].min(),
            samples["valid_time"].max(),
        )

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        """Look up each row's calendar cell and unpivot its 51 quantiles into ensemble members.

        Only ``time_series_id``, ``power_fcst_init_time``, and ``valid_time`` are read from
        ``data``, so the observed power of the forecast period cannot reach the forecast. A row in
        a cell with no lookup row is dropped, and one warning names the dropped count and the
        series. Empty input gives an empty frame.

        Args:
            data: Features engineered by ``NwpRunRowsWithoutWeatherFeatureEngineer``.
            fold_id: The value stamped onto every row's ``fold_id`` column.

        Returns:
            One row per ``(time_series_id, power_fcst_init_time, valid_time, ensemble_member)``,
            with members 0 to 50 in order of increasing quantile level.
        """
        config = self.model_params
        # Strip the Patito model: a cross-model join raises, and the select drops most columns.
        plain: pl.LazyFrame = pl.LazyFrame._from_pyldf(data._ldf)
        rows = (
            plain.select("time_series_id", "power_fcst_init_time", "valid_time")
            .with_columns(
                **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE)
            )
            .collect(engine="streaming")
        )
        unseen = rows.join(self._lookup.select(_CELL_KEY_COLUMNS), on=_CELL_KEY_COLUMNS, how="anti")
        if unseen.height > 0:
            logger.warning(
                "Climatology dropped %d forecast rows whose calendar cell has no training sample "
                "in its neighbourhood. Affected time_series_ids: %s.",
                unseen.height,
                sorted(unseen["time_series_id"].unique().to_list()),
            )
        members = (
            rows.join(self._lookup, on=_CELL_KEY_COLUMNS, how="inner")
            .unpivot(
                on=list(CLIMATOLOGY_QUANTILE_COLUMNS),
                index=["time_series_id", "power_fcst_init_time", "valid_time"],
                variable_name="quantile_column",
                value_name="quantile_value",
            )
            .select(
                "valid_time",
                "time_series_id",
                "power_fcst_init_time",
                ensemble_member=pl.col("quantile_column").replace_strict(
                    _QUANTILE_COLUMN_TO_MEMBER, return_dtype=pl.Int8
                ),
                power_fcst=pl.col("quantile_value").cast(pl.Float32),
                nwp_init_time=pl.lit(None, dtype=UTC_DATETIME_DTYPE),
                power_fcst_model_name=pl.lit(self.MODEL_NAME),
                power_fcst_model_version=pl.lit(self.MODEL_VERSION, dtype=pl.Int16),
                ml_flow_experiment_id=pl.lit(config.ml_flow_experiment_id, dtype=pl.Int32),
                experiment_name=pl.lit(config.experiment_name),
                fold_id=pl.lit(fold_id),
            )
        )
        return PowerForecast.validate(members)

    def save(self, path: Path) -> None:
        """Replace ``path`` with ``meta.json`` and the quantile lookup."""
        clear_directory_and_write_meta(path=path, forecaster=self)
        self._lookup.write_parquet(path / _LOOKUP_FILENAME)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct a ClimatologyForecaster from the files that ``save`` wrote.

        The trained population comes from ``meta.json``, not from the lookup.
        """
        meta = read_meta(path)
        instance = cls(cls.CONFIG_CLASS.model_validate(meta["model_params"]))
        instance._trained_ids = meta["trained_time_series_ids"]
        instance._lookup = pl.read_parquet(path / _LOOKUP_FILENAME)
        return instance
