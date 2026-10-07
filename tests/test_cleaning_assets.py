"""Materialisation tests for the ``clean_nged_power_data`` Dagster asset.

Every assertion on the cleaned table reads it back through ``pl.scan_delta``, so a dtype that
breaks Delta reads fails the test.
"""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from _cleaned_power_test_data import write_metadata
from contracts.common import UTC_DATETIME_DTYPE
from contracts.settings import Settings
from dagster import DagsterInstance, ExecuteInProcessResult, materialize
from delta_store.cleaned_power_time_series import read_cleaning_provenance
from deltalake import DeltaTable

from nged_substation_forecast.defs import cleaning_assets
from nged_substation_forecast.defs.cleaning_assets import clean_nged_power_data, current_git_sha

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point every managed data path at a temp dir, and return the resulting settings."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setenv("DATA_PATH_DELIVERY", str(tmp_path))
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    return Settings()


def _raw_frame(rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    """Build raw power rows from ``(time_series_id, half_hour_index, power)`` tuples."""
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([row[0] for row in rows], dtype=pl.Int32),
            "time": pl.Series([_T0 + timedelta(minutes=30 * row[1]) for row in rows]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([row[2] for row in rows], dtype=pl.Float32),
        }
    )


def _write_raw(settings: Settings, rows: list[tuple[int, int, float]]) -> None:
    _raw_frame(rows).write_delta(settings.power_time_series_data_path, mode="append")


def _write_metadata(settings: Settings, substation_types: dict[int, str]) -> None:
    write_metadata(settings.metadata_path, substation_types)


_RAW_ROWS = [(1, 0, 5.0), (1, 1, 0.0), (1, 2, 0.0), (2, 0, 0.0), (2, 1, 3.0)]
"""Series 1 is a `Primary` substation with two zeros; series 2 is a generator with one zero."""


@pytest.fixture
def raw_data(env: Settings) -> Settings:
    _write_raw(env, _RAW_ROWS)
    _write_metadata(env, {1: "Primary", 2: "HV Customer"})
    return env


def _materialize(instance: DagsterInstance, force: bool = False) -> ExecuteInProcessResult:
    run_config = {"ops": {"clean_nged_power_data": {"config": {"force": force}}}}
    result = materialize([clean_nged_power_data], instance=instance, run_config=run_config)
    assert result.success
    return result


def _metadata(result: ExecuteInProcessResult) -> dict[str, object]:
    (materialization,) = result.asset_materializations_for_node("clean_nged_power_data")
    return {key: value.value for key, value in materialization.metadata.items()}


def _cleaned(settings: Settings) -> pl.DataFrame:
    return (
        pl.scan_delta(settings.cleaned_power_time_series_data_path)
        .collect()
        .sort("time_series_id", "time")
    )


def _cleaned_version(settings: Settings) -> int:
    return DeltaTable(settings.cleaned_power_time_series_data_path).version()


def test_asset_writes_every_raw_row_with_the_flags(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    metadata = _metadata(_materialize(dagster_instance))

    cleaned = _cleaned(raw_data)
    assert cleaned.height == len(_RAW_ROWS)
    assert cleaned["drop_reason"].to_list() == [
        None,
        "substation_zero",
        "substation_zero",
        None,
        None,
    ]
    assert metadata["n_rows"] == len(_RAW_ROWS)
    assert metadata["n_rows_kept"] == 3
    assert metadata["n_time_series"] == 2
    assert metadata["drop_reason/substation_zero/n_rows"] == 2
    assert metadata["drop_reason/substation_zero/n_time_series"] == 1


def test_asset_reports_per_reason_stats(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    def fake_flag(power: pt.LazyFrame, metadata: pt.DataFrame) -> pt.LazyFrame:
        return pt.LazyFrame.from_existing(
            pl.LazyFrame._from_pyldf(power._ldf).with_columns(
                drop_reason=pl.when(pl.col("power") > 1)
                .then(pl.lit("substation_zero"))
                .otherwise(pl.lit(None, dtype=pl.String))
            )
        )

    monkeypatch.setattr(cleaning_assets, "flag_nged_power", fake_flag)

    metadata = _metadata(_materialize(dagster_instance))

    prefix = "drop_reason/substation_zero"
    assert metadata[f"{prefix}/n_rows"] == 2
    assert metadata[f"{prefix}/n_time_series"] == 2
    assert metadata[f"{prefix}/min_power"] == 3.0
    assert metadata[f"{prefix}/max_power"] == 5.0
    assert metadata[f"{prefix}/first_time"] == _T0.isoformat()
    assert metadata[f"{prefix}/last_time"] == (_T0 + timedelta(minutes=30)).isoformat()


def test_absent_raw_table_reports_zero_rows_without_raising(
    env: Settings, dagster_instance: DagsterInstance
) -> None:
    metadata = _metadata(_materialize(dagster_instance))

    assert metadata["n_rows"] == 0
    assert not DeltaTable.is_deltatable(env.cleaned_power_time_series_data_path)


def test_a_vacuum_failure_still_leaves_the_run_successful(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    def failing_vacuum(self: DeltaTable, **kwargs: object) -> list[str]:
        raise OSError("transient object-store error")

    monkeypatch.setattr(DeltaTable, "vacuum", failing_vacuum)
    reported: list[str] = []
    monkeypatch.setattr(
        cleaning_assets,
        "report_asset_degradation",
        lambda asset_name, exc: reported.append(asset_name),
    )

    metadata = _metadata(_materialize(dagster_instance))

    assert metadata["vacuum_failed"] is True
    assert reported == ["clean_nged_power_data"]
    assert _cleaned(raw_data).height == len(_RAW_ROWS)


def test_a_second_run_with_nothing_changed_is_skipped(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    version = _cleaned_version(raw_data)

    metadata = _metadata(_materialize(dagster_instance))

    assert metadata["skipped"] is True
    assert _cleaned_version(raw_data) == version


def test_a_new_raw_commit_triggers_a_rebuild(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    _write_raw(raw_data, [(1, 3, 7.0)])

    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert _cleaned(raw_data).height == len(_RAW_ROWS) + 1
    provenance = read_cleaning_provenance(raw_data.cleaned_power_time_series_data_path)
    assert provenance is not None
    assert provenance.raw_version == DeltaTable(raw_data.power_time_series_data_path).version()


def test_a_rebuilt_raw_table_with_the_same_version_triggers_a_rebuild(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    version = _cleaned_version(raw_data)
    shutil.rmtree(raw_data.power_time_series_data_path)
    _write_raw(raw_data, _RAW_ROWS)  # the same number of commits, so the same Delta version

    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert _cleaned_version(raw_data) > version


def test_a_metadata_change_with_no_new_power_triggers_a_rebuild(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    _write_metadata(raw_data, {1: "Primary", 2: "Primary"})  # series 2 is now a substation

    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert metadata["drop_reason/substation_zero/n_rows"] == 3
    assert _cleaned(raw_data)["drop_reason"].null_count() == 2


def test_a_different_code_hash_triggers_a_rebuild(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    version = _cleaned_version(raw_data)
    monkeypatch.setattr(cleaning_assets, "CLEANING_CODE_HASH", "an-edited-rule")

    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert _cleaned_version(raw_data) > version
    provenance = read_cleaning_provenance(raw_data.cleaned_power_time_series_data_path)
    assert provenance is not None
    assert provenance.code_hash == "an-edited-rule"


def test_a_cleaned_table_with_no_provenance_is_rebuilt(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _raw_frame(_RAW_ROWS).with_columns(drop_reason=pl.lit(None, dtype=pl.String)).write_delta(
        raw_data.cleaned_power_time_series_data_path,
        delta_write_options={"partition_by": ["time_series_id"]},
    )

    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert _cleaned(raw_data).height == len(_RAW_ROWS)


def test_force_rebuilds_when_nothing_has_changed(
    raw_data: Settings, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    version = _cleaned_version(raw_data)

    metadata = _metadata(_materialize(dagster_instance, force=True))

    assert "skipped" not in metadata
    assert _cleaned_version(raw_data) > version


def test_a_failed_rebuild_is_retried_by_the_next_run(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    _write_raw(raw_data, [(1, 3, 7.0)])
    real_flag = cleaning_assets.flag_nged_power

    def failing_flag(power: pt.LazyFrame, metadata: pt.DataFrame) -> pt.LazyFrame:
        raise RuntimeError("a cleaning rule is broken")

    monkeypatch.setattr(cleaning_assets, "flag_nged_power", failing_flag)
    run_config = {"ops": {"clean_nged_power_data": {"config": {"force": False}}}}
    failed = materialize(
        [clean_nged_power_data],
        instance=dagster_instance,
        run_config=run_config,
        raise_on_error=False,
    )
    assert not failed.success
    assert _cleaned(raw_data).height == len(_RAW_ROWS)  # the last good table survives

    monkeypatch.setattr(cleaning_assets, "flag_nged_power", real_flag)
    metadata = _metadata(_materialize(dagster_instance))

    assert "skipped" not in metadata
    assert _cleaned(raw_data).height == len(_RAW_ROWS) + 1


def test_the_recorded_git_sha_falls_back_to_the_environment_variable(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    unknown = {"git_sha": cleaning_assets.UNKNOWN, "git_dirty": cleaning_assets.UNKNOWN}
    monkeypatch.setattr(cleaning_assets, "get_git_info", lambda: unknown)
    monkeypatch.setenv("GIT_SHA", "abc123")

    _materialize(dagster_instance)

    provenance = read_cleaning_provenance(raw_data.cleaned_power_time_series_data_path)
    assert provenance is not None
    assert provenance.git_sha == "abc123"


def test_current_git_sha_prefers_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cleaning_assets, "get_git_info", lambda: {"git_sha": "abc", "git_dirty": "false"}
    )
    monkeypatch.setenv("GIT_SHA", "from_env")
    assert current_git_sha() == "abc"


@pytest.mark.parametrize("env_value", [None, ""])
def test_current_git_sha_is_unknown_without_git_or_env(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    unknown = {"git_sha": cleaning_assets.UNKNOWN, "git_dirty": cleaning_assets.UNKNOWN}
    monkeypatch.setattr(cleaning_assets, "get_git_info", lambda: unknown)
    if env_value is None:
        monkeypatch.delenv("GIT_SHA", raising=False)
    else:
        monkeypatch.setenv("GIT_SHA", env_value)
    assert current_git_sha() == cleaning_assets.UNKNOWN


def test_metadata_fingerprint_ignores_row_order_but_not_values() -> None:
    metadata = pl.DataFrame({"time_series_id": [1, 2, 3], "substation_type": ["a", "b", "c"]})
    changed = metadata.with_columns(substation_type=pl.Series(["a", "b", "z"]))

    assert cleaning_assets.metadata_fingerprint(metadata) == cleaning_assets.metadata_fingerprint(
        metadata.reverse()
    )
    assert cleaning_assets.metadata_fingerprint(metadata) != cleaning_assets.metadata_fingerprint(
        changed
    )


def test_the_asset_cleans_exactly_the_raw_version_it_records(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    _write_raw(raw_data, [(1, 3, 7.0)])  # raw version 1, one row more than version 0
    real_delta_table = cleaning_assets.DeltaTable

    class _LaggingDeltaTable(real_delta_table):  # type: ignore[misc, valid-type]
        """Reports one version less than the table's real current version."""

        def version(self) -> int:
            return super().version() - 1

    monkeypatch.setattr(cleaning_assets, "DeltaTable", _LaggingDeltaTable)

    _materialize(dagster_instance)

    assert _cleaned(raw_data).height == len(_RAW_ROWS)
    provenance = read_cleaning_provenance(raw_data.cleaned_power_time_series_data_path)
    assert provenance is not None
    assert provenance.raw_version == 0


def test_an_unknown_drop_reason_fails_the_run_and_leaves_the_cleaned_table_alone(
    raw_data: Settings, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    _materialize(dagster_instance)
    version = _cleaned_version(raw_data)

    def bad_flag(power: pt.LazyFrame, metadata: pt.DataFrame) -> pt.LazyFrame:
        return pt.LazyFrame.from_existing(
            pl.LazyFrame._from_pyldf(power._ldf).with_columns(
                drop_reason=pl.lit("not_a_real_reason", dtype=pl.String)
            )
        )

    monkeypatch.setattr(cleaning_assets, "flag_nged_power", bad_flag)

    failed = materialize(
        [clean_nged_power_data],
        instance=dagster_instance,
        run_config={"ops": {"clean_nged_power_data": {"config": {"force": True}}}},
        raise_on_error=False,
    )

    assert not failed.success
    assert _cleaned_version(raw_data) == version
