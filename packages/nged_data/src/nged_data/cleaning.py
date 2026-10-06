"""Flagging NGED's power telemetry that should not be trained or forecast on.

`flag_nged_power` is where cleaning rules live. The `clean_nged_power_data` Dagster asset calls it
with the whole raw `power_time_series` table and writes the result to the
`cleaned_power_time_series` Delta table, recording a `CleaningProvenance` in the write's commit so
a later run can tell whether the table is already up to date.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import DeltaTable
from ml_core.repro import ABSENT, UNKNOWN, get_git_info

log = logging.getLogger(__name__)

SUBSTATION_TYPES: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
"""The `TimeSeriesMetadata.substation_type` values that mean "a substation", as opposed to a
generator or a battery at an `HV Customer` or `EHV Customer` connection. The `substation_zero`
rule applies only to these."""

CLEANING_CODE_HASH: Final[str] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""The SHA-256 of this module's own source file, computed at import.

`clean_nged_power_data` skips a rebuild only when this hash matches the one recorded in the
cleaned table's newest write commit, so any edit to a cleaning rule forces a rebuild on the next
run, whether or not the edit is committed. The rules must live in this module, or the hash must be
extended to cover every file they live in.
"""

GIT_SHA_ENV_VAR: Final[str] = "GIT_SHA"
"""Environment variable holding the git SHA of the code, set when the container image is built.
`current_git_sha` falls back to it, because a container has no git repository to ask."""

_PROVENANCE_KEY_PREFIX: Final[str] = "cleaning_"
_HISTORY_WINDOW: Final[int] = 10
"""How many recent Delta commits `read_cleaning_provenance` walks back through to find the newest
`WRITE`. Every vacuum adds two commits after the write, so a small window is enough."""


@dataclass(frozen=True)
class CleaningProvenance:
    """What a cleaned table was built from, recorded in the write's Delta commit.

    Attributes:
        raw_table_id: The raw table's Delta table id. It guards against a deleted and rebuilt raw
            table, whose versions restart at 0.
        raw_version: The raw table's Delta version that the cleaning read.
        code_hash: `CLEANING_CODE_HASH` when the cleaning ran.
        git_sha: The git SHA of the code that did the cleaning. For provenance only; the skip
            compares `code_hash`.
    """

    raw_table_id: str
    raw_version: int
    code_hash: str
    git_sha: str

    def to_commit_metadata(self) -> dict[str, str]:
        """Return the `custom_metadata` to record on the Delta write commit."""
        return {
            f"{_PROVENANCE_KEY_PREFIX}raw_table_id": self.raw_table_id,
            f"{_PROVENANCE_KEY_PREFIX}raw_version": str(self.raw_version),
            f"{_PROVENANCE_KEY_PREFIX}code_hash": self.code_hash,
            f"{_PROVENANCE_KEY_PREFIX}git_sha": self.git_sha,
        }


def current_git_sha() -> str:
    """Return the git SHA of the running code, for `CleaningProvenance.git_sha`.

    Uses `ml_core.repro.get_git_info`, and when that returns `UNKNOWN` (as it does in a container)
    falls back to a non-empty `GIT_SHA` environment variable, else `UNKNOWN`. The fallback lives
    here rather than in `get_git_info` so the MLflow provenance tags of the other stages do not
    change.

    Returns:
        A git SHA, or `UNKNOWN`.
    """
    sha = get_git_info()["git_sha"]
    if sha != UNKNOWN:
        return sha
    return os.environ.get(GIT_SHA_ENV_VAR) or UNKNOWN


def read_cleaning_provenance(
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> CleaningProvenance | None:
    """Read the `CleaningProvenance` from the newest `WRITE` commit of the cleaned table.

    A vacuum adds commits after the write, so this walks back through the table's recent history
    to the newest `WRITE`. Never raises: a missing table, an unreadable table, no `WRITE` commit
    in the window, or a missing key all read as "no provenance", which always means rebuild.

    Args:
        table_uri: Path or URI of the `cleaned_power_time_series` Delta table.
        storage_options: delta-rs object-store options for a remote `table_uri`.

    Returns:
        The provenance, or `None` when there is none.
    """
    try:
        options = typeddict_to_dict(storage_options) or {}
        if not DeltaTable.is_deltatable(str(table_uri), storage_options=options):
            return None
        history = DeltaTable(str(table_uri), storage_options=options).history(limit=_HISTORY_WINDOW)
        write_commit = next(
            (commit for commit in history if commit.get("operation") == "WRITE"), None
        )
        if write_commit is None or f"{_PROVENANCE_KEY_PREFIX}raw_table_id" not in write_commit:
            return None
        return CleaningProvenance(
            raw_table_id=str(write_commit[f"{_PROVENANCE_KEY_PREFIX}raw_table_id"]),
            raw_version=int(write_commit[f"{_PROVENANCE_KEY_PREFIX}raw_version"]),
            code_hash=str(write_commit[f"{_PROVENANCE_KEY_PREFIX}code_hash"]),
            git_sha=str(write_commit[f"{_PROVENANCE_KEY_PREFIX}git_sha"]),
        )
    except Exception:
        log.warning(f"Could not read cleaning provenance from {table_uri}.", exc_info=True)
        return None


def cleaned_power_provenance_tag(
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> str:
    """Return the value an MLflow run stamps to record which cleaned power table it read.

    The cleaned table's own Delta versions are vacuumed after a few hours, so a run cannot be
    replayed with `scan_delta(version=N)`. The stamp names what the cleaning read instead:
    replaying means re-running the cleaning over that raw version at the cleaning's git SHA, which
    can differ from the run's own SHA because an unchanged table is not rebuilt.

    Args:
        table_uri: Path or URI of the `cleaned_power_time_series` Delta table.
        storage_options: delta-rs object-store options for a remote `table_uri`.

    Returns:
        `"raw_table_id=<id>;raw_version=<n>;git_sha=<sha>"`, or `ABSENT` when there is no
        provenance. Never raises.
    """
    provenance = read_cleaning_provenance(table_uri, storage_options)
    if provenance is None:
        return ABSENT
    return (
        f"raw_table_id={provenance.raw_table_id};raw_version={provenance.raw_version};"
        f"git_sha={provenance.git_sha}"
    )


def flag_nged_power(
    power: pt.LazyFrame[PowerTimeSeries],
    metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pt.LazyFrame[CleanedPowerTimeSeries]:
    """Flag the rows of NGED's power telemetry that cleaning rejects, by adding `drop_reason`.

    **Where this sits.** `clean_nged_power_data` calls this function with the whole raw power
    table and the roster, about four times a day, and writes the result to the cleaned table.
    Every reader of power then sees only the rows whose `drop_reason` is null.

    **How to add a rule.** Write a boolean Polars expression that is true for a row to drop. Add it
    as one more `.when(...).then(pl.lit("<reason>"))` branch of the chain below, and add the reason
    to `contracts.power_schemas.DROP_REASONS`. The first matching branch wins.

    **The contract a rule keeps.** Return every input row exactly once. Never change
    `time_series_id` or `time`. Flag rows rather than deleting them. Do not call `collect` inside
    this function.

    **Logging.** The asset reports to Dagster, per reason, the rows and series flagged, the
    minimum and maximum flagged power, and the first and last flagged time, so there is no logging
    code to write. Output from Python's `logging` reaches only the step's captured stderr.

    The one rule today, `substation_zero`, flags a reading of exactly 0 from a `Primary`, `BSP`,
    or `GSP` series. A substation almost never truly reads zero, so a zero is almost always a
    telemetry fault. A series missing from the roster has no `substation_type` and is not flagged.

    Args:
        power: The whole raw power table.
        metadata: The `TimeSeriesMetadata` roster.

    Returns:
        Every row of `power`, in the same order, with a `drop_reason` column: null for a row that
        passed, otherwise the name of the rule that flagged it.
    """
    # Join the roster columns the rules need onto the power rows. The roster is stripped of its
    # Patito model so Polars' cross-subclass join check does not reject the join.
    roster = (
        pl.DataFrame._from_pydf(metadata._df).select("time_series_id", "substation_type").lazy()
    )
    joined = pl.LazyFrame._from_pyldf(power._ldf).join(roster, on="time_series_id", how="left")

    flagged = joined.with_columns(
        drop_reason=pl.when(
            (pl.col("power") == 0) & pl.col("substation_type").is_in(SUBSTATION_TYPES)
        )
        .then(pl.lit("substation_zero"))
        # A bare `pl.lit(None)` has the `Null` dtype, which would not match the contract's String.
        .otherwise(pl.lit(None, dtype=pl.String))
    ).select("time_series_id", "time", "power", "drop_reason")
    return pt.LazyFrame.from_existing(flagged).set_model(CleanedPowerTimeSeries)
