"""The store behind the `ml-experiment` skill: experiment ids, prediction sets, and losses.

Each ml-experiment has an id, a five-digit integer that orders experiments by the order in which
they were started, and a slug that describes it. The experiment keeps its predictions in
`data/ml-experiments/<id>-<slug>/predictions/<name>/`, one parquet file and a `manifest.json` per
named set. A set is never overwritten, so a later experiment can use it as the benchmark to beat.
`read_predictions` refuses a set whose row set differs from the caller's, because two scores on
different rows are not a comparison.

Every function here runs in research and fails fast: it raises, and never degrades.
"""

import json
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypedDict
from uuid import uuid4

import polars as pl
from contracts.settings import PROJECT_ROOT

from studies.sources import ML_EXPERIMENTS_DATA_DIR, ML_EXPERIMENTS_DOCS_DIR

PredictionKindType = Literal["quantiles", "members", "stage1"]
"""What a prediction set holds: calibrated quantiles, ensemble members, or stage 1 outputs."""


class PredictionManifest(TypedDict):
    """The four keys of a prediction set's `manifest.json`."""

    kind: PredictionKindType
    row_set_id: str
    commit: str
    dirty: bool


EXPERIMENT_ID_WIDTH: Final[int] = 5
"""An experiment id is zero-padded to this many digits, so folder names sort in id order."""

IDS_FOLDER: Final[str] = ".ids"
"""The folder of `data/ml-experiments/` holding one marker folder per reserved id."""

SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
"""An experiment slug is lower-case words of letters and digits joined by single hyphens."""

SET_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+([-_][a-z0-9]+)*$")
"""A prediction set's name, such as `era5-pretrained` or `era5-pretrained-v2`."""

MANIFEST_NAME: Final[str] = "manifest.json"
"""The manifest's file name inside a prediction set's folder."""

PREDICTIONS_NAME: Final[str] = "predictions.parquet"
"""The parquet file's name inside a prediction set's folder."""

HOLDOUT_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The first instant of the holdout, which every experiment keeps out of its predictions.

No experiment scores a valid time on or after this date, so a finalist can still be scored once on
data that no experiment has seen.
"""

FOLD_A_TUESDAYS_V1: Final[str] = "fold_a_tuesdays_v1"
"""The id of the first shared benchmark row set, the one the stage 1 pilot scores.

ECMWF ENS runs at 00:00 UTC every 7 days from 2025-07-01 to 2026-06-30 (53 runs, all Tuesdays),
with a forecast origin at 09:00 UTC, all 51 members, valid times to about 14.6 days, and the 9
generator series (6 solar farms and 3 wind farms). The run stride is `SAMPLE_STRIDE_DAYS` in
`studies/quick_wins/quick_wins_inputs.py`.
"""


def _experiment_ids_in(*, folder: Path) -> list[int]:
    """Return the id of every child of `folder` whose name starts with five digits.

    Args:
        folder: A folder of `<id>-<slug>` or `<id>` children. A missing folder holds none.

    Returns:
        The ids, in no particular order.
    """
    if not folder.is_dir():
        return []
    prefix = re.compile(rf"^(\d{{{EXPERIMENT_ID_WIDTH}}})(-|$)")
    return [int(match[1]) for child in folder.iterdir() if (match := prefix.match(child.name))]


def reserve_experiment(
    *,
    slug: str,
    data_dir: Path = ML_EXPERIMENTS_DATA_DIR,
    docs_dir: Path = ML_EXPERIMENTS_DOCS_DIR,
) -> int:
    """Reserve the next experiment id and create the experiment's data folder.

    The next id is one more than the largest id in `data_dir/.ids` and in `docs_dir`, so an id is
    never reused, even after the data folder is lost. The reservation is
    the creation of the bare-id marker `data_dir/.ids/<id>` by `Path.mkdir(exist_ok=False)`, which
    is atomic on a local filesystem. Reserving the bare id, and not the id with the slug, is what
    stops two sessions that pick different slugs from taking one id.

    Args:
        slug: The experiment's slug, matching `SLUG_PATTERN`.
        data_dir: The folder of experiments' data.
        docs_dir: The folder of experiments' `result.md` pages.

    Returns:
        The reserved id.

    Raises:
        ValueError: If `slug` does not match `SLUG_PATTERN`.
    """
    if not SLUG_PATTERN.fullmatch(slug):
        msg = f"slug {slug!r} must be lower-case words joined by single hyphens"
        raise ValueError(msg)
    markers = data_dir / IDS_FOLDER
    markers.mkdir(parents=True, exist_ok=True)
    while True:
        known = [
            *_experiment_ids_in(folder=markers),
            *_experiment_ids_in(folder=docs_dir),
        ]
        experiment_id = max(known, default=0) + 1
        try:
            (markers / f"{experiment_id:0{EXPERIMENT_ID_WIDTH}d}").mkdir()
        except FileExistsError:
            continue
        (data_dir / f"{experiment_id:0{EXPERIMENT_ID_WIDTH}d}-{slug}").mkdir()
        return experiment_id


def _experiment_dir(*, experiment_id: int, data_dir: Path) -> Path:
    """Return the data folder of the experiment with this id.

    Args:
        experiment_id: The id that `reserve_experiment` returned.
        data_dir: The folder of experiments' data.

    Returns:
        The experiment's folder, `<id>-<slug>`.

    Raises:
        ValueError: If no folder, or more than one folder, carries the id.
    """
    matches = sorted(data_dir.glob(f"{experiment_id:0{EXPERIMENT_ID_WIDTH}d}-*"))
    if len(matches) != 1:
        msg = f"expected one folder for experiment {experiment_id} in {data_dir}, found {matches}"
        raise ValueError(msg)
    return matches[0]


def _require_utc_valid_time(*, frame: pl.DataFrame) -> None:
    """Raise unless the frame has a `valid_time` column of UTC datetimes.

    Args:
        frame: A prediction frame.

    Raises:
        ValueError: If `valid_time` is missing or is not a UTC datetime. The holdout check needs
            the column, and a naive datetime cannot be compared with the holdout instant.
    """
    dtype = frame.schema.get("valid_time")
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        msg = f"a prediction frame needs a UTC datetime `valid_time` column, found {dtype}"
        raise ValueError(msg)


def _require_before_holdout(*, frame: pl.DataFrame) -> None:
    """Raise if any `valid_time` falls on or after `HOLDOUT_START`."""
    if frame.select((pl.col("valid_time") >= HOLDOUT_START).any()).item():
        msg = f"a valid_time is on or after the holdout start {HOLDOUT_START:%Y-%m-%d}"
        raise ValueError(msg)


def _code_state() -> tuple[str, bool]:
    """Return the repository's current commit and whether the working tree has uncommitted changes.

    Returns:
        The full commit hash of `HEAD`, and whether `git status --porcelain` lists any change.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def write_predictions(
    *,
    experiment_id: int,
    name: str,
    frame: pl.DataFrame,
    kind: PredictionKindType,
    row_set_id: str,
    data_dir: Path = ML_EXPERIMENTS_DATA_DIR,
) -> Path:
    """Write one named prediction set, never overwriting an existing set.

    The parquet file and the manifest are written into a temporary folder that is renamed into
    place, so an interrupted write leaves no set at `name`, only a hidden staging folder. A re-run
    is saved under a new name, such as `name-v2`.

    Args:
        experiment_id: The id that `reserve_experiment` returned.
        name: The set's name, matching `SET_NAME_PATTERN`.
        frame: The predictions, with a UTC datetime `valid_time` column, none on or after the
            holdout start.
        kind: What the set holds.
        row_set_id: The id of the row set the frame covers, such as `FOLD_A_TUESDAYS_V1`.
        data_dir: The folder of experiments' data.

    Returns:
        The folder of the written set.

    Raises:
        ValueError: If the name is not allowed, the frame has no UTC `valid_time`, or a
            `valid_time` is on or after the holdout start.
        FileExistsError: If the set already exists.
    """
    if not SET_NAME_PATTERN.fullmatch(name):
        msg = f"set name {name!r} must be lower-case words joined by hyphens or underscores"
        raise ValueError(msg)
    _require_utc_valid_time(frame=frame)
    _require_before_holdout(frame=frame)
    predictions_dir = (
        _experiment_dir(experiment_id=experiment_id, data_dir=data_dir) / "predictions"
    )
    destination = predictions_dir / name
    if destination.exists():
        msg = f"{destination} exists; save the new fit under another name, such as {name}-v2"
        raise FileExistsError(msg)
    commit, dirty = _code_state()
    manifest: PredictionManifest = {
        "kind": kind,
        "row_set_id": row_set_id,
        "commit": commit,
        "dirty": dirty,
    }
    staging = predictions_dir / f".{name}.{uuid4().hex}"
    staging.mkdir(parents=True)
    frame.write_parquet(staging / PREDICTIONS_NAME)
    (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(destination)
    return destination


def read_predictions(
    *,
    experiment_id: int,
    name: str,
    row_set_id: str,
    data_dir: Path = ML_EXPERIMENTS_DATA_DIR,
) -> pl.DataFrame:
    """Read a prediction set that another experiment wrote, checking it covers the same rows.

    Args:
        experiment_id: The id of the experiment that wrote the set.
        name: The set's name.
        row_set_id: The row set the caller scores on.
        data_dir: The folder of experiments' data.

    Returns:
        The set's predictions.

    Raises:
        ValueError: If the set's `row_set_id` differs from `row_set_id`, or a `valid_time` is on or
            after the holdout start. A score on different rows is not a comparison, so the function
            raises and does not warn.
    """
    folder = _experiment_dir(experiment_id=experiment_id, data_dir=data_dir) / "predictions" / name
    stored: PredictionManifest = json.loads((folder / MANIFEST_NAME).read_text())
    if stored["row_set_id"] != row_set_id:
        msg = (
            f"{folder} covers row set {stored['row_set_id']!r}, not {row_set_id!r}; "
            "scores on different rows are not comparable"
        )
        raise ValueError(msg)
    frame = pl.read_parquet(folder / PREDICTIONS_NAME)
    _require_before_holdout(frame=frame)
    return frame


def losses_for_bootstrap(*, arm_losses: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Build the per-row loss frame that `studies.bootstrap` pairs and resamples.

    `studies.bootstrap.paired_differences` joins two arms on `(site, time, seed)`. Forecast rows are
    keyed by `(time_series_id, power_fcst_init_time, valid_time)`, and one valid time sits in 2 or
    3 runs, so joining on the valid time alone would pair rows many-to-many with no error. This
    function builds `time` from the run and the valid time together, takes the month label from
    the run's initialisation time so that a run stays in one month, and refuses arms whose seeds
    differ.

    Args:
        arm_losses: Each arm's name mapped to its per-row losses. Every frame has the columns
            `time_series_id`, `power_fcst_init_time`, `valid_time` and `seed`, and any number of
            loss columns, which are kept.

    Returns:
        One frame holding every arm's rows, with the columns `arm`, `site`, `time`, `seed`,
        `month`, and the loss columns. `time` is the run's initialisation time and the valid time,
        joined into one string that sorts in time order.

    Raises:
        ValueError: If the arms hold different seeds, or an arm holds two rows with the same
            `(time_series_id, power_fcst_init_time, valid_time, seed)`.
    """
    key_columns = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    seeds = {arm: sorted(frame["seed"].unique().to_list()) for arm, frame in arm_losses.items()}
    if len({tuple(arm_seeds) for arm_seeds in seeds.values()}) > 1:
        msg = f"every arm needs the same seeds, found {seeds}"
        raise ValueError(msg)
    for arm, frame in arm_losses.items():
        if frame.select(key_columns).is_duplicated().any():
            msg = f"arm {arm!r} holds two rows with the same run, valid time, series, and seed"
            raise ValueError(msg)
    return pl.concat(
        [
            frame.with_columns(
                arm=pl.lit(arm),
                site=pl.col("time_series_id"),
                time=pl.concat_str(
                    pl.col("power_fcst_init_time").dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    pl.col("valid_time").dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    separator="|",
                ),
                month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            )
            for arm, frame in arm_losses.items()
        ]
    )
