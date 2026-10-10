"""Run a study's XGBoost fits in checkpointed groups, so a reboot costs one group, not the run.

Written for the IFS solar-variables study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>, and the same code as the
ERA5 variable ladder's `era5_ladder_fit.py` uses. A fit script gives `fit_in_groups` its jobs
(`studies.arm_runner.Job`), and the function writes each group's per-row losses to a checkpoint file
the moment the group finishes. A rerun reads the groups that exist and fits only the others.

**A group's file name holds a hash of its job names, and the file's own (arm, setting) pairs are
checked against the group**, so a rerun with different arms refits rather than reuses. A checkpoint
is matched by arm and setting names only, so the caller moves or deletes the checkpoint directory
whenever the dataset or an arm's columns change.
"""

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import polars as pl

from studies.arm_runner import Job, run_all
from studies.cross_validation import DeviceType

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

MAX_LOAD_PER_CORE: Final[float] = 0.5
"""A run refuses to start if the one-minute load average per core is above this."""

JOBS_PER_CHECKPOINT: Final[int] = 4
"""How many (arm, setting) jobs one checkpoint file holds."""


def choose_device(*, requested: str) -> DeviceType:
    """Return the XGBoost device: the GPU if `nvidia-smi` shows one, unless the caller chooses.

    Args:
        requested: `auto`, `cpu`, or `cuda`.

    Returns:
        `"cuda"` or `"cpu"`.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False, text=True)
    return "cuda" if probe.returncode == 0 and "GPU" in probe.stdout else "cpu"


def refuse_if_machine_is_busy(*, ignore: bool) -> None:
    """Raise if another job is already using the CPU, because it would slow every fit.

    Args:
        ignore: Skip the check.

    Raises:
        RuntimeError: If the load average per core is above `MAX_LOAD_PER_CORE`.
    """
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    _LOG.info("one-minute load average per core: %.2f", load)
    if load > MAX_LOAD_PER_CORE and not ignore:
        msg = f"load per core is {load:.2f}, above {MAX_LOAD_PER_CORE}; stop the other job first"
        raise RuntimeError(msg)


def job_labels(*, group: list[Job]) -> list[str]:
    """Return a group's sorted `arm@setting` labels, which name its checkpoint file."""
    return sorted(f"{name}@{setting}" for name, setting, *_ in group)


def checkpoint_name(*, labels: list[str]) -> str:
    """Return the checkpoint file name for a group's labels."""
    digest = hashlib.sha1("|".join(labels).encode(), usedforsecurity=False).hexdigest()[:10]
    return f"group_{digest}.parquet"


def fit_in_groups(
    *,
    dataset: pl.DataFrame,
    jobs: list[Job],
    checkpoint_dir: Path,
    max_workers: int,
    device: DeviceType,
    jobs_per_checkpoint: int = JOBS_PER_CHECKPOINT,
) -> pl.DataFrame:
    """Fit every job in checkpointed groups and return the stacked per-row losses.

    Args:
        dataset: The rows with folds, the target, and every column any job uses.
        jobs: The fits to run.
        checkpoint_dir: Where each group writes its losses.
        max_workers: How many (arm, site) fits run at once.
        device: XGBoost's device.
        jobs_per_checkpoint: How many jobs one checkpoint file holds.

    Returns:
        The losses of every job, labelled with the arm, the setting, and the target.

    Raises:
        RuntimeError: If a checkpoint file holds other arms or settings than its name says.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pl.DataFrame] = []
    for start in range(0, len(jobs), jobs_per_checkpoint):
        group = jobs[start : start + jobs_per_checkpoint]
        labels = job_labels(group=group)
        path = checkpoint_dir / checkpoint_name(labels=labels)
        if path.exists():
            part = pl.read_parquet(path)
            held = sorted(
                f"{name}@{setting}"
                for name, setting in part.select("arm", "setting").unique().rows()
            )
            if held != labels:
                msg = f"{path.name} holds {held}, expected {labels}"
                raise RuntimeError(msg)
            _LOG.info(
                "checkpoint group %d (of %d jobs): reading %s",
                start // jobs_per_checkpoint + 1,
                len(jobs),
                path.name,
            )
        else:
            part = run_all(dataset=dataset, jobs=group, max_workers=max_workers, device=device)
            temporary = path.with_name(path.name + ".tmp")
            part.write_parquet(temporary)
            temporary.rename(path)
        parts.append(part)
    return pl.concat(parts)
