"""Reproducibility provenance: the git SHA and Delta-table versions behind an MLflow run.

Answers "exactly which code and which data produced this?" for any MLflow run. The git SHA pins
the code — stamped **explicitly** because MLflow's ``mlflow.source.git.commit`` auto-detection
needs gitpython installed *and* the working directory inside the repo, neither of which holds in a
production container. Each Delta table's ``version()`` pins the data: Delta Lake time travel makes
data versioning one integer per table, so a run can later be replayed with
``pl.scan_delta(path, version=N)`` after ``git checkout {sha}``.

Every function here is deliberately **non-raising**: provenance is metadata, and a missing ``.git``
directory (containers) or an absent Delta table must never fail the surrounding training or
forecasting run. Such failures degrade to the sentinels ``"unknown"`` / ``"absent"``.

``provenance_tags`` **stage-prefixes** its keys (``register_``, ``train_``, ``predict_``,
``metrics_``) because a single MLflow fold run is written by three separate assets —
``trained_cv_model``, ``cv_power_forecasts`` and ``metrics`` — each potentially on a different code
revision and data state. Un-prefixed keys would clobber one another; the prefix preserves all
three provenance snapshots side by side.
"""

import logging
import subprocess
from pathlib import Path
from typing import Final

from deltalake import DeltaTable

logger = logging.getLogger(__name__)

UNKNOWN: Final[str] = "unknown"
"""Sentinel git SHA / dirty flag returned when no git repository is reachable (e.g. a container)."""

ABSENT: Final[str] = "absent"
"""Sentinel Delta version returned when a table does not exist or cannot be read."""

_GIT_CWD: Final[Path] = Path(__file__).resolve().parent
"""Directory the git commands run from — inside the repo for an editable/workspace install, so the
SHA is captured independently of the process's working directory; outside any repo for a wheel
install in a container, where the commands fail and ``get_git_info`` returns ``UNKNOWN``.

Caveat: ``git rev-parse`` walks *upward* for a ``.git`` directory, so if an install location that
is not this project nonetheless sits inside some *other* git repo (e.g. a Docker build context that
copied the project's ``.git``, or a rogue ``.git`` above ``site-packages``), the returned SHA is
that repo's HEAD, not this project's. In this workspace-install project that does not arise, but a
confidently-wrong SHA is worse than ``UNKNOWN``; treat the SHA as trustworthy only for
editable/workspace installs."""

_GIT_TIMEOUT_S: Final[float] = 5.0
"""Hard cap on each git subprocess so a stalled ``.git`` (NFS, a stale lock) can never hang the
surrounding Dagster asset — provenance is best-effort metadata, not a blocking dependency."""


def get_git_info(cwd: Path | None = None) -> dict[str, str]:
    """Return ``{"git_sha", "git_dirty"}`` for the current checkout, never raising.

    ``git_dirty`` is ``"true"`` when the working tree has uncommitted changes, ``"false"`` when
    clean. When the SHA cannot be read (no ``.git``, no ``git`` binary, a timeout) both values are
    ``UNKNOWN``; when the SHA is read but the dirty check fails, the SHA is kept and only
    ``git_dirty`` degrades to ``UNKNOWN`` — a good SHA is never discarded.

    Args:
        cwd: Directory the ``git`` commands run from. Defaults to this module's directory
            (``_GIT_CWD``) — inside the repo for an editable/workspace install, so the SHA is
            captured regardless of the process's working directory. Overridable for testing.
    """
    run_from = cwd if cwd is not None else _GIT_CWD

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=run_from,
            capture_output=True,
            text=True,
            errors="replace",  # non-UTF-8 bytes (e.g. an odd filename) must not raise on decode.
            check=True,
            timeout=_GIT_TIMEOUT_S,
        ).stdout

    try:
        sha = _git("rev-parse", "HEAD").strip()
    except Exception:  # noqa: BLE001 — provenance must never fail the surrounding run.
        return {"git_sha": UNKNOWN, "git_dirty": UNKNOWN}
    try:
        porcelain = _git("status", "--porcelain")
    except Exception:  # noqa: BLE001 — keep the SHA we already have; only dirtiness is unknown.
        return {"git_sha": sha, "git_dirty": UNKNOWN}
    return {"git_sha": sha, "git_dirty": "true" if porcelain.strip() else "false"}


def get_delta_versions(
    paths: dict[str, str], storage_options: dict[str, str] | None = None
) -> dict[str, str]:
    """Return ``{f"delta_version__{name}": str(version)}`` for each named Delta table, never raising.

    Args:
        paths: ``{logical_name: table_uri}``. The URI may be local or a remote object-store URI.
        storage_options: delta-rs storage options for a remote URI; ``None``/empty for local.

    Returns:
        One entry per input path. A table that does not exist (or cannot be read) maps to
        ``ABSENT`` rather than raising, so provenance capture never fails the calling run.
    """
    versions: dict[str, str] = {}
    for name, path in paths.items():
        key = f"delta_version__{name}"
        try:
            options = storage_options or {}
            if not DeltaTable.is_deltatable(path, storage_options=options):
                versions[key] = ABSENT
                continue
            versions[key] = str(DeltaTable(path, storage_options=options).version())
        except Exception:  # noqa: BLE001 — provenance must never fail the surrounding run.
            logger.warning("Could not read Delta version for %r at %s", name, path, exc_info=True)
            versions[key] = ABSENT
    return versions


def provenance_tags(
    stage: str,
    delta_paths: dict[str, str] | None = None,
    storage_options: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build stage-prefixed MLflow tags stamping the code (git) and data (Delta versions) provenance.

    Args:
        stage: Prefix identifying the writing asset (``"register"``, ``"train"``, ``"predict"``,
            ``"metrics"``). Keeps the tags of assets that share one MLflow run from clobbering.
        delta_paths: ``{logical_name: table_uri}`` for the Delta tables this stage reads; omit for
            a stage that reads no data (registration).
        storage_options: delta-rs storage options for remote table URIs; ``None``/empty for local.

    Returns:
        e.g. for ``stage="train"``: ``{"train_git_sha", "train_git_dirty",
        "train_delta_version__power_time_series", ...}``.
    """
    git = get_git_info()
    tags = {f"{stage}_git_sha": git["git_sha"], f"{stage}_git_dirty": git["git_dirty"]}
    if delta_paths:
        for key, version in get_delta_versions(delta_paths, storage_options).items():
            tags[f"{stage}_{key}"] = version
    return tags
