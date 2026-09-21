"""Reproducibility provenance: the git SHA and Delta-table versions behind an MLflow run.

Answers "exactly which code and which data produced this?" for any MLflow run. The git SHA pins
the code. The SHA is stamped **explicitly** because MLflow's ``mlflow.source.git.commit``
auto-detection needs gitpython installed *and* the working directory inside the repo. Neither
condition holds in a production container. Each Delta table's ``version()`` pins the data: Delta
Lake time travel makes data versioning one integer per table. A run can therefore later be
replayed with ``pl.scan_delta(path, version=N)`` after ``git checkout {sha}``.

Every function here is deliberately **non-raising**: the git SHA, the dirty flag, and each Delta
table's version are a record *about* a run rather than an input to that run. A missing ``.git``
directory (containers) or an absent Delta table must never fail the surrounding training or
forecasting run. Each absence degrades to the sentinels ``"unknown"`` / ``"absent"`` instead.

``provenance_tags`` **stage-prefixes** its keys (``register_``, ``train_``, ``predict_``,
``metrics_``) because four separate writers stamp provenance onto the same MLflow runs. Three are
Dagster assets writing one fold run: ``trained_cv_model``, ``cv_power_forecasts``, and
``metrics``. Each of the three can be on a different code revision and at a different set of
Delta table versions. The fourth is the ``register_experiment`` op inside
``register_experiment_job``, which stamps the experiment's parent run; the ``metrics`` asset
stamps that parent run too. Un-prefixed keys would clobber one another, and the prefix preserves
every writer's provenance snapshot side by side.
"""

import logging
import subprocess
from pathlib import Path
from typing import Final, Literal

from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import DeltaTable

logger = logging.getLogger(__name__)

MlflowTags = dict[str, str]
"""A ``{tag_key: tag_value}`` mapping ready to hand to ``mlflow.set_tags``."""

StageType = Literal["register", "train", "predict", "metrics"]
"""The four stages that stamp provenance; each value becomes a tag-key prefix (see
``provenance_tags``). ``"train"``, ``"predict"``, and ``"metrics"`` are Dagster assets, and
``"register"`` is the ``register_experiment`` op. Add a new ``StageType`` value when a new asset or
op starts stamping."""

TableNameType = Literal[
    "power_time_series",
    "nwp_data",
    "eligible_time_series",
    "power_forecasts",
    "effective_capacity",
]
"""Logical names of the Delta tables whose versions get stamped — the keys of a ``delta_paths``
mapping. Add a new ``TableNameType`` value when a stage starts reading (and stamping) another
table."""

UNKNOWN: Final[str] = "unknown"
"""Sentinel git SHA / dirty flag returned when no git repository is reachable (e.g. a container)."""

ABSENT: Final[str] = "absent"
"""Sentinel Delta version returned when a table does not exist or cannot be read."""

_GIT_CWD: Final[Path] = Path(__file__).resolve().parent
"""Directory the git commands run from — inside the repo for an editable/workspace install, so the
SHA is captured independently of the process's working directory; outside any repo for a wheel
install in a container, where the commands fail and ``get_git_info`` returns ``UNKNOWN``.

Caveat: ``git rev-parse`` walks *upward* for a ``.git`` directory. If an install location that is
not this project nonetheless sits inside some *other* git repo (e.g. a Docker build context that
copied the project's ``.git``, or a rogue ``.git`` above ``site-packages``), the returned SHA is
that repo's HEAD, not this project's. In this workspace-install project that does not arise. A
confidently-wrong SHA is nonetheless worse than ``UNKNOWN``, so treat the SHA as trustworthy only
for editable and workspace installs."""

_GIT_TIMEOUT_S: Final[float] = 5.0
"""Hard cap on each git subprocess so a stalled ``.git`` (a network file system, a stale lock) can
never hang the surrounding Dagster asset — the git SHA and the Delta table versions are best-effort
records, not a blocking dependency."""


def get_git_info(cwd: Path | None = None) -> MlflowTags:
    """Return ``{"git_sha", "git_dirty"}`` for the current checkout, never raising.

    ``git_dirty`` is ``"true"`` when the working tree has uncommitted changes, ``"false"`` when
    clean. When the SHA cannot be read (no ``.git``, no ``git`` binary, a timeout) both values are
    ``UNKNOWN``. When the SHA is read but the dirty check fails, the SHA is kept and only
    ``git_dirty`` degrades to ``UNKNOWN``, because a good SHA is never discarded.

    Args:
        cwd: Directory the ``git`` commands run from. Defaults to this module's directory
            (``_GIT_CWD``). That directory is inside the repo for an editable or workspace install,
            so the SHA is captured regardless of the process's working directory. Overridable for
            testing.

    Returns:
        ``{"git_sha": sha, "git_dirty": dirty}``, both plain strings. ``sha`` is the 40-character
        commit hash, and ``dirty`` is either ``"true"`` or ``"false"``. Either value may instead be
        ``UNKNOWN``, per the degradation described above.
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
    except Exception:  # noqa: BLE001 — provenance must never fail the surrounding Dagster asset.
        return {"git_sha": UNKNOWN, "git_dirty": UNKNOWN}
    try:
        porcelain = _git("status", "--porcelain")
    except Exception:  # noqa: BLE001 — keep the SHA we already have; only dirtiness is unknown.
        return {"git_sha": sha, "git_dirty": UNKNOWN}
    return {"git_sha": sha, "git_dirty": "true" if porcelain.strip() else "false"}


def get_delta_versions(
    paths: dict[TableNameType, str], storage_options: ObjectStoreOptions | None = None
) -> MlflowTags:
    """Return ``{f"delta_version__{name}": str(version)}`` for each named Delta table.

    Never raises.

    Args:
        paths: ``{logical_name: table_uri}``. The URI may be local or a remote object-store URI.
        storage_options: object-store options for a remote URI; ``None``/empty for local. Delta
            Lake access runs through the delta-rs library, whose signature takes a plain ``dict``,
            so this function widens the narrower ``ObjectStoreOptions`` mapping to that ``dict``
            at the call boundary.

    Returns:
        One entry per input path. A table that does not exist (or cannot be read) maps to
        ``ABSENT`` rather than raising, so provenance capture never fails the calling run.
    """
    options = typeddict_to_dict(storage_options) or {}
    versions: MlflowTags = {}
    for name, path in paths.items():
        key = f"delta_version__{name}"
        try:
            if not DeltaTable.is_deltatable(path, storage_options=options):
                versions[key] = ABSENT
                continue
            versions[key] = str(DeltaTable(path, storage_options=options).version())
        except Exception:  # Provenance must never fail the surrounding Dagster asset.
            logger.warning("Could not read Delta version for %r at %s", name, path, exc_info=True)
            versions[key] = ABSENT
    return versions


def provenance_tags(
    stage: StageType,
    delta_paths: dict[TableNameType, str] | None = None,
    storage_options: ObjectStoreOptions | None = None,
) -> MlflowTags:
    """Build stage-prefixed MLflow tags stamping code (git) and data (Delta version) provenance.

    Args:
        stage: Prefix identifying the writing asset or op. Keeps the tags of the assets and ops that
            share one MLflow run from clobbering.
        delta_paths: ``{logical_name: table_uri}`` for the Delta tables this stage reads; omit for
            a stage that reads no data (registration).
        storage_options: object-store options for remote table URIs; ``None``/empty for local.

    Returns:
        e.g. for ``stage="train"``: ``{"train_git_sha", "train_git_dirty",
        "train_delta_version__power_time_series", ...}``.
    """
    git = get_git_info()
    tags: MlflowTags = {f"{stage}_git_sha": git["git_sha"], f"{stage}_git_dirty": git["git_dirty"]}
    if delta_paths:
        for key, version in get_delta_versions(delta_paths, storage_options).items():
            tags[f"{stage}_{key}"] = version
    return tags
