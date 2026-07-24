"""Regression tests for PROJECT_ROOT resolution under different install layouts.

PROJECT_ROOT used to be ``Path(__file__).parents[4]`` — a hard-coded directory depth that only
held for an editable install, and silently resolved to the venv root under a non-editable
(``uv sync --no-editable``) install (issue #287). These tests pin the marker-based replacement:
walk up to the nearest ancestor holding ``uv.lock``, falling back to the current working
directory when no ancestor qualifies.
"""

from pathlib import Path

import pytest
from contracts.settings import PROJECT_ROOT, Settings, _find_project_root


def test_project_root_is_the_workspace_root():
    """In this dev checkout, PROJECT_ROOT holds the workspace marker and the resource files.

    Also guards the repo layout itself: if ``conf/`` or ``metadata/`` move, the
    PROJECT_ROOT-relative Settings defaults break, and this is the test that should fail.
    """
    assert (PROJECT_ROOT / "uv.lock").is_file()
    assert (PROJECT_ROOT / "conf" / "cv" / "default.yaml").is_file()
    assert (PROJECT_ROOT / "metadata" / "nwp_metadata.csv").is_file()


def test_settings_defaults_anchor_at_project_root():
    """The repo-resource Settings defaults are PROJECT_ROOT-relative (not venv-relative)."""
    assert Settings.model_fields["cv_config_path"].default == (
        PROJECT_ROOT / "conf" / "cv" / "default.yaml"
    )
    assert Settings.model_fields["nwp_metadata_csv_path"].default == (
        PROJECT_ROOT / "metadata" / "nwp_metadata.csv"
    )


def test_find_project_root_from_in_repo_non_editable_venv(tmp_path: Path):
    """A non-editable install whose venv sits inside the repo resolves to the repo root.

    This is the exact layout that broke under ``parents[4]``: the installed module lives at
    ``<repo>/.venv/lib/python3.14/site-packages/contracts/settings.py``, five levels below the
    repo root rather than four.
    """
    repo = tmp_path / "repo"
    installed_pkg = repo / ".venv" / "lib" / "python3.14" / "site-packages" / "contracts"
    installed_pkg.mkdir(parents=True)
    (repo / "uv.lock").touch()
    settings_file = installed_pkg / "settings.py"
    settings_file.touch()

    assert _find_project_root(settings_file) == repo


def test_find_project_root_outside_any_workspace_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With no ``uv.lock`` in any ancestor, resolution falls back to the working directory.

    This is the wheel-installed-outside-a-checkout case: such a deployment must either run
    with its cwd at a directory laid out like the repo root, or set every path setting
    explicitly via env vars (as documented on ``_find_project_root``).
    """
    installed_pkg = tmp_path / "venv" / "lib" / "python3.14" / "site-packages" / "contracts"
    installed_pkg.mkdir(parents=True)
    settings_file = installed_pkg / "settings.py"
    settings_file.touch()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    assert _find_project_root(settings_file) == workdir
