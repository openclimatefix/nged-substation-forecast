"""Tests of `scripts/deploy/_image_tag.sh`, which derives the production image tag from a bundle."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy" / "_image_tag.sh"

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")


def _tag(bundle_dir: Path) -> subprocess.CompletedProcess[str]:
    # The deploy scripts call `image_tag` inside a command substitution under `set -euo pipefail`.
    command = (
        f'set -euo pipefail; source "{_SCRIPT}"; tag="$(image_tag "{bundle_dir}")"; echo "$tag"'
    )
    return subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False, cwd=_SCRIPT.parent
    )


_MANIFEST = {
    "model_release": 3,
    "research_git": {"commit": "246984d2e8da19999f38fded"},
    "refit_month": "2026-10",
}


def test_the_tag_names_the_commit_the_release_the_research_commit_and_the_refit_month(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(_MANIFEST))
    result = _tag(tmp_path)
    assert result.returncode == 0, result.stderr
    commit, release, research, refit = result.stdout.strip().split("-")
    assert len(commit) == 8
    assert (release, research, refit) == ("r3", "246984d2", "202610")


def test_two_refits_from_one_research_commit_get_different_tags(tmp_path: Path) -> None:
    tags = []
    for month in ("2026-10", "2026-11"):
        (tmp_path / "manifest.json").write_text(json.dumps(_MANIFEST | {"refit_month": month}))
        tags.append(_tag(tmp_path).stdout)
    assert tags[0] != tags[1]


@pytest.mark.parametrize(
    "manifest",
    [
        {key: value for key, value in _MANIFEST.items() if key != "model_release"},
        {key: value for key, value in _MANIFEST.items() if key != "refit_month"},
        _MANIFEST | {"refit_month": "October 2026"},
    ],
    ids=["no model_release", "no refit_month", "malformed refit_month"],
)
def test_a_manifest_without_a_field_of_the_tag_exits_2_rather_than_printing_a_tag(
    tmp_path: Path, manifest: dict
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = _tag(tmp_path)
    assert result.returncode == 2
    assert result.stdout == ""
