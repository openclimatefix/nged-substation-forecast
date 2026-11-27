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


def test_the_tag_names_the_commit_the_release_and_the_research_commit(tmp_path: Path) -> None:
    manifest = {"model_release": 3, "research_git": {"commit": "246984d2e8da19999f38fded"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = _tag(tmp_path)
    assert result.returncode == 0, result.stderr
    commit, release, research = result.stdout.strip().split("-")
    assert len(commit) == 8
    assert (release, research) == ("r3", "246984d2")


def test_a_manifest_without_a_model_release_fails_rather_than_printing_a_tag(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"research_git": {"commit": "a" * 40}}))
    result = _tag(tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
