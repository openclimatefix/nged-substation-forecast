"""Import the `prose-review` skill's scripts, which pytest cannot collect from their own directory.

The scripts live under `.claude/skills/prose-review/scripts/`. That directory is hidden, so pytest
never collects it and a plain `import` cannot reach it either. Every test module for those scripts
loads them through this helper, and their tests live in `tests/`, where CI runs them.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPTS: Final[Path] = REPO_ROOT / ".claude" / "skills" / "prose-review" / "scripts"
"""Where the sweep's scripts live, imported by path because `.claude/` is not a package."""


def load(name: str) -> ModuleType:
    """Import one script from the skill directory by its module name.

    Args:
        name: The script's stem, such as `apply_findings_py`.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_repo(root: Path) -> None:
    """Initialise a git repository at `root` with a committer identity and no commits yet.

    Args:
        root: The directory to turn into a repository.
    """
    run(root, "init", "-q", "-b", "main")
    run(root, "config", "user.email", "test@example.invalid")
    run(root, "config", "user.name", "Test")


def commit(root: Path, message: str = "commit") -> str:
    """Stage everything under `root` and commit it, returning the new commit's hash.

    Args:
        root: The repository's working tree.
        message: The commit message.

    Returns:
        The full hash of the commit just made.
    """
    run(root, "add", "-A")
    run(root, "commit", "-q", "-m", message)
    return run(root, "rev-parse", "HEAD").stdout.strip()


def run(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run one git command in `cwd`, raising if it fails.

    Args:
        cwd: The directory to run in.
        arguments: The git sub-command and its arguments.

    Returns:
        The completed process, with its output captured as text.
    """
    return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True)
