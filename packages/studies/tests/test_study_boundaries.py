"""Guard the import rule between production code and the study code.

The rule: `src/` and every package under `packages/` except `packages/studies` must never import
`studies` or a script under `studies/`, because humans review that code and the study code is
fast-moving and agent-written.
"""

import ast
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
STUDIES_SCRIPTS_DIR: Final[Path] = REPO_ROOT / "studies"


def _imported_top_level_modules(source_path: Path) -> set[str]:
    """Return the first dotted component of every absolute import in a Python file."""
    tree = ast.parse(source_path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            modules.add(node.module.split(".")[0])
    return modules


def _study_script_basenames() -> set[str]:
    return {
        path.stem
        for path in STUDIES_SCRIPTS_DIR.rglob("*.py")
        if "era_fold_design" not in path.parts
    }


def _production_python_files() -> list[Path]:
    """Return every Python file in `src/` and in each package other than `packages/studies`."""
    roots = [REPO_ROOT / "src"] + [
        package
        for package in sorted((REPO_ROOT / "packages").iterdir())
        if package.name != "studies"
    ]
    return [path for root in roots if root.is_dir() for path in root.rglob("*.py")]


def test_production_does_not_import_studies():
    forbidden = {"studies"} | _study_script_basenames()
    files = _production_python_files()
    assert files, "found no production files to scan, so the scan would pass vacuously"

    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(_imported_top_level_modules(path) & forbidden)
        for path in files
        if _imported_top_level_modules(path) & forbidden
    }

    assert offenders == {}
