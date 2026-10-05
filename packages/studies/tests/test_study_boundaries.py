"""Enforce the import rules between study scripts, the `studies` package, and production code.

**The rules.** A study script under `studies/` may import from its own folder, from `studies.*`
(`packages/studies`), and from the other reviewed packages in `packages/*`. `src/` and every
package under `packages/` except `packages/studies` must never import `studies` or a study script,
because humans review that code and the study code is fast-moving and agent-written.

**What the tests check, over every script under `studies/` except those in `era_fold_design`:**

- no import of a module that lives in a different `studies/` folder;
- no use of `sys.path` and no `site.addsitedir`;
- no use of `importlib.util.spec_from_file_location`;
- no basename shared by two scripts, because every study folder is on pytest's path;
- no module under `packages/studies/src` that imports the basename of a script;
- no import of `studies` or of a script from `src/` or from a package other than `packages/studies`.

The rules are documented in `CLAUDE.md` (Architecture, "Import rules"), in `studies/README.md`, and
in `.claude/skills/study/SKILL.md` ("Where a study's pieces live").
"""

import ast
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
FROZEN_FOLDER: Final[str] = "era_fold_design"
"""The one folder the rules skip: its scripts are a record of how a report was made."""


def _scripts_by_folder(*, studies_dir: Path) -> dict[str, list[Path]]:
    """Return each study folder's scripts, skipping the frozen folder."""
    return {
        folder.name: sorted(folder.glob("*.py"))
        for folder in sorted(studies_dir.iterdir())
        if folder.is_dir() and folder.name != FROZEN_FOLDER
    }


def _imported_modules(*, source_path: Path) -> set[str]:
    """Return the first dotted component of every absolute import in a Python file."""
    tree = ast.parse(source_path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            modules.add(node.module.split(".")[0])
    return modules


def _path_mutations(*, source_path: Path) -> list[str]:
    """Return each use of `sys.path`, `site.addsitedir`, or `spec_from_file_location`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source_path.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if isinstance(owner, ast.Name) and (owner.id, node.attr) in {
            ("sys", "path"),
            ("site", "addsitedir"),
        }:
            found.append(f"{owner.id}.{node.attr}")
        if node.attr == "spec_from_file_location":
            found.append("spec_from_file_location")
    return found


def crossing_imports(*, studies_dir: Path) -> dict[str, list[str]]:
    """Return, for each script, the modules it imports from a different study folder."""
    scripts = _scripts_by_folder(studies_dir=studies_dir)
    folder_of = {path.stem: folder for folder, paths in scripts.items() for path in paths}
    crossings: dict[str, list[str]] = {}
    for folder, paths in scripts.items():
        for path in paths:
            foreign = sorted(
                module
                for module in _imported_modules(source_path=path)
                if module in folder_of and folder_of[module] != folder
            )
            if foreign:
                crossings[f"{folder}/{path.name}"] = foreign
    return crossings


def path_mutations(*, studies_dir: Path) -> dict[str, list[str]]:
    """Return, for each script, its uses of `sys.path`, `site.addsitedir` and path loaders."""
    return {
        f"{folder}/{path.name}": found
        for folder, paths in _scripts_by_folder(studies_dir=studies_dir).items()
        for path in paths
        if (found := _path_mutations(source_path=path))
    }


def shared_basenames(*, studies_dir: Path) -> dict[str, list[str]]:
    """Return each script basename that two scripts share, with the folders holding it."""
    folders_by_stem: dict[str, list[str]] = {}
    for folder, paths in _scripts_by_folder(studies_dir=studies_dir).items():
        for path in paths:
            folders_by_stem.setdefault(path.stem, []).append(folder)
    return {stem: folders for stem, folders in folders_by_stem.items() if len(folders) > 1}


def package_imports_of_scripts(*, studies_dir: Path, package_src_dir: Path) -> dict[str, list[str]]:
    """Return, for each package module, the script basenames it imports."""
    basenames = {
        path.stem
        for paths in _scripts_by_folder(studies_dir=studies_dir).values()
        for path in paths
    }
    return {
        str(path.relative_to(package_src_dir)): sorted(imported)
        for path in sorted(package_src_dir.rglob("*.py"))
        if (imported := _imported_modules(source_path=path) & basenames)
    }


def production_imports_of_studies(*, repo_root: Path) -> dict[str, list[str]]:
    """Return, for each production file, the `studies` or script modules it imports."""
    forbidden = {"studies"} | {
        path.stem
        for paths in _scripts_by_folder(studies_dir=repo_root / "studies").values()
        for path in paths
    }
    packages = repo_root / "packages"
    roots = [repo_root / "src"] + [
        package for package in sorted(packages.iterdir()) if package.name != "studies"
    ]
    files = [path for root in roots if root.is_dir() for path in root.rglob("*.py")]
    assert files, "found no production files to scan, so the scan would pass vacuously"
    return {
        str(path.relative_to(repo_root)): sorted(imported)
        for path in files
        if (imported := _imported_modules(source_path=path) & forbidden)
    }


def _write(*, root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def test_no_study_script_imports_a_script_of_another_folder():
    assert crossing_imports(studies_dir=REPO_ROOT / "studies") == {}


def test_no_study_script_touches_sys_path_or_loads_a_script_by_path():
    assert path_mutations(studies_dir=REPO_ROOT / "studies") == {}


def test_no_two_study_scripts_share_a_basename():
    assert shared_basenames(studies_dir=REPO_ROOT / "studies") == {}


def test_no_package_module_imports_a_study_script_by_its_bare_name():
    result = package_imports_of_scripts(
        studies_dir=REPO_ROOT / "studies",
        package_src_dir=REPO_ROOT / "packages" / "studies" / "src",
    )

    assert result == {}


def test_production_does_not_import_studies():
    assert production_imports_of_studies(repo_root=REPO_ROOT) == {}


def test_an_import_of_a_script_in_another_folder_is_a_crossing_and_one_in_the_same_folder_is_not(
    tmp_path: Path,
):
    _write(
        root=tmp_path,
        files={
            "alpha/one.py": "import two\nimport sibling\nimport numpy\n",
            "alpha/sibling.py": "",
            "beta/two.py": "from sibling import x\n",
            "beta/three.py": "import numpy, _private\nimport two.sub\n",
            "alpha/_private.py": "",
            "era_fold_design/frozen.py": "import one\n",
        },
    )

    assert crossing_imports(studies_dir=tmp_path) == {
        "alpha/one.py": ["two"],
        "beta/three.py": ["_private"],
        "beta/two.py": ["sibling"],
    }


def test_every_form_of_path_mutation_and_path_loading_is_found(tmp_path: Path):
    _write(
        root=tmp_path,
        files={
            "alpha/a.py": "import sys\nsys.path.insert(0, '.')\n",
            "alpha/b.py": "import sys\nsys.path[:0] = ['.']\n",
            "alpha/c.py": "import site\nsite.addsitedir('.')\n",
            "alpha/d.py": "import importlib.util as u\nu.spec_from_file_location('a', 'b')\n",
            "alpha/f.py": "import importlib\nimportlib.util.spec_from_file_location('a', 'b')\n",
            "alpha/e.py": "import sys\nprint(len(sys.argv))\n",
        },
    )

    assert path_mutations(studies_dir=tmp_path) == {
        "alpha/a.py": ["sys.path"],
        "alpha/b.py": ["sys.path"],
        "alpha/c.py": ["site.addsitedir"],
        "alpha/d.py": ["spec_from_file_location"],
        "alpha/f.py": ["spec_from_file_location"],
    }


def test_a_basename_in_two_folders_is_reported_with_both_folders(tmp_path: Path):
    _write(
        root=tmp_path,
        files={"alpha/helper.py": "", "beta/helper.py": "", "beta/other.py": ""},
    )

    assert shared_basenames(studies_dir=tmp_path) == {"helper": ["alpha", "beta"]}


def test_a_package_module_importing_a_scripts_bare_name_is_reported(tmp_path: Path):
    _write(root=tmp_path / "studies", files={"alpha/one.py": ""})
    _write(
        root=tmp_path / "src",
        files={
            "studies/bad.py": "from one import y\n",
            "studies/good.py": "import polars\n",
            "studies/nested/deeper.py": "import numpy, one\n",
        },
    )

    assert package_imports_of_scripts(
        studies_dir=tmp_path / "studies", package_src_dir=tmp_path / "src"
    ) == {"studies/bad.py": ["one"], "studies/nested/deeper.py": ["one"]}


def test_production_code_importing_studies_or_a_script_is_reported(tmp_path: Path):
    _write(root=tmp_path / "studies", files={"alpha/one.py": ""})
    _write(root=tmp_path / "packages" / "studies", files={"src/studies/x.py": "import one\n"})
    _write(
        root=tmp_path / "packages" / "other",
        files={
            "src/other/a.py": "import studies\n",
            "src/other/b.py": "import one\n",
            "src/other/c.py": "import studies.sources\n",
        },
    )
    _write(root=tmp_path / "src", files={"app/c.py": "from studies.sources import X\nimport os\n"})

    assert production_imports_of_studies(repo_root=tmp_path) == {
        "packages/other/src/other/a.py": ["studies"],
        "packages/other/src/other/b.py": ["one"],
        "packages/other/src/other/c.py": ["studies"],
        "src/app/c.py": ["studies"],
    }
