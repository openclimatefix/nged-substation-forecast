"""Tests for `scripts/lint/lint_docstring_markdown.py`.

The script hands each docstring to `pymarkdown` on its own and then shifts the reported line
numbers back onto the Python file. Two things have to hold for that to be useful, and neither is
visible in a passing lint run: the dedent step must preserve the docstring's line count, and the
configuration must reach the scanner. `test_violation_reports_the_source_line` and
`test_line_count_is_preserved_through_dedent` cover the first;
`test_document_only_rules_are_disabled` and the two `md013` tests cover the second, because a
config that failed to load would light up every docstring in the repo with `MD041` instead.

The scanner is configured from `.pymarkdown-docstrings.json` and `pyproject.toml`, both named
relative to the working directory, so every test runs from `REPO_ROOT`.

`test_pool_and_serial_paths_agree` runs the script as a subprocess, the way the pre-commit hook
does, because the process pool only engages from `POOL_FILE_COUNT` files upwards.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "lint" / "lint_docstring_markdown.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `lint_docstring_markdown.py` from its path in `scripts/lint/`."""
    spec = importlib.util.spec_from_file_location("lint_docstring_markdown", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lint_docstring_markdown = _load_script()

NESTED_LIST_VIOLATION: Final[str] = "PML101"
"""A sub-bullet indented 2 spaces instead of 4, the rule `pyproject.toml` enables in md007's place.

Used as the test corpus's violation throughout, because it fires on a two-line list and so keeps
every fixture short.
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from `REPO_ROOT`, where the two pymarkdown config files sit."""
    monkeypatch.chdir(REPO_ROOT)


def _write_module(path: Path, body: str) -> Path:
    """Write `body` to `path` as a Python file and return `path`."""
    path.write_text(body, encoding="utf-8")
    return path


def test_clean_docstrings_report_nothing(tmp_path: Path) -> None:
    """A file whose docstrings are valid markdown produces no violations."""
    path = _write_module(
        tmp_path / "clean.py",
        '"""A module summary.\n\nA second paragraph of ordinary prose.\n"""\n',
    )
    assert lint_docstring_markdown._lint_file(path) == []


def test_violation_reports_the_source_line(tmp_path: Path) -> None:
    """A violation is reported against the Python file's own line, not the docstring's.

    The offending sub-bullet is on line 7 of the file: three lines of module docstring, a blank
    line, the `def`, the docstring's opening line, then the list.
    """
    path = _write_module(
        tmp_path / "nested.py",
        '"""Module summary."""\n'
        "\n"
        "\n"
        "def f() -> None:\n"
        '    """Summary.\n'
        "\n"
        "    - parent\n"
        "      - child\n"
        '    """\n',
    )
    violations = lint_docstring_markdown._lint_file(path)
    assert len(violations) == 1
    assert violations[0].startswith(f"{path}:8:")
    assert NESTED_LIST_VIOLATION in violations[0]


def test_line_count_is_preserved_through_dedent() -> None:
    """Dedenting must not drop blank lines, or every reported line number shifts.

    `inspect.cleandoc` would strip the leading and trailing blank lines here, which is exactly
    why the script does not use it.
    """
    raw = "\n\n    Indented body.\n\n    More body.\n\n    "
    dedented = lint_docstring_markdown._dedent_docstring(raw)
    assert dedented.count("\n") == raw.count("\n")
    assert "Indented body." in dedented
    assert "\n    Indented body." not in dedented


def test_document_only_rules_are_disabled(tmp_path: Path) -> None:
    """A docstring is a fragment, so it need not open with a heading nor end with a newline.

    `MD041` and `MD047` are the two rules `.pymarkdown-docstrings.json` turns off. If that file
    stopped being read, this docstring would report both.
    """
    path = _write_module(
        tmp_path / "fragment.py", '"""Just prose, no heading, no final newline."""\n'
    )
    assert lint_docstring_markdown._lint_file(path) == []


def _wrappable_line(length: int) -> str:
    """Return a line of `length` characters that md013 could have wrapped.

    md013 runs non-strict here, so it ignores a long line it could not have broken — a bare
    `"x" * 110` passes whatever the configured width is, and would make these two tests agree
    with each other for the wrong reason. Five characters per "word " makes the arithmetic exact.
    """
    assert length % 5 == 4, "a whole number of 5-character words, less the trailing space"
    return " ".join(["word"] * ((length + 1) // 5))


def test_md013_allows_a_line_under_the_repo_width(tmp_path: Path) -> None:
    """`pyproject.toml` sets the line length to 100, so an 89-character line passes.

    pymarkdown's own default is 80. A failure here means `pyproject.toml`'s `[tool.pymarkdown]`
    is not reaching the scanner.
    """
    path = _write_module(tmp_path / "short.py", f'"""Summary.\n\n{_wrappable_line(89)}\n"""\n')
    assert lint_docstring_markdown._lint_file(path) == []


def test_md013_flags_a_line_over_the_repo_width(tmp_path: Path) -> None:
    """A 124-character line is over the configured 100, so it is reported."""
    path = _write_module(tmp_path / "long.py", f'"""Summary.\n\n{_wrappable_line(124)}\n"""\n')
    violations = lint_docstring_markdown._lint_file(path)
    assert len(violations) == 1
    assert "MD013" in violations[0]


def test_every_kind_of_docstring_is_scanned(tmp_path: Path) -> None:
    """Module, class, method, function, and async function docstrings are all reached.

    `ast.walk` is what reaches the method nested inside the class, so a regression that walked
    only the top level would report three violations here rather than five.
    """
    bad_list = "\n\n    - parent\n      - child\n    "
    path = _write_module(
        tmp_path / "every.py",
        f'"""Module.{bad_list.replace("\n    ", "\n")}"""\n'
        "\n"
        "\n"
        "class C:\n"
        f'    """Class.{bad_list}"""\n'
        "\n"
        "    def m(self) -> None:\n"
        f'        """Method.{bad_list.replace("\n    ", "\n        ")}"""\n'
        "\n"
        "\n"
        "def f() -> None:\n"
        f'    """Function.{bad_list}"""\n'
        "\n"
        "\n"
        "async def g() -> None:\n"
        f'    """Async.{bad_list}"""\n',
    )
    violations = lint_docstring_markdown._lint_file(path)
    assert len(violations) == 5
    assert all(NESTED_LIST_VIOLATION in violation for violation in violations)


def _run_script(paths: list[Path]) -> subprocess.CompletedProcess[str]:
    """Run the script as a subprocess over `paths`, the way the pre-commit hook does."""
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_clean_files_exit_zero(tmp_path: Path) -> None:
    """The whole script exits 0 and prints nothing when every docstring is clean."""
    paths = [
        _write_module(tmp_path / f"clean_{index}.py", '"""Ordinary summary."""\n')
        for index in range(3)
    ]
    result = _run_script(paths)
    assert result.returncode == 0
    assert result.stdout == ""


def test_a_violation_exits_one_and_is_printed(tmp_path: Path) -> None:
    """The whole script exits 1 and names the offending file and line."""
    path = _write_module(tmp_path / "bad.py", '"""Summary.\n\n- parent\n  - child\n"""\n')
    result = _run_script([path])
    assert result.returncode == 1
    assert f"{path}:4:" in result.stdout


def test_pool_and_serial_paths_agree(tmp_path: Path) -> None:
    """Above `POOL_FILE_COUNT` files the work moves to a process pool, which must find the same set.

    The pool is the only part of the script whose behaviour depends on how many files it is
    given, so it is worth running one batch either side of the threshold and comparing.
    """
    pool_size = lint_docstring_markdown.POOL_FILE_COUNT
    paths = [
        _write_module(tmp_path / f"m_{index}.py", '"""Summary.\n\n- parent\n  - child\n"""\n')
        for index in range(pool_size + 1)
    ]
    serial = _run_script(paths[: pool_size - 1])
    pooled = _run_script(paths)

    assert serial.returncode == 1
    assert pooled.returncode == 1
    assert len(serial.stdout.splitlines()) == pool_size - 1
    assert len(pooled.stdout.splitlines()) == pool_size + 1
    assert set(serial.stdout.splitlines()) <= set(pooled.stdout.splitlines())
