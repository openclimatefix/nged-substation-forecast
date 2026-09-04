"""Unit tests for ``scripts/check_docstring_signatures.py``.

The cases below are the three ruff's D417 lets through, plus the shapes that must *not* be
reported. The false-positive cases carry as much weight as the true-positive ones: a checker that
fires on a class's ``Attributes:`` block, or on a description that merely contains a colon, gets
switched off, and then it protects nothing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "check_docstring_signatures.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `check_docstring_signatures.py` from its path in `scripts/`."""
    spec = importlib.util.spec_from_file_location("check_docstring_signatures", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_docstring_signatures = _load_script()
check_file = check_docstring_signatures.check_file


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_stale_args_entry_is_reported(tmp_path: Path) -> None:
    """An ``Args:`` entry naming a parameter that was renamed away is the core case."""
    findings = check_file(
        _write(
            tmp_path,
            '''
def f(new_name: int) -> int:
    """Do a thing.

    Args:
        old_name: what it used to be called.
    """
    return new_name
''',
        )
    )
    assert len(findings) == 1, findings
    assert "'old_name'" in findings[0].message
    assert findings[0].function == "f"


def test_parameters_under_attributes_are_reported(tmp_path: Path) -> None:
    """The ``geo.h3`` shape: a real parameter block under the wrong heading."""
    findings = check_file(
        _write(
            tmp_path,
            '''
def f(alpha: int, beta: int) -> int:
    """Do a thing.

    Attributes:
        alpha: the first.
        beta: the second.
    """
    return alpha + beta
''',
        )
    )
    assert len(findings) == 1, findings
    assert "`Attributes:`" in findings[0].message


def test_prose_after_a_parameter_block_does_not_hide_it(tmp_path: Path) -> None:
    """A paragraph *after* the parameter block must not swallow the section.

    ``geo.h3`` had exactly this layout, and it made the section read as documenting nothing, so
    the misplaced heading went unreported. This is the regression test for that.
    """
    findings = check_file(
        _write(
            tmp_path,
            '''
def f(alpha: int) -> int:
    """Do a thing.

    Attributes:
        alpha: the first.

    A trailing paragraph that sits at the docstring's own indent.
    """
    return alpha
''',
        )
    )
    assert len(findings) == 1, findings
    assert "'alpha'" in findings[0].message


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            '''
class C:
    """A class.

    Attributes:
        alpha: a real attribute.
    """

    alpha: int
''',
            "a class's Attributes block is correct google convention",
        ),
        (
            '''
def f(alpha: int) -> int:
    """Do a thing.

    Args:
        alpha: the first.
    """
    return alpha
''',
            "a correct Args block",
        ),
        (
            '''
def f(alpha: int) -> int:
    """Do a thing with no sections at all."""
    return alpha
''',
            "no parameter documentation is D417's business, not this script's",
        ),
        (
            '''
def f(*args: int, **kwargs: int) -> int:
    """Do a thing.

    Args:
        *args: positional.
        **kwargs: keyword.
    """
    return len(args) + len(kwargs)
''',
            "starred parameters are documented under their declared names",
        ),
        (
            '''
def f(alpha: int) -> int:
    """Do a thing.

    Args:
        alpha: a description whose continuation mentions beta: not an entry.
    """
    return alpha
''',
            "a colon inside a continuation line is not a new entry",
        ),
        (
            '''
class C:
    def f(self, alpha: int) -> int:
        """Do a thing.

        Args:
            alpha: the first.
        """
        return alpha
''',
            "self is not a documentable parameter",
        ),
    ],
)
def test_shapes_that_must_not_be_reported(tmp_path: Path, source: str, reason: str) -> None:
    assert check_file(_write(tmp_path, source)) == [], reason


def test_the_repository_is_clean() -> None:
    """The whole repo passes, so the hook starts green and any later failure is a real one."""
    findings = [
        finding
        for path in check_docstring_signatures._tracked_python_files()
        for finding in check_file(REPO_ROOT / path)
    ]
    assert findings == [], "\n".join(str(finding) for finding in findings)
