"""Report docstrings that document a parameter the function does not have.

Ruff's pydocstyle rules are configured with the google convention, which turns on D417 — but D417
only fires when an ``Args:`` section is *present and incomplete*. Measured against a probe file,
three other ways a docstring can misdescribe a signature pass it silently:

- The parameters are documented under ``Attributes:`` (or ``Parameters:``, or ``Arguments:``)
  rather than ``Args:``. D417 sees no ``Args:`` section and says nothing, and neither does any
  other rule, so the parameter documentation never renders as parameter documentation.
- The ``Args:`` section documents a name that is not a parameter — because the parameter was
  renamed or removed and the docstring was not.
- There is no parameter documentation at all. D417 declines to flag this by design, and so does
  this script: not every function needs an ``Args:`` block, and a rule that demanded one would be
  noise.

Both of the first two were found by hand in ``geo.h3`` in September 2026, months after the rename
that caused them: ``compute_h3_grid_weights`` documented ``grid_size`` and ``child_res`` after both
had been renamed, and ``compute_h3_grid_weights_for_boundary`` headed its four parameters
``Attributes:``. A docstring that names a parameter the reader cannot find is worse than one that
says nothing, because the reader trusts it and goes looking.

**Only functions and methods are checked, never classes.** A class docstring legitimately carries
an ``Attributes:`` section describing its attributes, and in this repo a Patito or pydantic model's
fields are exactly that. Checking classes would turn every schema in ``contracts`` into a finding.

**The wrong-heading check requires the section to name at least one real parameter.** A function
whose ``Attributes:`` section happens to describe something else — a module-level constant it
mutates, say — is not documenting its parameters under the wrong heading, and flagging it would be
a false positive. Requiring an overlap with the signature keeps the check specific.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Final, NamedTuple

_ARGS_HEADING: Final[str] = "Args"
"""The one google-convention heading under which parameters belong."""

_MISPLACED_HEADINGS: Final[tuple[str, ...]] = ("Attributes", "Parameters", "Arguments")
"""Headings that hold parameters in some other convention, and so get written here by habit.

``Parameters`` is numpydoc's spelling and ``Arguments`` an older Google one; ``Attributes`` is a
real google-convention section, but on a *function* a block of parameter names under it is the
mistake this script exists to catch.
"""

_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<name>[A-Z][A-Za-z ]*):\s*$")
"""A google-convention section heading: a capitalised word at the start of a line, then a colon."""

_ENTRY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_]\w*)\s*(\([^)]*\))?:"
)
"""One documented parameter: its name, an optional ``(type)``, then a colon.

The optional leading asterisks match ``*args`` and ``**kwargs``, which are documented under the
names they are declared with.
"""


class Finding(NamedTuple):
    """One docstring that misdescribes its function's signature."""

    path: Path
    line: int
    function: str
    message: str

    def __str__(self) -> str:
        """One finding on one line, in the file:line:message shape an editor can jump to."""
        return f"{self.path}:{self.line}: {self.function}: {self.message}"


def _sections(docstring: str) -> dict[str, list[str]]:
    """Split a google-convention docstring into ``{heading: [body line, ...]}``.

    Lines before the first heading belong to no section and are dropped, because this script only
    ever asks about the parameter sections.

    A section ends at the next non-blank line indented no further than its own heading. Without
    that rule a paragraph written *after* a parameter block — which ``geo.h3`` really had — is
    swallowed into the section, and because it sits at the docstring's base indent it redefines
    what "an entry" is indented to, so every real entry is then skipped and the section reads as
    documenting nothing.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    heading_indent = 0
    for raw_line in docstring.splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip())
        heading = _SECTION_PATTERN.match(stripped)
        if heading and not stripped.startswith(("http", "https")):
            current = heading.group("name").strip()
            heading_indent = indent
            sections[current] = []
        elif current is not None:
            if stripped and indent <= heading_indent:
                current = None  # Dedented out of the section, back to ordinary docstring prose.
            else:
                sections[current].append(raw_line)
    return sections


def _documented_names(body_lines: list[str]) -> set[str]:
    """The parameter names a section documents.

    Only lines at the section's own base indent start an entry; a continuation line is indented
    further and is skipped, so a description mentioning ``foo: bar`` cannot be read as an entry.
    """
    entries = [line for line in body_lines if line.strip()]
    if not entries:
        return set()
    base_indent = min(len(line) - len(line.lstrip()) for line in entries)
    return {
        match.group("name").lstrip("*")
        for line in entries
        if len(line) - len(line.lstrip()) == base_indent
        and (match := _ENTRY_PATTERN.match(line.strip()))
    }


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every parameter the function declares, less ``self`` and ``cls``."""
    args = node.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names - {"self", "cls"}


def check_file(path: Path) -> list[Finding]:
    """Every docstring in one file that documents a parameter its function does not have."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError, UnicodeDecodeError:
        return []  # Not ours to report: ruff and the formatter already fail on an unparseable file.

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        line = node.body[0].lineno
        parameters = _parameter_names(node)
        sections = _sections(docstring)

        documented = _documented_names(sections.get(_ARGS_HEADING, []))
        if stale := sorted(documented - parameters):
            findings.append(
                Finding(
                    path,
                    line,
                    node.name,
                    f"`Args:` documents {', '.join(repr(name) for name in stale)}, which"
                    f" {'is' if len(stale) == 1 else 'are'} not a parameter of this function.",
                )
            )

        for heading in _MISPLACED_HEADINGS:
            misplaced = _documented_names(sections.get(heading, [])) & parameters
            if misplaced:
                findings.append(
                    Finding(
                        path,
                        line,
                        node.name,
                        f"`{heading}:` documents the parameters"
                        f" {', '.join(repr(name) for name in sorted(misplaced))}. Parameters"
                        " belong under `Args:`, which is the section the google convention and"
                        " mkdocstrings both render as parameters.",
                    )
                )
    return findings


def _tracked_python_files() -> list[Path]:
    """Every git-tracked ``.py`` file, less the marimo notebooks.

    A marimo notebook is generated, and its cells are functions whose parameters are the names
    other cells export — a shape this check has nothing useful to say about.
    """
    listing = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    paths = [Path(name) for name in listing]
    return [
        path for path in paths if not path.read_text(encoding="utf-8").startswith("import marimo")
    ]


def main(argv: list[str]) -> int:
    """Check the named files, or every tracked Python file when none are named."""
    paths = [Path(name) for name in argv] if argv else _tracked_python_files()
    findings = [finding for path in paths if path.suffix == ".py" for finding in check_file(path)]
    for finding in findings:
        print(finding)
    print(f"checked {len(paths)} files, {len(findings)} docstring(s) disagreeing with a signature")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
