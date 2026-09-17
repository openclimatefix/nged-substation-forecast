"""Prove that a prose sweep changed the *structure* of no Python file.

A sweep of docstrings and comments has no business changing behaviour, but the edits land in the
same files the interpreter reads, and a mis-placed splice can close a string early or turn a comment
into code. Nothing else in the toolchain notices: `ruff`, `ty` and `pytest` all pass on a file whose
docstring now says the opposite of what it said, and equally on one where an edit moved a line of
code.

The check is to parse each file at a git revision and again in the working tree, blank every string
constant, and compare `ast.dump()`. Blanking the strings is what makes the comparison a prose test
rather than a diff: a docstring rewrite changes a string constant and nothing else, so the two dumps
agree.

**The comparison proves that the syntax tree is the same shape, and proves nothing at all about any
string value.** Every one of these reports `prose-only`: renaming an entry in `__all__`, renaming a
dictionary key, changing a `Literal["fast", "slow"] = "fast"` default to `Literal["quick", "slow"] =
"slow"`, rewriting a doctest's expected output, editing the string a `match`/`case` or a `getattr`
names, and editing a decorator's string argument. A comment never reaches the tree at all, so a
deleted `# noqa`, a changed `# type:` pragma and a removed coding line are equally invisible.
`pytest` is what catches an edited runtime string, `ruff` what catches a deleted `# noqa`, and a
human reading the diff is what catches the rest.

A guard's one unforgivable outcome is a silent pass, so every reason a file could not be compared —
it does not parse at the revision, it does not parse now, it has gone from the working tree, it
cannot be decoded — is a failure rather than a skip, and a run that compared no Python file at all
fails too.

Usage::

    python3 check_prose_only.py <rev> <path> [<path> ...]

Each path is resolved against the current directory, at the revision as well as in the working
tree. Exits non-zero when any file's structure changed, listing each file and why.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

BLANK: Final[str] = ""
"""What every string constant becomes before the two trees are compared."""

VerdictType = Literal[
    "prose-only",
    "new file",
    "unparseable at rev",
    "unparseable",
    "unreadable",
    "gone from working tree",
    "BEHAVIOUR CHANGED",
]
"""What became of one file."""

FAILING_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        "unparseable at rev",
        "unparseable",
        "unreadable",
        "gone from working tree",
        "BEHAVIOUR CHANGED",
    }
)
"""The verdicts that make the run exit non-zero. Only `prose-only` and `new file` are a pass."""


class _StringBlanker(ast.NodeTransformer):
    """Replace every string constant in a tree with the empty string.

    Docstrings, prose comments' neighbouring literals and genuine runtime strings are blanked
    alike. Blanking a runtime string is deliberate: a sweep must not edit one, but if it does, the
    edit is a prose-shaped change to a literal rather than a change of behaviour in the structural
    sense this check tests for. `pytest` is what catches an edited runtime string.
    """

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Blank `node` if it holds a string, and return it either way.

        Args:
            node: The constant to consider.

        Returns:
            The constant, with a string value replaced by the empty string.
        """
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=BLANK, kind=node.kind), node)
        return node


def structure_of(source: str) -> str:
    """Return the dump of `source`'s syntax tree with every string constant blanked.

    Args:
        source: The full text of a Python file.

    Returns:
        A dump that is equal for two files differing only in their string constants.
    """
    tree = _StringBlanker().visit(ast.parse(source))
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _at_revision(*, rev: str, path: Path) -> str | None:
    """Return the text of `path` at `rev`, or None where the file did not exist there.

    `<rev>:./<path>` is git's cwd-relative form. Without the `./`, git resolves the path against
    the repository root, so run from any subdirectory every lookup fails, every file is reported
    `new file`, and this guard exits zero having compared nothing.

    Args:
        rev: The git revision to read.
        path: The file to read, relative to the current directory.

    Returns:
        The file's text at that revision, or None.
    """
    completed = subprocess.run(
        ["git", "show", f"{rev}:./{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def verify_revision(rev: str) -> None:
    """Exit unless `rev` names a commit in the repository the current directory sits in.

    Every per-file lookup below reads a git failure as "the file is new", which is a pass. A
    revision nobody can resolve would therefore turn the whole run into a pass, so it is settled
    once, here, before any file is looked at.

    Args:
        rev: The revision named on the command line.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.exit(f"FAIL - {rev!r} does not name a commit in this repository")


def verdict_for(*, rev: str, path: Path) -> VerdictType:
    """Return the one-word verdict for one file.

    Args:
        rev: The git revision to compare the working tree against.
        path: The file to check.

    Returns:
        One of the `VerdictType` members.
    """
    before = _at_revision(rev=rev, path=path)
    if before is None:
        return "new file"
    if not path.is_file():
        return "gone from working tree"
    try:
        after = path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return "unreadable"
    try:
        old_structure = structure_of(before)
    except SyntaxError:
        return "unparseable at rev"
    try:
        new_structure = structure_of(after)
    except SyntaxError:
        return "unparseable"
    return "BEHAVIOUR CHANGED" if old_structure != new_structure else "prose-only"


def main() -> None:
    """Check every path named on the command line and exit non-zero on any structural change."""
    parser = argparse.ArgumentParser(
        description="Prove a prose sweep changed the structure of no Python file."
    )
    parser.add_argument("rev", help="the git revision to compare the working tree against")
    parser.add_argument("paths", nargs="+", type=Path, help="the files to check")
    arguments = parser.parse_args()
    verify_revision(arguments.rev)

    failures: list[str] = []
    checked = 0
    for path in arguments.paths:
        if path.suffix != ".py":
            print(f"{'not python':>22}  {path}")
            continue
        checked += 1
        verdict = verdict_for(rev=arguments.rev, path=path)
        print(f"{verdict:>22}  {path}")
        if verdict in FAILING_VERDICTS:
            failures.append(str(path))

    if failures:
        sys.exit(f"\nFAIL - {len(failures)} file(s) changed more than prose: {', '.join(failures)}")
    if not checked:
        sys.exit("\nFAIL - no Python file was checked; every path named was something else")
    print(f"\nOK - every Python file changed prose only ({checked} checked)")


if __name__ == "__main__":
    main()
