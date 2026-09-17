"""Prove that a prose sweep changed prose and nothing else.

A sweep of docstrings and comments has no business changing behaviour, but the edits land in the
same files the interpreter reads, and a mis-placed splice can close a string early or turn a comment
into code. Nothing else in the toolchain notices: `ruff`, `ty` and `pytest` all pass on a file whose
docstring now says the opposite of what it said, and equally on one where an edit moved a line of
code.

The check is to parse each file at a git revision and again in the working tree, blank every string
constant, and compare `ast.dump()`. Blanking the strings is what makes the comparison a prose test
rather than a diff: a docstring rewrite changes a string constant and nothing else, so the two dumps
agree. A comment cannot reach the tree at all, so comment edits are invisible here by construction.
Anything that survives the blanking is a behavioural change, and belongs in the pull-request body as
a list a reviewer can reject as a unit.

A file that fails to parse is reported as `unparseable`, which is the loudest failure this can find
and always a real one.

Usage::

    python3 check_prose_only.py <rev> <path> [<path> ...]

Exits non-zero when any file's behaviour changed, listing each file and why.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Final

BLANK: Final[str] = ""
"""What every string constant becomes before the two trees are compared."""


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

    Args:
        rev: The git revision to read.
        path: The file to read.

    Returns:
        The file's text at that revision, or None.
    """
    completed = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def verdict_for(*, rev: str, path: Path) -> str:
    """Return the one-word verdict for one file.

    Args:
        rev: The git revision to compare the working tree against.
        path: The file to check.

    Returns:
        `prose-only`, `new file`, `unparseable`, or `BEHAVIOUR CHANGED`.
    """
    before = _at_revision(rev=rev, path=path)
    if before is None:
        return "new file"
    try:
        after = path.read_text(encoding="utf-8")
        changed = structure_of(before) != structure_of(after)
    except SyntaxError:
        return "unparseable"
    return "BEHAVIOUR CHANGED" if changed else "prose-only"


def main() -> None:
    """Check every path named on the command line and exit non-zero on any behavioural change."""
    arguments = sys.argv[1:]
    if len(arguments) < 2:
        sys.exit(__doc__)
    rev, *names = arguments

    failures: list[str] = []
    for name in names:
        path = Path(name)
        if path.suffix != ".py":
            continue
        verdict = verdict_for(rev=rev, path=path)
        print(f"{verdict:>18}  {name}")
        if verdict in {"BEHAVIOUR CHANGED", "unparseable"}:
            failures.append(name)

    if failures:
        sys.exit(f"\nFAIL - {len(failures)} file(s) changed more than prose: {', '.join(failures)}")
    print("\nOK - every Python file changed prose only")


if __name__ == "__main__":
    main()
