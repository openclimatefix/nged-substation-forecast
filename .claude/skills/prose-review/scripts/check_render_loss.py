"""Find markdown content that MkDocs silently drops at render time.

Python-Markdown discards a table row's surplus cells when the row carries more cells than its
header, so a paragraph written into a fourth cell of a three-column table never reaches the
published site. `pymarkdown scan` and `mkdocs build --strict` both pass on it, and the page looks
correct in the repo, which is why this needs its own check.

Usage::

    python3 check_render_loss.py [path ...]

With no arguments, scans every markdown file under `docs/` plus the repo's top-level README and
CLAUDE.md. Exits non-zero when any row would lose a cell.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

DEFAULT_GLOBS: Final[tuple[str, ...]] = ("docs/**/*.md", "README.md", "CLAUDE.md")
DELIMITER_ROW: Final[re.Pattern[str]] = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$"
)


def cell_count(line: str) -> int:
    """Count a pipe-table row's cells, ignoring the optional leading and trailing pipes."""
    stripped = line.strip()
    stripped = stripped.removeprefix("|").removesuffix("|")
    # A pipe inside inline code or escaped with a backslash does not separate cells.
    stripped = re.sub(r"`[^`]*`", "CODE", stripped)
    stripped = stripped.replace(r"\|", "ESC")
    return len(stripped.split("|"))


def check(path: Path) -> list[str]:
    """Return one message per row that would lose cells in `path`."""
    problems: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    header_cells: int | None = None
    in_fence = False

    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if DELIMITER_ROW.match(line) and "|" in line and number >= 2:
            header_cells = cell_count(lines[number - 2])
            continue

        if header_cells is None:
            continue
        if "|" not in line or not line.strip():
            header_cells = None
            continue

        found = cell_count(line)
        if found > header_cells:
            lost = found - header_cells
            problems.append(
                f"{path}:{number}: row has {found} cells, header has {header_cells} "
                f"— {lost} cell(s) dropped at render: {line.strip()[-90:]!r}"
            )
    return problems


def main() -> None:
    """Scan the given paths, or the repo's markdown by default, and report dropped cells."""
    root = Path.cwd()
    if len(sys.argv) > 1:
        paths = [Path(arg) for arg in sys.argv[1:]]
    else:
        paths = sorted({p for pattern in DEFAULT_GLOBS for p in root.glob(pattern)})

    problems = [message for path in paths if path.is_file() for message in check(path)]

    if problems:
        for message in problems:
            print(message)
        sys.exit(f"\nFAIL - {len(problems)} table row(s) lose content when rendered")
    print(f"PASS - {len(paths)} file(s) scanned, no table row loses content when rendered")


if __name__ == "__main__":
    main()
