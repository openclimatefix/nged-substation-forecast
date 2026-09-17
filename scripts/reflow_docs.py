"""Rewrap every markdown file given on the command line to `markdown_wrap.WIDTH`.

Headings, tables, fenced and indented code blocks, and YAML frontmatter are left untouched: only
prose paragraphs, list items and blockquotes are re-flowed. A markdown link is kept whole even when
its label contains spaces, so a line break can never fall inside `[label](target)`.

Usage::

    python3 scripts/reflow_docs.py docs/index.md README.md ...
"""

import sys
from pathlib import Path

from markdown_wrap import reflow_text


def main() -> None:
    """Rewrap each file named on the command line, in place."""
    if len(sys.argv) < 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <file.md> [<file.md> ...]")
    for arg in sys.argv[1:]:
        path = Path(arg)
        source = path.read_text()
        rewritten = reflow_text(source)
        if rewritten != source:
            path.write_text(rewritten)
            print(f"{path}: reflowed")
        else:
            print(f"{path}: unchanged")


if __name__ == "__main__":
    main()
