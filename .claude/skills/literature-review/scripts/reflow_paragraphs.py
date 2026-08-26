"""Reflow the paragraphs of a markdown file that contain a given anchor phrase.

A wrap-tolerant substitution leaves a paragraph's line breaks where they were, so the edited lines
overrun the file's width while the rest of the document stays put. Reflowing the whole file instead
buries the real edit in a diff nobody can review, so this script rewraps only the paragraphs an
anchor phrase identifies, and asserts that nothing but the wrapping changed.
"""

import sys
from pathlib import Path
from typing import Final

WIDTH: Final[int] = 100
"""Column at which prose is wrapped, matching the repo's markdown line length."""

SKIP_PREFIXES: Final[tuple[str, ...]] = ("#", "|", "```", "- ", "* ", ">", "    ")
"""Block openings whose line breaks carry meaning, so they must never be rewrapped."""


def _wrap(words: list[str]) -> list[str]:
    """Greedily wrap `words` to `WIDTH`, keeping any line from starting with a `#`.

    Python-Markdown reads a line starting `#` as a heading even without the space CommonMark
    requires, so a wrapped link whose continuation lands on `#anchor](url)` renders as a heading.
    Any such line is pulled back onto its predecessor.
    """
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > WIDTH and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    for i in range(1, len(lines)):
        if lines[i].startswith("#"):
            head, _, tail = lines[i].partition(" ")
            lines[i - 1] += f" {head}"
            lines[i] = tail
    return [line for line in lines if line]


def reflow(path: Path, anchors: list[str]) -> int:
    """Rewrap every paragraph in `path` containing one of `anchors`, and return how many.

    Args:
        path: The markdown file to rewrite in place.
        anchors: Phrases identifying the paragraphs to rewrap. Matching ignores the existing line
            breaks, so an anchor may span them.

    Raises:
        AssertionError: If rewrapping changed the text rather than only its line breaks.
    """
    source = path.read_text()
    paragraphs = source.split("\n\n")
    flat_anchors = [" ".join(anchor.split()) for anchor in anchors]
    reflowed = 0

    for i, paragraph in enumerate(paragraphs):
        flat = " ".join(paragraph.split())
        if not any(anchor in flat for anchor in flat_anchors):
            continue
        if paragraph.lstrip().startswith(SKIP_PREFIXES) or paragraph.startswith("    "):
            continue
        paragraphs[i] = "\n".join(_wrap(words=flat.split()))
        reflowed += 1

    rewritten = "\n\n".join(paragraphs)
    assert "".join(rewritten.split()) == "".join(source.split()), (
        "reflow changed the text, not just the wrapping"
    )
    if reflowed:
        path.write_text(rewritten)
    return reflowed


def main() -> None:
    """Rewrap the paragraphs named on the command line."""
    if len(sys.argv) < 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <file.md> <anchor phrase> [<anchor phrase> ...]")
    path = Path(sys.argv[1])
    reflowed = reflow(path=path, anchors=sys.argv[2:])
    print(f"{path}: reflowed {reflowed} paragraph(s)")


if __name__ == "__main__":
    main()
