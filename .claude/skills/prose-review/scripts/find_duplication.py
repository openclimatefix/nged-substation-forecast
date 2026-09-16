"""Find prose that more than one page carries, as a starting hint for a cross-page pass.

An idea should live on the one page a reader would look for it on, and every other page should
link to that page rather than explaining the idea again. This script finds the pages that repeat
each other *verbatim*, by counting the overlapping word runs they share.

**Read the output as a hint, never as the work list.** Verbatim overlap is the small half of the
problem: across this repo's docs it accounts for only 3% to 15% of any page, while the redundancy
that actually costs a reader is the same idea written out twice in different words, which no
shingle count can see. Use the pairs below to choose which pages to read side by side, then find
the real duplication by reading them.

Usage::

    python3 find_duplication.py docs
    python3 find_duplication.py docs --pairs 30
    python3 find_duplication.py docs/roadmap docs/techniques

Prints the page pairs sharing the most word runs, then the pages with the largest share of their
own prose appearing somewhere else. Always exits zero: duplication is a judgement, not a gate.
"""

from __future__ import annotations

import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

SHINGLE_LEN: Final[int] = 8
"""Word runs shorter than this match by coincidence; longer ones miss a lightly reworded copy."""

MAX_PAGES_PER_RUN: Final[int] = 4
"""A run on more than this many pages is boilerplate — a nav line, a heading, a stock phrase."""

MIN_RUNS_FOR_SHARE: Final[int] = 400
"""Below roughly this length a page's duplication share swings wildly on one shared sentence."""

DEFAULT_PAIRS: Final[int] = 20
"""How many page pairs to print unless `--pairs` says otherwise."""


def shingles(text: str) -> set[tuple[str, ...]]:
    """Every overlapping run of `SHINGLE_LEN` words, with code blocks and link targets removed.

    A link's target is dropped and its label kept, so two pages linking the same URL do not read
    as sharing prose, while two pages using the same sentence do.
    """
    stripped = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    stripped = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped)
    words = re.sub(r"[^\w\s]", " ", stripped.lower()).split()
    return {tuple(words[at : at + SHINGLE_LEN]) for at in range(len(words) - SHINGLE_LEN + 1)}


def _pages(roots: list[str]) -> dict[str, set[tuple[str, ...]]]:
    """Every markdown file under `roots`, mapped to its set of word runs."""
    found: dict[str, set[tuple[str, ...]]] = {}
    for root in roots:
        path = Path(root)
        files = sorted(path.rglob("*.md")) if path.is_dir() else [path]
        for markdown in files:
            found[str(markdown)] = shingles(markdown.read_text(encoding="utf-8"))
    return found


def _owners(pages: dict[str, set[tuple[str, ...]]]) -> dict[tuple[str, ...], set[str]]:
    """Which pages carry each word run."""
    owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for page, runs in pages.items():
        for run in runs:
            owners[run].add(page)
    return owners


def _report_pairs(
    owners: dict[tuple[str, ...], set[str]],
    limit: int,
) -> None:
    """Print the page pairs sharing the most word runs, with one shared run as evidence."""
    shared: Counter[tuple[str, str]] = Counter()
    example: dict[tuple[str, str], str] = {}
    for run, pages in owners.items():
        if not 2 <= len(pages) <= MAX_PAGES_PER_RUN:
            continue
        for pair in itertools.combinations(sorted(pages), 2):
            shared[pair] += 1
            example.setdefault(pair, " ".join(run))

    print(f"{'shared runs':>11}  page pair")
    for first, second in [pair for pair, _ in shared.most_common(limit)]:
        print(f"{shared[(first, second)]:>11}  {first}")
        print(f"{'':>11}  {second}")
        print(f'{"":>11}    e.g. "{example[(first, second)]}"')


def _report_shares(
    pages: dict[str, set[tuple[str, ...]]],
    owners: dict[tuple[str, ...], set[str]],
) -> None:
    """Print how much of each page's own prose appears somewhere else as well."""
    rows: list[tuple[float, int, int, str]] = []
    for page, runs in pages.items():
        if len(runs) < MIN_RUNS_FOR_SHARE:
            continue
        duplicated = sum(1 for run in runs if len(owners[run]) > 1)
        rows.append((duplicated / len(runs) * 100, duplicated, len(runs), page))

    print(f"\n{'share':>6} {'shared':>8} {'runs':>7}  page")
    for share, duplicated, total, page in sorted(rows, reverse=True):
        print(f"{share:5.1f}% {duplicated:8} {total:7}  {page}")


def main() -> None:
    """Report the pages that repeat each other, most heavily overlapping pair first."""
    arguments = sys.argv[1:]
    if not arguments:
        sys.exit(__doc__)
    limit = DEFAULT_PAIRS
    if "--pairs" in arguments:
        at = arguments.index("--pairs")
        limit = int(arguments[at + 1])
        arguments = arguments[:at] + arguments[at + 2 :]

    pages = _pages(arguments)
    owners = _owners(pages)
    _report_pairs(owners, limit)
    _report_shares(pages, owners)
    print(
        "\nVerbatim overlap is the small half of the problem. Read each pair above side by side"
        "\nand ask what each page says that the other one also says, in whatever words."
    )


if __name__ == "__main__":
    main()
