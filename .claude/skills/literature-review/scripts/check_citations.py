"""Check that a literature review's citations and its reference list agree.

Run over a markdown file whose references live under a `## References` heading, whose entries
look like::

    - Surname, A., Other, B. and Third, C. (2024). [Title](https://doi.org/...). *Journal*.

and whose in-text citations look like ``[Surname et al. (2024)](https://doi.org/...)``.

Usage::

    python3 check_citations.py path/to/review.md

Exits non-zero if any check fails, so it can go in a verification set or a pre-commit hook. The
file is assumed to be hard-wrapped, so every match runs against a whitespace-normalised copy of
the text while reported line numbers come from the wrapped file.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

NAME: Final[str] = r"[A-ZÀ-Ü][A-Za-zÀ-ſ'\-]+"
CITE_TEXT: Final[str] = r"\[([^\]\[]*?\(\d{4}[a-z]?\))\]\((https?://[^)]+)\)"
ANY_LINK: Final[str] = r"\[([^\]\[]*?)\]\((https?://[^)]+)\)"
BARE_CITE: Final[str] = (
    r"\b(" + NAME + r"(?:\s+(?:et al\.|and\s+" + NAME + r"|&\s+" + NAME + r"))?\s*\(\d{4}[a-z]?\))"
)
ENTRY_HEAD: Final[str] = r"\((\d{4}[a-z]?)\)\.\s*\[(.*?)\]\((https?://[^)]+)\)"
REFS_HEADING: Final[str] = "## References"


@dataclass(frozen=True)
class Reference:
    """One entry from the reference list at the end of the review."""

    entry: str
    """The whole entry, whitespace-normalised."""

    urls: tuple[str, ...]
    """Every URL the entry links to, canonicalised."""

    year: str | None
    """Publication year, or `None` for a live web resource that has no meaningful year."""

    authors: str
    """The author list as written, e.g. `Surname, A., Other, B. and Third, C.`."""

    surname: str
    """First author's surname, used for the alphabetical-order check."""


def canonical_url(url: str) -> str:
    """Return a canonical form of `url`, so one source linked two ways still matches itself."""
    stripped = url.strip().rstrip(".").rstrip("/")
    stripped = re.sub(r"^https?://(www\.)?", "", stripped)
    stripped = re.sub(r"^arxiv\.org/pdf/", "arxiv.org/abs/", stripped)
    if "arxiv.org/abs" in stripped:
        stripped = re.sub(r"v\d+$", "", stripped)
    return stripped.lower()


def _normalise(text: str) -> str:
    """Collapse every run of whitespace to a single space."""
    return re.sub(r"\s+", " ", text)


def find_line(lines: list[str], needle: str) -> int:
    """Return the 1-based line where `needle` starts, allowing for hard wrapping.

    Args:
        lines: The file's lines, unwrapped and in order.
        needle: Whitespace-normalised text to locate.

    Returns:
        The 1-based line number, or 0 if `needle` was not found.
    """
    for index in range(len(lines)):
        if needle in _normalise(" ".join(lines[index : index + 4])):
            return index + 1
    return 0


def parse_references(refs: str) -> list[Reference]:
    """Parse the bulleted reference list into one `Reference` per entry."""
    references: list[Reference] = []
    for raw in re.findall(r"^- (.+?)(?=\n- |\Z)", refs, re.DOTALL | re.MULTILINE):
        entry = _normalise(raw).strip()
        head = re.search(ENTRY_HEAD, entry)
        authors = entry[: head.start()].strip() if head else entry.split(".")[0]
        references.append(
            Reference(
                entry=entry,
                urls=tuple(canonical_url(u) for u in re.findall(r"\((https?://[^)]+)\)", entry)),
                year=head.group(1) if head else None,
                authors=authors,
                surname=authors.split(",")[0].strip(),
            )
        )
    return references


def check_hyperlinked(body: str, lines: list[str]) -> list[str]:
    """Return every citation-shaped string in the body that is not inside a hyperlink."""
    stripped = re.sub(ANY_LINK, "  ", body)
    found = sorted(set(re.findall(BARE_CITE, stripped)))
    return [f"line {find_line(lines, cite)}: {cite}" for cite in found]


def check_body_in_references(
    body_urls: dict[str, list[str]], reference_urls: set[str]
) -> list[str]:
    """Return every link in the body whose URL has no matching reference entry."""
    return [
        f"{body_urls[url][0]!r} -> {url}" for url in sorted(body_urls) if url not in reference_urls
    ]


def check_references_cited(
    references: list[Reference], body_urls: dict[str, list[str]]
) -> list[str]:
    """Return every reference entry whose URLs appear nowhere in the body."""
    return [ref.entry[:150] for ref in references if not any(url in body_urls for url in ref.urls)]


def check_duplicates(references: list[Reference]) -> list[str]:
    """Return every URL listed by more than one reference entry."""
    counts = Counter(url for ref in references for url in set(ref.urls))
    return [f"{count}x {url}" for url, count in counts.items() if count > 1]


def check_alphabetical(references: list[Reference]) -> list[str]:
    """Return every adjacent pair of entries that is out of alphabetical order by surname."""
    surnames = [ref.surname for ref in references]
    return [
        f"{first!r} precedes {second!r}"
        for first, second in pairwise(surnames)
        if first.lower() > second.lower()
    ]


def check_labels_match(
    citations: set[tuple[str, str]], reference_urls: dict[str, Reference]
) -> list[str]:
    """Return every in-text label whose surname or year disagrees with the entry it links to."""
    mismatches: list[str] = []
    for label, url in sorted(citations):
        ref = reference_urls.get(canonical_url(url))
        if ref is None:
            continue
        year_match = re.search(r"\((\d{4}[a-z]?)\)$", label)
        year = year_match.group(1) if year_match else "?"
        head = label[: year_match.start()] if year_match else label
        first_author = head.strip().split()[0].rstrip(",")
        if ref.year is not None and year != ref.year:
            mismatches.append(f"{label}: year {year} but the reference says {ref.year}")
        if first_author.lower() not in ref.authors.lower():
            mismatches.append(f"{label}: {first_author!r} is not among {ref.authors[:60]!r}")
    return mismatches


def _report(title: str, items: list[str]) -> int:
    """Print `items` under `title` if there are any, and return how many there were."""
    if items:
        print(f"\nFAIL - {title} ({len(items)})")
        for item in items:
            print("   ", item)
    return len(items)


def main(path: Path) -> int:
    """Run every check over `path` and return a process exit code."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if REFS_HEADING not in (line.strip() for line in lines):
        print(f"FAIL: no {REFS_HEADING!r} heading found")
        return 1
    split_at = next(i for i, line in enumerate(lines) if line.strip() == REFS_HEADING)

    body = _normalise("\n".join(lines[:split_at]))
    references = parse_references(refs="\n".join(lines[split_at:]))

    body_urls: dict[str, list[str]] = defaultdict(list)
    for label, url in re.findall(ANY_LINK, body):
        body_urls[canonical_url(url)].append(label)
    reference_urls: dict[str, Reference] = {}
    for ref in references:
        for url in ref.urls:
            reference_urls.setdefault(url, ref)
    citations = set(re.findall(CITE_TEXT, body))

    print(
        f"{path}: {len(citations)} distinct in-text citations, {len(body_urls)} distinct URLs "
        f"cited, {len(references)} reference entries"
    )

    failures = sum(
        (
            _report(
                "citation-shaped text in the body that is not hyperlinked",
                check_hyperlinked(body=body, lines=lines),
            ),
            _report(
                "cited in the body but missing from the reference list",
                check_body_in_references(body_urls=body_urls, reference_urls=set(reference_urls)),
            ),
            _report(
                "in the reference list but never cited in the body",
                check_references_cited(references=references, body_urls=body_urls),
            ),
            _report(
                "duplicate URLs in the reference list", check_duplicates(references=references)
            ),
            _report(
                "reference list out of alphabetical order",
                check_alphabetical(references=references),
            ),
            _report(
                "in-text label disagrees with its reference entry",
                check_labels_match(citations=citations, reference_urls=reference_urls),
            ),
        )
    )

    undated = [ref.entry[:110] for ref in references if ref.year is None]
    if undated:
        print(
            f"\nNOTE - {len(undated)} reference entries carry no year. That is right for a live "
            "web resource and wrong for a paper, so check each one:"
        )
        for entry in undated:
            print("   ", entry)

    print(
        "\nPASS - citations and reference list agree"
        if not failures
        else f"\n{failures} problem(s) found"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
