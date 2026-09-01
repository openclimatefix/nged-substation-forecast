"""Check that a restructure of a markdown file dropped no information.

Compares a file's committed state against its current state and reports what the older version
contains that the newer one does not. Reordering, rewording and splitting paragraphs are all
expected during a restructure; losing a number, a citation, a quotation or a bolded term is not.

Usage::

    python3 check_information_loss.py <git-ref> path/to/page.md
    python3 check_information_loss.py HEAD docs/background/energy-forecasting-review.md
    python3 check_information_loss.py 34a8164a docs/roadmap/capacity-estimation.md

Exits non-zero when anything in the first three categories went missing, so it can go in a
verification set. Shingles are advisory and never fail the run: a deliberate rewording drops
shingles by design, so that section is a list to read rather than a gate to pass.

The file is assumed to be hard-wrapped, so every comparison runs against a whitespace-normalised
copy of the text.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

NUMBER: Final[str] = r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?(?![\w.])"
LINK_URL: Final[str] = r"\]\((https?://[^)\s]+)\)"
# The two quote families are matched separately: allowing a curly opener to close on a straight
# quote pairs marks belonging to different sentences and swallows the prose between them.
QUOTATIONS: Final[tuple[str, ...]] = (r"“([^“”]{25,}?)”", r"\"([^\"]{25,}?)\"")
BOLD_TERM: Final[str] = r"\*\*([^*]{3,80})\*\*"
SHINGLE_LEN: Final[int] = 9
MAX_LISTED: Final[int] = 40


def normalise(text: str) -> str:
    """Collapse the hard wrapping so a phrase spanning a line break still matches."""
    return re.sub(r"\s+", " ", text)


def code_blocks_removed(text: str) -> str:
    """Drop fenced code, so a renamed variable is not reported as a lost number."""
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def extract(text: str, pattern: str) -> set[str]:
    """Every distinct capture of `pattern`, stripped."""
    found = re.findall(pattern, text)
    return {(m if isinstance(m, str) else m[0]).strip() for m in found}


def quotations(text: str) -> set[str]:
    """Direct quotations only: a span between matching quote marks that reads as prose."""
    out: set[str] = set()
    for pattern in QUOTATIONS:
        for candidate in extract(text, pattern):
            # A link or a reference-list dash means the marks belong to different sentences.
            if "](" in candidate or ". - " in candidate:
                continue
            out.add(candidate)
    return out


def shingles(text: str) -> set[tuple[str, ...]]:
    """Every overlapping run of `SHINGLE_LEN` words, lowercased and stripped of punctuation."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {tuple(words[i : i + SHINGLE_LEN]) for i in range(len(words) - SHINGLE_LEN + 1)}


def report(label: str, lost: set[str], *, gate: bool) -> bool:
    """Print one category. Returns True when it should fail the run."""
    if not lost:
        print(f"PASS - no {label} lost")
        return False
    marker = "FAIL" if gate else "NOTE"
    print(f"\n{marker} - {len(lost)} {label} present before and absent now:")
    for item in sorted(lost)[:MAX_LISTED]:
        print(f"    {item[:110]}")
    if len(lost) > MAX_LISTED:
        print(f"    ... and {len(lost) - MAX_LISTED} more")
    return gate


def main() -> None:
    """Compare `path` at a git ref against its current state and report what went missing."""
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, path = sys.argv[1], sys.argv[2]

    try:
        before_raw = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        sys.exit(f"could not read {path} at {ref}: {exc.stderr.strip()}")

    after_raw = Path(path).read_text(encoding="utf-8")

    before = normalise(code_blocks_removed(before_raw))
    after = normalise(code_blocks_removed(after_raw))

    failed = False
    failed |= report("numbers", extract(before, NUMBER) - extract(after, NUMBER), gate=True)
    failed |= report("links", extract(before, LINK_URL) - extract(after, LINK_URL), gate=True)
    failed |= report("quotations", quotations(before) - quotations(after), gate=True)
    report("bolded terms", extract(before, BOLD_TERM) - extract(after, BOLD_TERM), gate=False)

    dropped = shingles(before) - shingles(after)
    if dropped:
        print(f"\nNOTE - {len(dropped)} {SHINGLE_LEN}-word runs no longer appear anywhere.")
        print("       Each is a deliberate rewording or a deletion; only you can tell which.")
        for run in sorted(dropped)[:MAX_LISTED]:
            print(f"    {' '.join(run)}")
        if len(dropped) > MAX_LISTED:
            print(f"    ... and {len(dropped) - MAX_LISTED} more")

    print()
    if failed:
        sys.exit("FAIL - the restructure dropped information that cannot be a rewording")
    print("PASS - nothing that cannot be a rewording went missing")


if __name__ == "__main__":
    main()
