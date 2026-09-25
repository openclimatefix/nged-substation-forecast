"""Stop unless every number the weather-station section adds to the page is in `report.md`.

One-off throwaway check for the weather-station addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. It reads the lines the
page gained since `--base` (default `origin/main`), pulls out every number that is not a single
digit, and requires each to appear as a number in `station_past_solar.py`'s `report.md`, or to be
one of `ALLOWED` with its reason. A sign, a thousands separator, and a trailing percent sign are
ignored, so "−2.082" matches "-2.082" and "5.299%" matches "5.299". A single digit is ignored
because the page writes small counts as words or as digits in ordinary prose ("all 5 folds").

Run it with `uv run python studies/beam_diffuse_split/check_station_page_numbers.py`, after
`station_past_solar.py` has written its report and the section is in the page.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from station_past_solar import OUTPUT_DIR

PAGE: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "studies" / "past-weather" / "solar.md"
)
NUMBER: Final[re.Pattern[str]] = re.compile(r"\d[\d,]*(?:\.\d+)?")
URL_OR_ANCHOR: Final[re.Pattern[str]] = re.compile(r"\]\([^)]*\)|`[^`]*`|https?://\S+")

ALLOWED: Final[dict[str, str]] = {
    "100": "the 100% coverage threshold the script's docstring says leaves a farm with no station",
}


def _normalise(*, text: str) -> str:
    return text.replace(",", "")


def added_lines(*, base: str) -> list[str]:
    """Return the page's lines gained since `base`.

    Args:
        base: A git revision.

    Returns:
        The added lines, without their leading plus sign.
    """
    diff = subprocess.run(
        ["git", "diff", base, "--unified=0", "--", str(PAGE)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def main() -> int:
    """Check every non-trivial number in the added lines against the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    arguments = parser.parse_args()
    report = (OUTPUT_DIR / "report.md").read_text()
    printed = {_normalise(text=token) for token in NUMBER.findall(report)}
    text = URL_OR_ANCHOR.sub(" ", "\n".join(added_lines(base=arguments.base)))
    text = re.sub(r"Figures?\s+\d+(?:\s+to\s+\d+)?", " ", text)
    missing: dict[str, int] = {}
    checked = 0
    for token in NUMBER.findall(text):
        token = _normalise(text=token)  # noqa: PLW2901
        if len(token) == 1:
            continue
        checked += 1
        if token not in printed and token not in ALLOWED:
            missing[token] = missing.get(token, 0) + 1
    sys.stdout.write(f"{checked} numbers checked against report.md\n")
    if missing:
        sys.stdout.write(f"{len(missing)} not in the report: {sorted(missing)}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
