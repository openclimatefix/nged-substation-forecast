"""Match site names that two registers spell differently.

Used by the solar-BMU census to match BMUs to planning-database rows, and by the embedded-battery
census to match Embedded Capacity Register sites to BMUs.
"""

import difflib
import re
from typing import Final

MATCH_THRESHOLD: Final[float] = 0.85
"""The lowest name-similarity ratio at which a site name counts as a match.

At 0.8 a REPD row for a different farm with a one-letter-different name matched; at 0.85 that REPD
row did not match.
"""
MIN_SUBSTRING_NOISE_LENGTH: Final[int] = 4
NOISE_WORDS: Final[frozenset[str]] = frozenset(
    {"solar", "farm", "pv", "park", "project", "power", "ltd", "limited", "array", "energy"}
    | {"photovoltaic", "photovoltaics", "plant", "station", "extension", "bess", "battery"}
    | {"storage"}
)


def normalise(name: str) -> str:
    """Lower-case a site name and drop punctuation, spaces, and generic words.

    Whole generic words are dropped first. Generic words of at least `MIN_SUBSTRING_NOISE_LENGTH`
    letters are then also dropped inside a longer token, so "Energyfarm" and "Energy Farm"
    normalise alike.
    """
    words = [word for word in re.findall(r"[a-z0-9]+", name.lower()) if word not in NOISE_WORDS]
    joined = "".join(words)
    for noise in sorted(NOISE_WORDS, key=len, reverse=True):
        if len(noise) >= MIN_SUBSTRING_NOISE_LENGTH:
            joined = joined.replace(noise, "")
    return joined


def best_match(*, site_name: str, candidates: dict[str, str]) -> tuple[str, float] | None:
    """Return the key and similarity of the candidate name closest to `site_name`, or None.

    Args:
        site_name: The BMU's site name.
        candidates: Maps a key (a project identifier) to that candidate's name.

    Returns:
        The key and the similarity ratio (rounded to 2 decimal places) of the best candidate at or
        above `MATCH_THRESHOLD`. None when no candidate reaches the threshold, or when `site_name`
        is made only of generic words and normalises to nothing. A tie goes to the key that sorts
        first, so the result does not depend on dict order.
    """
    target = normalise(site_name)
    if not target:
        return None
    best: tuple[str, float] | None = None
    for key in sorted(candidates):
        score = difflib.SequenceMatcher(None, target, normalise(candidates[key])).ratio()
        if score >= MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (key, score)
    return None if best is None else (best[0], round(best[1], 2))
