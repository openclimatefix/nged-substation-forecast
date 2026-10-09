import difflib

import pytest
from studies.name_matching import MATCH_THRESHOLD, best_match, normalise


def _ratio(first: str, second: str) -> float:
    return difflib.SequenceMatcher(None, normalise(first), normalise(second)).ratio()


def test_a_name_that_is_88_per_cent_similar_matches() -> None:
    site, candidate = "abcdefghijklmnopqrstuvwxy", "abcdefghijklmnopqrstuvQQQ"
    assert _ratio(site, candidate) == pytest.approx(0.88)

    assert best_match(site_name=site, candidates={"k": candidate}) == ("k", 0.88)


def test_a_name_exactly_at_the_threshold_matches_and_one_below_does_not() -> None:
    site = "abcdefghijklmnopqrst"
    at_threshold = "abcdefghijklmnopqXYZ"
    below_threshold = "abcdefghijklmnopXYZW"
    assert _ratio(site, at_threshold) == MATCH_THRESHOLD
    assert _ratio(site, below_threshold) < MATCH_THRESHOLD

    assert best_match(site_name=site, candidates={"k": at_threshold}) == ("k", 0.85)
    assert best_match(site_name=site, candidates={"k": below_threshold}) is None


def test_a_short_generic_word_is_dropped_only_when_it_stands_alone() -> None:
    assert normalise("Oltdo") == "oltdo"  # "ltd" has fewer than four letters
    assert normalise("Oltdo Ltd") == "oltdo"
    assert normalise("Xfarmx") == "xx"  # "farm" has four letters, so it goes inside a token
