import pytest
from studies.anonymise import SITE_LABELS, site_labels_for

# The six identifiers of the metered generators the beam/diffuse study ran on, and the labels the
# published write-up gives them. A change to the seed, to the generator, or to SITE_LABELS' order
# would relabel every site and orphan every label already in print, and this is what notices.
PUBLISHED_LABELS: dict[int, str] = {31: "A", 22: "B", 30: "C", 29: "D", 23: "E", 21: "F"}


def test_reproduces_the_published_labels():
    assert site_labels_for(eligible_ids=list(PUBLISHED_LABELS)) == PUBLISHED_LABELS


def test_the_roster_order_does_not_change_the_labels():
    assert site_labels_for(eligible_ids=sorted(PUBLISHED_LABELS, reverse=True)) == PUBLISHED_LABELS


def test_a_roster_of_the_wrong_size_raises():
    with pytest.raises(ValueError, match="eligible generators to label"):
        site_labels_for(eligible_ids=[*PUBLISHED_LABELS, 99])


def test_every_label_is_used_exactly_once():
    assert sorted(site_labels_for(eligible_ids=list(PUBLISHED_LABELS)).values()) == sorted(
        SITE_LABELS
    )
