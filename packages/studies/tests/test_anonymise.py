import pytest
from studies.anonymise import site_labels_for

# Placeholder identifiers, not NGED's. The labels are a shuffle over the sorted identifiers, so any
# six pin the permutation the published write-up rests on. A change to the seed, to the generator,
# or to SITE_LABELS' order would relabel every site and orphan every label already in print, and
# this is what notices.
PUBLISHED_LABELS: dict[int, str] = {6: "A", 2: "B", 5: "C", 4: "D", 3: "E", 1: "F"}


def test_reproduces_the_published_labels():
    assert site_labels_for(eligible_ids=list(PUBLISHED_LABELS)) == PUBLISHED_LABELS


def test_a_roster_of_the_wrong_size_raises():
    with pytest.raises(ValueError, match="eligible generators to label"):
        site_labels_for(eligible_ids=[*PUBLISHED_LABELS, 99])
