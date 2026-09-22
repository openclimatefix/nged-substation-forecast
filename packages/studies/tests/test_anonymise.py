import pytest
from studies.anonymise import site_labels_for

# Placeholder identifiers, not NGED's. The labels are a shuffle over the sorted identifiers, so any
# six pin the permutation the published write-up rests on. A change to the seed, to the generator,
# or to SITE_LABELS' order would relabel every site and orphan every label already in print, and
# this is what notices. These six iterate out of order as a Python set, so dropping the sort is
# noticed too.
PUBLISHED_LABELS: dict[int, str] = {97: "A", 4: "B", 65: "C", 33: "D", 6: "E", 2: "F"}


def test_reproduces_the_published_labels():
    assert site_labels_for(eligible_ids=list(PUBLISHED_LABELS)) == PUBLISHED_LABELS


def test_a_roster_of_the_wrong_size_raises():
    with pytest.raises(ValueError, match="eligible generators to label"):
        site_labels_for(eligible_ids=[*PUBLISHED_LABELS, 99])
