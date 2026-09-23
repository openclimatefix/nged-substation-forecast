"""The one mapping from a metered generator's identifier to the label a study may publish."""

from collections.abc import Sequence
from typing import Final

import numpy as np

SITE_LABELS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
"""The anonymous labels, in the order the permutation below is drawn against.

A metered generator's output is commercially sensitive, so NGED have asked that it leaves the
project anonymised. Nothing a study writes — a chart, a table, a results file, a write-up — may
carry a `time_series_id`, a site name, or a coordinate.
"""

LABEL_PERMUTATION_SEED: Final[int] = 784
"""Seeds the shuffle that assigns `SITE_LABELS` to identifiers.

**What protects the mapping is that the roster is private, not that the labels are shuffled.** The
value is load-bearing only for consistency: it fixes which generator is called `A`, and the
published write-up refers to the sites by these labels. Changing the seed, the generator, or
`SITE_LABELS`' order silently relabels every site and orphans every label already in print.
"""


WIND_SITE_LABELS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous labels for the metered wind generators, drawn separately from the solar ones."""

WIND_LABEL_PERMUTATION_SEED: Final[int] = 826
"""Seeds the shuffle that assigns `WIND_SITE_LABELS` to identifiers.

The same rules as `LABEL_PERMUTATION_SEED` hold: changing it relabels every wind generator and
orphans every label in print.
"""


def site_labels_for(
    *,
    eligible_ids: Sequence[int],
    labels: Sequence[str] = SITE_LABELS,
    seed: int = LABEL_PERMUTATION_SEED,
) -> dict[int, str]:
    """Map each eligible generator's identifier to its anonymous label.

    **One implementation, because a second derivation is a second chance to disagree with the
    first.** The mapping is a shuffle over a sorted list, so it is stable only while the list is:
    two callers that select their generators differently get different labels for the same
    generator, and nothing about the output says so. Deriving it here means each caller's argument
    is the only thing it can get wrong, and that argument is visible at the call site.

    Args:
        eligible_ids: Every generator in the roster. Order is irrelevant — the identifiers are
            sorted here — but membership is not.
        labels: The labels to shuffle: `SITE_LABELS` for the solar generators, or
            `WIND_SITE_LABELS` for the wind generators.
        seed: The seed paired with those labels.

    Returns:
        One entry per identifier, mapping it to its label.

    Raises:
        ValueError: If the roster does not hold exactly `len(labels)` generators, which means the
            caller's selection has drifted from the one the labels were drawn against.
    """
    if len(set(eligible_ids)) != len(labels):
        msg = (
            f"expected {len(labels)} eligible generators to label, got "
            f"{len(set(eligible_ids))}. The labels are a shuffle over a fixed-length list, so a "
            "roster of a different size cannot be labelled without relabelling the rest."
        )
        raise ValueError(msg)
    shuffled = np.random.default_rng(seed).permutation(list(labels))
    return dict(zip(sorted(set(eligible_ids)), (str(label) for label in shuffled), strict=True))
