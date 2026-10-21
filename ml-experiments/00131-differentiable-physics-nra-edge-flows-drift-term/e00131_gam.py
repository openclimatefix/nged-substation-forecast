"""Where experiment 116's frozen convex-model fits are stored, for experiment 131.

Experiment 131 fits no convex model. It reads the out-of-sample residuals that experiment 116's arm
`a0` wrote at each cut (`work/gam/a0/<site>_<cut>.parquet` in 116's private data folder), so that
the drift term is the only difference between the two experiments' residuals.
"""

from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import e00131_common as common

ArmType = Literal["a0", "a1"]
"""`a0` keeps the switching events in training, `a1` drops them."""

SITES: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D16",
    "D19",
    "D21",
    "D22",
)
"""Every Primary in the decision graph. The two series labelled G-c are left out."""


def job_path(*, arm: ArmType, site: str, cut: datetime) -> Path:
    """Return the file that holds experiment 116's fit for an arm, a site, and a cut."""
    return common.GAM_DIR / arm / f"{site}_{common.cut_label(cut=cut)}.parquet"
