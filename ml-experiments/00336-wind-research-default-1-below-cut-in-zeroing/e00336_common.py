"""Paths, constants, and small functions shared by the scripts of experiment 00336.

The scripts import experiment 00326's modules (which import experiments 00303, 00285, 00259, and
00230), so run them with
`PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms`.
Everything read from other experiments' folders is read only.
"""

import json
from pathlib import Path
from typing import Final

import e00326_common as c326  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c303 = c326.c303
EXPERIMENT_ID: Final[int] = 336
SLUG: Final[str] = "wind-research-default-1-below-cut-in-zeroing"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
MIX_DIR: Final[Path] = WORK_DIR / "mix"
FINAL_DIR: Final[Path] = WORK_DIR / "final"
STATS_DIR: Final[Path] = WORK_DIR / "stats"
FARMS: Final[tuple[str, ...]] = c326.FARMS
SEEDS: Final[tuple[int, ...]] = c326.SEEDS
MEMBERS: Final[int] = c326.MEMBERS
REPLICATE_START = c326.REPLICATE_START
PLACEBO_SEED: Final[int] = 336
CORES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "b1_cores.json"
)
"""Experiment 230's physics core, fitted per farm on the ERA5 training window."""
GENERIC_CUT_IN_M_S: Final[float] = 3.0
CUT_IN_RANGE_M_S: Final[tuple[float, float]] = (2.0, 5.0)

# Mixture arms fitted here. Each name is a file prefix in MIX_DIR.
# x1m*: member-level zeroing from the refit default booster (cut-in as named).
# x2*: training rows below cut-in dropped; x3*: training target zeroed there; *pl: shuffled mask.
CUT_IN_VARIANTS: Final[dict[str, float | None]] = {"": 0.0, "lo": -1.0, "hi": 1.0, "g3": None}
"""Suffix -> shift added to the farm's cut-in; `None` means the generic 3.0 m/s."""
FIT_ARMS: Final[tuple[str, ...]] = (
    "default",
    "x1m",
    "x1mlo",
    "x1mhi",
    "x1mg3",
    "x2",
    "x2e",
    "x2lo",
    "x2hi",
    "x2g3",
    "x3",
    "x2pl",
    "x3pl",
)
POST_HOC_ARMS: Final[tuple[str, ...]] = (
    "x1",
    "x1lo",
    "x1hi",
    "x1g3",
    "x1a",
    "x1pl",
)
"""Arms computed from the default's final quantiles: zero where the case mask is set."""


def cut_in_speeds() -> dict[str, float]:
    """Return each farm's cut-in speed (m/s) from experiment 230's fitted physics core.

    A fitted value outside 2 to 5 m/s is replaced by 3.0 m/s, as the plan states.
    """
    fitted = json.loads(CORES_PATH.read_text())
    low, high = CUT_IN_RANGE_M_S
    out = {}
    for farm in FARMS:
        value = float(fitted[farm]["cut_in_m_s"])
        out[farm] = value if low <= value <= high else GENERIC_CUT_IN_M_S
    return out


def cut_in_for(*, farm: str, suffix: str) -> float:
    """Return the cut-in used by the variant `suffix` ("", "lo", "hi", "g3") for `farm`."""
    shift = CUT_IN_VARIANTS[suffix]
    return GENERIC_CUT_IN_M_S if shift is None else cut_in_speeds()[farm] + shift


def random_like(*, flag: np.ndarray, group: np.ndarray, seed: int) -> np.ndarray:
    """Return a mask with the same count of True rows as `flag` in each `group`, chosen at random.

    The placebo for a speed mask: the same share of rows within each farm-and-month group, placed
    at random.

    Args:
        flag: Boolean mask to imitate.
        group: Label per row (a month, for one farm); the count is matched within each label.
        seed: Seeds the random choice.

    Returns:
        A boolean array of the same length.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(flag.size, dtype=bool)
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        count = int(flag[rows].sum())
        if count:
            out[rng.choice(rows, size=count, replace=False)] = True
    return out


def month_labels(*, valid_time: np.ndarray) -> np.ndarray:
    """Return the calendar month of each valid time as a label."""
    return valid_time.astype("datetime64[M]").astype(str)


def stats_path(*, kind: str, set_name: str, farm: str) -> Path:
    """Return the path of a per-case statistics file (`speed` or `era5`)."""
    return STATS_DIR / f"{kind}_{set_name}_{farm}.npy"


def below_cut_in(*, speed: np.ndarray, cut_in: float) -> np.ndarray:
    """Return the rows whose speed is below the cut-in speed (the mask under test)."""
    return speed < cut_in


def known_answer_test() -> dict[str, float]:
    """Check that the speed mask flags exactly the rows below cut-in in a synthetic series.

    The synthetic series has 20,000 half-hours with speeds drawn uniformly from 0 to 20 m/s. Rows
    below the cut-in get a noisy near-zero power (uniform 0 to 2% of capacity). A known 5% of the
    rows above the cut-in also get noisy near-zero values. The mask must flag every noisy row below
    the cut-in, flag no row above it, and so miss every injected noisy row above it (the speed mask
    cannot see them; this is the shape of experiment 210's finding).

    Returns:
        Recall below cut-in, false-flag share above cut-in, the share of injected noisy rows above
        cut-in that the mask caught, and the dropped share after zeroing.
    """
    rng = np.random.default_rng(PLACEBO_SEED)
    cut_in = 3.0
    speed = rng.uniform(0.0, 20.0, size=20_000)
    below = speed < cut_in
    injected = (~below) & (rng.random(speed.size) < 0.05)
    flagged = below_cut_in(speed=speed, cut_in=cut_in)
    target = np.where(
        below | injected, rng.uniform(0.0, 0.02, speed.size), np.clip((speed - cut_in) / 9, 0, 1)
    )
    zeroed = np.where(flagged, 0.0, target)
    return {
        "recall_below_cut_in": float((flagged & below).sum() / below.sum()),
        "false_flag_share_above": float((flagged & ~below).sum() / (~below).sum()),
        "injected_caught": float((flagged & injected).sum() / injected.sum()),
        "zeroed_target_max_below": float(zeroed[below].max()),
    }


def idle_mask_frame() -> pl.DataFrame:
    """Return experiment 210's mask rows with a `masked` flag, ready for a left join."""
    return c303.idle_mask().with_columns(masked=pl.lit(value=True))
