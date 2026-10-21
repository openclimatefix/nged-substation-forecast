"""Choose the deviation penalties of experiment 158 on the tuning months.

The arms `ps<h>_<o>` pool the heating curve at penalty `h` and every other shared family at `o`,
over the grid `common.PENALTY_GRID`. The chosen pair has the lowest all-lead quantile score at the
five decision substations for the 3-month history, among the pairs whose 24-month score is at most
`NON_INFERIORITY_POINTS` above K0's on the same months. A tie within `TIE_POINTS` goes to the
smaller `h + o` (less pooling). PH takes the chosen `h`. The pair with the lowest 24-month score on
its own is stored too, as a labelled variant. The choice is stored in `work/tuned.json`.

    uv run python ml-experiments/00158-*/e00158_tune.py
"""

import json
from itertools import product
from typing import Final

import e00158_common as common
import e00158_report as report
import polars as pl

TIE_POINTS: Final[float] = 0.005
NON_INFERIORITY_POINTS: Final[float] = 0.02
TUNED_PATH: Final = common.WORK_DIR / "tuned.json"


def pair_name(*, heating: float, other: float) -> str:
    """Return the arm name of a penalty pair."""
    return f"ps{heating:g}_{other:g}"


def scores(*, history: str, configs: list[str]) -> dict[str, float]:
    """Return each arm's all-lead score at the decision substations in the tuning months."""
    frames = {}
    for config in configs:
        frame = report.load(arm=f"{history}_{config}", period="tuning")
        if frame is not None:
            frames[config] = frame
    shared = report.common_rows(frames=frames)
    out = {}
    for config, frame in shared.items():
        losses = report.with_losses(frame=frame).filter(
            pl.col("site").is_in(list(common.DECISION_SITES))
        )
        out[config] = float(losses["qs"].to_numpy().mean())
    return out


def choose(*, scores_by_pair: dict[tuple[float, float], float]) -> tuple[float, float]:
    """Return the pair with the lowest score, preferring the smaller `h + o` within the tie."""
    best = min(scores_by_pair.values())
    tied = [pair for pair, score in scores_by_pair.items() if score <= best + TIE_POINTS]
    return min(tied, key=lambda pair: (sum(pair), pair))


def main() -> None:
    """Tune the penalty pair and store the choice."""
    pairs = [(h, o) for h, o in product(common.PENALTY_GRID, common.PENALTY_GRID)]
    names = {pair: pair_name(heating=pair[0], other=pair[1]) for pair in pairs}
    full = scores(history="full", configs=["k0", *names.values()])
    short = scores(history="3m", configs=["k0", *names.values()])
    print("full", json.dumps(full, indent=1))
    print("3m", json.dumps(short, indent=1))
    allowed = {
        pair: short[name]
        for pair, name in names.items()
        if full[name] <= full["k0"] + NON_INFERIORITY_POINTS
    }
    if not allowed:
        msg = "no penalty pair is non-inferior at 24 months on the tuning months"
        raise ValueError(msg)
    heating, other = choose(scores_by_pair=allowed)
    own = choose(scores_by_pair={pair: full[name] for pair, name in names.items()})
    stored = {
        "chosen": {"heating": heating, "other": other},
        "own_24_month": {"heating": own[0], "other": own[1]},
        "full": {"ph": f"ph{heating:g}", "ps": names[(heating, other)]},
        "3m": {"ph": f"ph{heating:g}", "ps": names[(heating, other)]},
        "own_24_month_arm": names[own],
        "scores": {"full": full, "3m": short},
    }
    print(json.dumps({k: v for k, v in stored.items() if k != "scores"}, indent=1))
    common.write_json_atomic(path=TUNED_PATH, content=stored)


if __name__ == "__main__":
    main()
