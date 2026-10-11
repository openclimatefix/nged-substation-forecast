"""Build round 2's two new draws of planted losses and its non-adjacent positive control.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**Round 2 is post hoc.** Round 1's tools and prompt were found weak on draws 01 and 02, and round
2's changes (a local scan and a softer prompt) were informed by those draws, so round 2 is judged on
two new draws (seeds 11443 and 11444) that nobody tuned on. The positive control plants steps of
30% so that no two steps at one plant fall in consecutive months.

Writes `<root>/built/round2.json` (the draws and the control). Run it with
`uv run python studies/llm_capacity_labeller/build_round2_inputs.py`.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
from labeller_paths import (
    CONTROL2_SEED,
    CONTROL_SIZE,
    ROUND2_DRAW_SEEDS,
    built_dir,
    claim_root,
    event_to_json,
    load_first_events,
    load_first_record,
    write_json_atomic,
)
from studies.capacity_recovery import draw_events
from studies.commissioning import SETTLED_OUTPUT_FROM
from studies.guards import refuse_to_overwrite
from studies.labeller_scoring import draw_nonadjacent_control_steps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_round2_inputs")

CONTROL_MONTHS_PER_PLANT: Final[int] = 3
"""How many steps the control plants at each plant (the first study's number)."""


def main() -> int:
    """Build the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    target = built_dir(root=root) / "round2.json"
    refuse_to_overwrite(paths=[target])
    record = load_first_record()
    first = load_first_events()
    settled = {
        record.sites.index(site): int((np.datetime64(when.date()) - record.dates[0]).astype(int))
        for site, when in SETTLED_OUTPUT_FROM.items()
    }
    draws = [
        draw_events(
            record=record,
            seed=seed,
            reserved=first.reserved,
            earliest_first_day=settled,
            may_have_fewer=list(settled),
        )
        for seed in ROUND2_DRAW_SEEDS
    ]
    control = draw_nonadjacent_control_steps(
        record=record,
        months_per_plant=CONTROL_MONTHS_PER_PLANT,
        size=CONTROL_SIZE,
        scored_months=first.scored_months,
        barred={first.low_plant: first.low_months},
        seed=CONTROL2_SEED,
    )
    write_json_atomic(
        path=target,
        payload={
            "seeds": list(ROUND2_DRAW_SEEDS),
            "control_seed": CONTROL2_SEED,
            "draws": [[event_to_json(event=e) for e in draw] for draw in draws],
            "control2": [event_to_json(event=e) for e in control],
        },
    )
    _LOG.info(
        "wrote %s: %s planted events, %d control steps",
        target,
        [len(d) for d in draws],
        len(control),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
