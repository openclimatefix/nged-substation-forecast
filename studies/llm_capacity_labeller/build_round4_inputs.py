"""Build round 4's eight further fresh draws and the mid-difficulty grid (post hoc).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**Round 4 is post hoc, and its design was committed before any round 4 call.** It has two parts:

- **Draws 07 to 14** (seeds 11447 to 11454): eight further draws of one-month planted losses, made
  with round 2's procedure. The model arm (tools plus model, version 2, 2 samples, hindsight only)
  runs on them with the amended step check frozen exactly as pre-registered for round 3, and the
  headline pools them with round 3's draws 05 and 06. Round 3 is reported alone beside the pooled
  figure.
- **The mid-difficulty grid**: two records each of a 30% loss lasting 90 days (`grid01`, `grid02`)
  and of a 60% loss lasting 30 days (`grid03`, `grid04`), at every plant, starting on a random day,
  built like the easy case. The easy case (60% for 90 days) changed size and duration together
  against the 30% one-month control, so the grid separates the two.

Writes `<root>/built/round4.json`. Run it with
`uv run python studies/llm_capacity_labeller/build_round4_inputs.py`.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
from labeller_paths import (
    GRID_DESIGNS,
    GRID_SEEDS,
    ROUND4_DRAW_SEEDS,
    built_dir,
    claim_root,
    event_to_json,
    load_first_events,
    load_first_record,
    write_json_atomic,
)
from studies.capacity_recovery import PlantedEvent, draw_events
from studies.commissioning import SETTLED_OUTPUT_FROM
from studies.guards import refuse_to_overwrite
from studies.labeller_scoring import draw_easy_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_round4_inputs")


def main() -> int:
    """Build the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    target = built_dir(root=root) / "round4.json"
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
        for seed in ROUND4_DRAW_SEEDS
    ]
    barred = {first.low_plant: first.low_months}
    grid: list[list[PlantedEvent]] = []
    for design_index, (size, length) in enumerate(GRID_DESIGNS):
        avoid: dict[int, list[tuple[int, int]]] = {}
        for seed in GRID_SEEDS[2 * design_index : 2 * design_index + 2]:
            events = draw_easy_events(
                record=record,
                size=size,
                length=length,
                seed=seed,
                avoid=avoid,
                barred_months=barred,
                earliest_first_day=settled,
            )
            grid.append(events)
            for event in events:
                avoid.setdefault(event.plant, []).append(
                    (event.first_day, event.first_day + event.month_days - 1)
                )
    write_json_atomic(
        path=target,
        payload={
            "draw_seeds": list(ROUND4_DRAW_SEEDS),
            "grid_seeds": list(GRID_SEEDS),
            "grid_designs": [list(design) for design in GRID_DESIGNS],
            "draws": [[event_to_json(event=e) for e in draw] for draw in draws],
            "grid": [[event_to_json(event=e) for e in events] for events in grid],
        },
    )
    _LOG.info(
        "wrote %s: draws %s, grid %s", target, [len(d) for d in draws], [len(g) for g in grid]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
