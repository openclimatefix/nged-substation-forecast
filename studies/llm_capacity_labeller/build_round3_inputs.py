"""Build round 3's fresh draws (05 and 06, pre-registered) and the easy-case records (post hoc).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**Round 3's draws are the only evaluation of the pre-registered amended step check**, which
accepts a knot pair when the local scan or the changepoint list has a candidate within 14 days (see
`studies.capacity_sanity.check_steps`). The amended check, its test, and this design were committed
before any round 3 call was made. Round 3 keeps the round 2 tools, prompt, and samples frozen.

**Outputs under `<root>/built/round3.json`:**

- `draws`: two draws of one-month planted losses (seeds 11445 and 11446), with round 2's procedure.
- `easy`: two records that each plant one 60% step lasting 90 days at every plant, starting on a
  random day (so the loss is not calendar-aligned), in windows that do not overlap between the two
  records and avoid each plant's real low months. The easy case asks whether the model follows a
  clear signal at all.

Run it with `uv run python studies/llm_capacity_labeller/build_round3_inputs.py`.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
from labeller_paths import (
    EASY_DAYS,
    EASY_SEEDS,
    EASY_SIZE,
    ROUND3_DRAW_SEEDS,
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
from studies.labeller_scoring import draw_easy_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_round3_inputs")


def main() -> int:
    """Build the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    target = built_dir(root=root) / "round3.json"
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
        for seed in ROUND3_DRAW_SEEDS
    ]
    barred = {first.low_plant: first.low_months}
    easy: list[list] = []
    avoid: dict[int, list[tuple[int, int]]] = {}
    for seed in EASY_SEEDS:
        events = draw_easy_events(
            record=record,
            size=EASY_SIZE,
            length=EASY_DAYS,
            seed=seed,
            avoid=avoid,
            barred_months=barred,
            earliest_first_day=settled,
        )
        easy.append(events)
        for event in events:
            avoid.setdefault(event.plant, []).append(
                (event.first_day, event.first_day + event.month_days - 1)
            )
    write_json_atomic(
        path=target,
        payload={
            "draw_seeds": list(ROUND3_DRAW_SEEDS),
            "easy_seeds": list(EASY_SEEDS),
            "draws": [[event_to_json(event=e) for e in draw] for draw in draws],
            "easy": [[event_to_json(event=e) for e in events] for events in easy],
        },
    )
    _LOG.info(
        "wrote %s: draws %s, easy %s", target, [len(d) for d in draws], [len(e) for e in easy]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
