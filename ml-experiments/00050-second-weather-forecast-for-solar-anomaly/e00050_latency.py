"""Record when Dynamical.org's GEFS store holds the 00 UTC run, and decide which run the arms use.

Production's forecast origin is the 00 UTC run plus 9 hours (`PUBLICATION_DELAY`). The arms may use
the same day's 00 UTC GEFS run only if the store reliably holds it by 09:00 UTC. Otherwise they use
the previous day's run, whose leads are 24 hours longer, and the lead bands stay measured from the
forecast origin.

- `observe` opens the store, reads the newest `init_time` in it, and appends the observation (UTC
  clock time, newest `init_time`, and whether today's 00 UTC run is present) to
  `work/results/latency_observations.json`. Run it at two or three clock times on different days.
- `decide` reads the observations. The same-day run is chosen (offset 0) only when at least 3
  observations were taken between 07:00 and 09:00 UTC on at least 3 different dates and every one
  of them found the same day's run. Otherwise the previous day's run is chosen (offset 1). The
  decision is written to `work/results/latency_decision.json`, which every later script requires.

    uv run python ml-experiments/00050-*/e00050_latency.py observe
    uv run python ml-experiments/00050-*/e00050_latency.py decide
"""

import argparse
import json
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Final

import dynamical_catalog
import e00050_common as common
import numpy as np

OBSERVATIONS_PATH: Final[Path] = common.RESULTS_DIR / "latency_observations.json"
WINDOW_START: Final[time] = time(7, 0)
WINDOW_END: Final[time] = time(9, 0)
MIN_OBSERVATIONS: Final[int] = 3


def observe() -> dict[str, str | bool]:
    """Append one observation of the store's newest GEFS run and return it."""
    dataset = dynamical_catalog.open(common.GEFS_STORE)
    newest = np.datetime64(dataset["init_time"].values.max(), "s").astype(datetime)
    now = datetime.now(UTC)
    newest_utc = newest.replace(tzinfo=UTC)
    observation = {
        "observed_at_utc": now.isoformat(timespec="seconds"),
        "newest_init_time": newest_utc.isoformat(timespec="seconds"),
        "todays_run_present": bool(
            newest_utc >= now.replace(hour=0, minute=0, second=0, microsecond=0)
        ),
    }
    previous = json.loads(OBSERVATIONS_PATH.read_text()) if OBSERVATIONS_PATH.exists() else []
    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OBSERVATIONS_PATH.write_text(json.dumps([*previous, observation], indent=2) + "\n")
    print(observation)
    return observation


def choose_offset(*, observations: list[dict[str, str | bool]]) -> tuple[int, str]:
    """Return the run offset in days (0 same day, 1 previous day) and the reason for it.

    Args:
        observations: The recorded observations.

    Returns:
        The offset and a sentence stating the evidence.
    """
    window = [
        o
        for o in observations
        if WINDOW_START <= datetime.fromisoformat(str(o["observed_at_utc"])).time() <= WINDOW_END
    ]
    dates = {datetime.fromisoformat(str(o["observed_at_utc"])).date() for o in window}
    present = sum(bool(o["todays_run_present"]) for o in window)
    if (
        len(window) >= MIN_OBSERVATIONS
        and len(dates) >= MIN_OBSERVATIONS
        and present == len(window)
    ):
        return (
            0,
            f"today's run was present in all {len(window)} observations from 07:00 to 09:00 UTC",
        )
    return (
        1,
        (
            f"today's run was present in {present} of {len(window)} observations from 07:00 to "
            f"09:00 UTC on {len(dates)} dates, short of the {MIN_OBSERVATIONS} needed to rely on it"
        ),
    )


def decide() -> None:
    """Write the decision from the recorded observations."""
    observations = json.loads(OBSERVATIONS_PATH.read_text()) if OBSERVATIONS_PATH.exists() else []
    offset, reason = choose_offset(observations=observations)
    common.LATENCY_PATH.write_text(
        json.dumps(
            {"run_offset_days": offset, "reason": reason, "observations": observations}, indent=2
        )
        + "\n"
    )
    print(f"run_offset_days = {offset}: {reason}")


def main() -> None:
    """Run the requested step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("observe", "decide"))
    args = parser.parse_args()
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "observe":
        observe()
    else:
        decide()


if __name__ == "__main__":
    main()
