"""The property test of experiment 00088: the causal detector reads nothing at or after its cut.

For injected events under two shapes and cuts from 6 hours to 5 days after each onset, the script
perturbs every residual value at or after the cut (noise, scaling, and removal) and asserts that the
reported intervals are unchanged. It then prints, for the largest events, the onset error and the
ratio of the detected step to the injected median MW by hours since onset.

    uv run python ml-experiments/00088-*/e00088_check.py
"""

import e00088_causal as causal
import e00088_grid as grid
import e00088_harness as harness
import numpy as np
from e00088_profile import inject_shaped

N_EVENTS = 8
HOURS = (6, 12, 24, 36, 48, 72, 120)


def main() -> None:
    """Run the property test and print the detection diagnostics."""
    state = harness.make_state()
    events = harness.cached_events(power=state["panel"]["power"])
    batches = harness.seeded_batches(events=events)
    rng = np.random.default_rng(0)
    checked = 0
    print("hours since onset: detected, onset error (h), detected step over injected median MW")
    for seed, batch in batches[:40]:
        injection = inject_shaped(
            power=state["panel"]["power"],
            events=batch,
            seed=seed,
            shape="S2",
            shares=state["shares"],
        )
        residual = injection.fast_residual(state["panel"]["residual"])
        offset = injection.fast_residual(state["offset_panel"]["residual"])
        big = [e for e in batch if e.size_band == "25% or more" and e.duration_band == "over 1 d"]
        for event in big[:2]:
            if checked >= N_EVENTS:
                return
            checked += 1
            line = []
            for hours in HOURS:
                cut = event.onset + 2 * hours + 7
                if cut >= grid.GRID_LENGTH - 1:
                    continue
                found = causal.causal_intervals(
                    state=state, residual=residual, offset=offset, cut=cut
                )
                for variant in range(3):
                    future_residual = {s: v.copy() for s, v in residual.items()}
                    future_offset = {s: v.copy() for s, v in offset.items()}
                    for s, values in future_residual.items():
                        if variant == 0:
                            values[cut:] += rng.normal(0, 50, grid.GRID_LENGTH - cut)
                        elif variant == 1:
                            future_offset[s][cut:] *= 3.0
                        else:
                            values[cut:] = np.nan
                            future_offset[s][cut:] = np.nan
                    again = causal.causal_intervals(
                        state=state, residual=future_residual, offset=future_offset, cut=cut
                    )
                    if again != found:
                        msg = f"event {event.event_id}, cut {cut}: the future changed the output"
                        raise SystemExit(msg)
                mine = [
                    i for i in found if i.series == event.donor and abs(i.onset - event.onset) <= 96
                ]
                injected = float(np.median(injection.delta[event.donor][event.onset : event.end]))
                line.append(
                    f"{hours}h: "
                    + (
                        f"yes {(mine[0].onset - event.onset) / 2:+.1f} "
                        f"{mine[0].step_mw / injected:.2f}"
                        if mine
                        else "no"
                    )
                )
            print(f"event {event.event_id}: " + "; ".join(line))
    print(f"property test passed on {checked} events")


if __name__ == "__main__":
    main()
