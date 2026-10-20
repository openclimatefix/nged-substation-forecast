"""Print the substation cards of experiment 123 as markdown tables, from `numbers_<stage>.json`.

uv run python ml-experiments/00123-*/e00123_cards.py headline
"""

import argparse
import json

import e00123_common as common

PROFILE_FIRST_MINUTE_LABEL = {30: "00:30"}


def clock(*, minute: int) -> str:
    """Return a start minute of day as `HH:MM`."""
    return f"{minute // 60:02d}:{minute % 60:02d}"


def main() -> None:
    """Print one row per fitted series: the numbers a card shows, and a second table for timing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["headline", "all"])
    stage = parser.parse_args().stage
    numbers = json.loads((common.WORK_DIR / f"numbers_{stage}.json").read_text())
    cross = {(r["site"], r["arm"]): r for r in numbers["cross_checks"]}
    raw = {r["site"]: r for r in numbers["raw"]}
    starts = {r["site"]: r for r in numbers["start_times"]}
    jumps = {r["site"]: r for r in numbers["bst_jump"]}
    timing = {r["site"]: r for r in numbers["timing"]}
    print(
        "| Series | Step at 3 C, points (90% interval) | Step, MW | Slope, points per degree | "
        "Cold-dependent part, points | Flat part, points | "
        "Summer reference, points | Winters |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in numbers["unpooled"]:
        site = r["site"]
        print(
            f"| {site} | {r['step']:.1f} ({r['step_low']:.1f} to {r['step_high']:.1f}) | "
            f"{r['step_mw']:.2f} | "
            f"{r['slope']:.2f} ({r['slope_low']:.2f} to {r['slope_high']:.2f}) | "
            f"{r['slope'] * common.REFERENCE_COLD_C:.1f} | {r['flat']:.1f} | "
            f"{raw[site]['d_summer']:.1f} | {r['winters'].replace(',', ', ')} |"
        )
    print()
    print(
        "| Series | Start of the step (half-hour starting) | Cold BST nights | "
        "Pre-registered label | Post hoc jump shift on BST nights, half-hours | R / full step "
        "(baseline model) |"
    )
    print("|---|---|---|---|---|---|")
    for r in numbers["unpooled"]:
        site = r["site"]
        start = starts[site].get("start_half_hour_start_minute")
        ratio = cross.get((site, "b0"))
        shift = jumps[site].get("shift_half_hours", "cannot tell")
        print(
            f"| {site} | {clock(minute=start) if start is not None else 'cannot tell'} | "
            f"{timing[site]['cold_bst_nights']} | {timing[site]['label']} | {shift} | "
            f"{f'{ratio["ratio_to_d"]:.2f}' if ratio else 'n/a'} |"
        )


if __name__ == "__main__":
    main()
