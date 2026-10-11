"""Draw the page's figures of each estimator's capacity over a plant's measured power.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
Reads `tables/capacity_series.npz` (written by `score_labeller.py`) and the first study's record,
and writes one SVG and one PNG per figure under `<root>/figures/`. **The stub pipeline writes the
same figures under `_smoke/`, so the layout can be checked before any real run.**

**Four figures, each with seven panels on the same axes** (see `studies.capacity_estimate_charts`):

1. a plant with a planted step loss (a fresh draw);
2. a plant with a planted ramp loss (a fresh draw);
3. a plant in a quiet stretch of the real record, which should show no change;
4. a plant in its real low-output month, with no plant letter or month in the figure.

**No figure carries a calendar date,** and the figure that shows a real low-output month names no
plant (the study's `dataviz` anonymisation rule).

Run it with `uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py`; `--smoke`
reads and writes under `_smoke/`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import vl_convert as vlc
from estimate_part1 import bundle_of, estimator_function, static_scale
from labeller_paths import (
    claim_root,
    events_of,
    load_first_events,
    load_first_record,
    record_for,
    tables_dir,
)
from studies.capacity_estimate_charts import (
    ESTIMATOR_PANELS,
    ROUND2_PANELS,
    EstimatorPanel,
    estimate_figure,
)
from studies.capacity_recovery import PlantedEvent, calendar_month_spans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("llm_capacity_labeller_charts")

DAYS_BEFORE: Final[int] = 40
"""Days shown before the stretch a figure is about."""

DAYS_AFTER: Final[int] = 60
"""Days shown after it, which is the first study's scoring footprint."""

SUBTITLE: Final[tuple[str, ...]] = (
    (
        "Grey dots: the highest hourly power of each day, in megawatts (MW). Orange line: the "
        "estimator's capacity in MW, which should sit near the top of the dots and drop where "
        "capacity is lost."
    ),
    "Every panel uses one vertical range. Day 1 is the first day shown; no dates are given.",
)
"""The subtitle lines every figure carries."""


def _window(*, first: int, length: int, days: int) -> tuple[int, int]:
    """Return a window around a stretch of days, kept inside the record."""
    return max(first - DAYS_BEFORE, 0), min(first + length + DAYS_AFTER, days)


def _capacities(
    *,
    series: dict[str, np.ndarray],
    record_id: str,
    plant: int,
    panel_set: tuple[EstimatorPanel, ...] = ESTIMATOR_PANELS,
) -> dict[str, np.ndarray | None]:
    """Return each panel's capacity series for one plant, `None` where the store has none."""
    return {
        panel.key: series[f"{record_id}/{panel.key}"][plant]
        if f"{record_id}/{panel.key}" in series
        else None
        for panel in panel_set
    }


def _first_event(
    *, events: list[PlantedEvent], shape: str, avoid_plant: int
) -> PlantedEvent | None:
    """Return the first event of a shape at a plant other than `avoid_plant`, or `None`."""
    return next((e for e in events if e.shape == shape and e.plant != avoid_plant), None)


def _write(*, figure: object, path: Path) -> None:
    """Write a figure as SVG and PNG."""
    spec = figure.to_dict()  # ty: ignore[unresolved-attribute]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".svg").write_text(vlc.vegalite_to_svg(spec))
    path.with_suffix(".png").write_bytes(vlc.vegalite_to_png(spec, scale=1.5))
    _LOG.info("wrote %s", path.with_suffix(".svg"))


def round2_series(*, root: Path, record_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return round 2's capacity series, with the static power, HYST2, and HM2 added.

    Args:
        root: The output root.
        record_ids: The records whose series are wanted.

    Returns:
        The series by `<record>/<key>`, shape (plants, days).
    """
    with np.load(root / "tables_round2" / "capacity_series.npz") as stored:
        series = {key: stored[key] for key in stored.files}
    chosen = json.loads((root / "part1" / "chosen.json").read_text())
    original = load_first_record()
    _, kf_reference = bundle_of(record=original, kf_reference=None)
    for record_id in record_ids:
        bundle, _ = bundle_of(
            record=record_for(root=root, record_id=record_id),  # ty: ignore[invalid-argument-type]
            kf_reference=kf_reference,
        )
        series[f"{record_id}/S0"] = static_scale(record=bundle.record).scale_mw
        for name in ("HYST2", "HM2"):
            function = estimator_function(name=name, params=chosen[name])
            series[f"{record_id}/{name}"] = function(bundle).scale_mw
    return series


def main() -> int:
    """Draw the four figures of round 1, or of round 2 with `--round 2`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    second = arguments.round == 2
    draws = ("draw03", "draw04") if second else ("draw01", "draw02")
    panel_set = ROUND2_PANELS if second else ESTIMATOR_PANELS
    prefix = "estimators2" if second else "estimators"
    note = "Round 2 (post hoc). " if second else ""
    if second:
        series = round2_series(root=root, record_ids=("original", *draws))
    else:
        with np.load(tables_dir(root=root) / "capacity_series.npz") as stored:
            series = {key: stored[key] for key in stored.files}
    first = load_first_events()
    original = load_first_record()
    days = len(original.dates)
    figures = root / "figures"
    for number, shape in (("1", "step"), ("2", "ramp")):
        for record_id in draws:
            events = events_of(root=root, record_id=record_id)
            event = _first_event(events=events, shape=shape, avoid_plant=first.anm_plant)
            if event is not None:
                break
        if event is None:
            _LOG.warning("no planted %s loss to draw", shape)
            continue
        record = record_for(root=root, record_id=record_id)
        figure = estimate_figure(
            record=record,
            plant=event.plant,
            window=_window(first=event.first_day, length=event.month_days, days=days),
            capacities=_capacities(
                series=series, record_id=record_id, plant=event.plant, panel_set=panel_set
            ),
            events=events,
            number=number,
            title=(
                f"{note}Each estimator on a planted {shape} loss of {event.size:.0%} "
                f"at plant {record.sites[event.plant]}"
            ),
            subtitle=(*SUBTITLE, "Blue shading: the month in which the loss was planted."),
            panel_set=panel_set,
        )
        _write(figure=figure, path=figures / f"{prefix}_planted_{shape}")
    plant, quiet_first, length = first.false_movement_windows[0]
    figure = estimate_figure(
        record=original,
        plant=plant,
        window=_window(first=quiet_first, length=length, days=days),
        capacities=_capacities(
            series=series, record_id="original", plant=plant, panel_set=panel_set
        ),
        events=[],
        number="3",
        title=(
            f"{note}Each estimator on a quiet stretch of the real record "
            f"at plant {original.sites[plant]}"
        ),
        subtitle=(
            *SUBTITLE,
            "No loss was planted; the capacity should stay near the top of the dots.",
        ),
        panel_set=panel_set,
    )
    _write(figure=figure, path=figures / f"{prefix}_quiet")
    month = first.low_months[-1]
    start, month_days = calendar_month_spans(record=original)[month]
    figure = estimate_figure(
        record=original,
        plant=first.low_plant,
        window=_window(first=start, length=month_days, days=days),
        capacities=_capacities(
            series=series, record_id="original", plant=first.low_plant, panel_set=panel_set
        ),
        events=[],
        number="4",
        title=f"{note}Each estimator on a real low-output month at one plant",
        subtitle=(*SUBTITLE, "The plant and the date are not given."),
        panel_set=panel_set,
    )
    _write(figure=figure, path=figures / f"{prefix}_real_low_month")
    return 0


if __name__ == "__main__":
    sys.exit(main())
