"""Score round 2 (post hoc): v1 against v2 on two new draws, the controls, and the quoted reasons.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**Everything here is post hoc.** Round 1 (`score_labeller.py`) holds the planned contrasts; round 2
changed the tools and the prompt after seeing draws 01 and 02, so it is judged on draws 03 and 04
and every contrast carries the label `post hoc`. The script makes no model call: it rebuilds each
prompt and reads the answer from the cache.

**Arms, on draws 03 and 04:** tools only v1 (the fixed rule on the changepoint list), tools only v2
(the fixed rule on the local scan), tools plus model v1 and v2 (2 samples each), and the model on
weekly summaries. **Post hoc contrasts** (mean absolute capacity error, points; 97.5% bootstrap over
(event, sample) pairs): v2 against v1 for the tools-only rule, v2 against v1 for the model, the
model against tools only within each version, and the v2 model against the raw-summary arm.

**Controls.** Round 2's positive control plants 30% steps with no two in adjacent months at a
plant. Round 1's control is also rescored per run of adjacent months (three consecutive planted
months count as one loss), post hoc. The null record gives the false-alarm floor.

**Reasons.** Round 1's answers carry no reason, so the report lists the matching-candidate cases (a
down candidate within 14 days of a planted loss that the model left unmarked) with the candidate
lines and the answer. Round 2's answers carry a `note` on why there is no change, which is quoted.

Outputs under `<root>/tables_round2/`, including `capacity_series.npz` that the charts draw.
"""

import argparse
import json
import logging
import sys
from datetime import date
from typing import Final

import numpy as np
import polars as pl
from labeller_jobs import (
    CAUSAL_ISSUE_DAYS,
    CONTROL_RECORDS,
    HINDSIGHT_RECORDS,
    ROUND2_CAUSAL_RECORDS,
    ROUND2_CONTROL_RECORDS,
    LabellerContext,
    VersionType,
    run_causal_chain,
)
from labeller_paths import (
    TUNING_SEED,
    RecordIdType,
    calls_dir,
    claim_root,
    events_of,
    write_json_atomic,
)
from score_labeller import CachedOnlyModel
from studies.capacity_evidence import Evidence
from studies.capacity_labeller import (
    Answer,
    AnswerError,
    Prompt,
    cache_key,
    capacity_series,
    tools_only_answer,
    tools_only_v2_answer,
)
from studies.capacity_llm import (
    MODEL_ID,
    BudgetGuard,
    CallStore,
    ModelError,
    labelled_answer,
)
from studies.capacity_recovery import (
    CONTRAST_CONFIDENCE,
    FALSE_MOVEMENT_GATE,
    PlantedEvent,
    score_events,
)
from studies.capacity_sanity import apply_hindsight
from studies.labeller_scoring import (
    answers_to_scale,
    event_bootstrap_interval,
    false_alarms_per_plant,
    movement_by_window,
    movement_interval,
    score_draw,
    score_event,
    segments_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("score_round2")

NEW_DRAWS: Final[tuple[RecordIdType, ...]] = ("draw03", "draw04")
"""The two draws round 2 is judged on."""

OLD_DRAWS: Final[tuple[RecordIdType, ...]] = ("draw01", "draw02")
"""The draws that informed round 2."""

SAMPLES: Final[int] = 2
"""Samples per plant and record in the arms with the tools."""


def merge_adjacent_events(*, events: list[PlantedEvent]) -> list[PlantedEvent]:
    """Merge events at one plant that start the day after the previous one ends into one event.

    Args:
        events: Planted events (steps of one size).

    Returns:
        One event per run of adjacent months, with the run's length.
    """
    merged: list[PlantedEvent] = []
    for event in sorted(events, key=lambda e: (e.plant, e.first_day)):
        last = merged[-1] if merged else None
        if (
            last
            and last.plant == event.plant
            and last.first_day + last.month_days == event.first_day
        ):
            merged[-1] = PlantedEvent(
                last.plant,
                last.shape,
                last.size,
                last.first_day,
                last.month_days + event.month_days,
            )
        else:
            merged.append(event)
    return merged


def note_of(*, text: str) -> str:
    """Return the `note` a round-2 reply gave for no change, or an empty string."""
    start, stop = text.find("{"), text.rfind("}")
    try:
        payload = json.loads(text[start : stop + 1])
    except json.JSONDecodeError:
        return ""
    return str(payload.get("note", "")) if isinstance(payload, dict) else ""


def main() -> int:  # noqa: PLR0912, PLR0915
    """Score round 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    out = root / "tables_round2"
    out.mkdir(exist_ok=True)
    context = LabellerContext(root=root)
    plants = len(context.sites)
    dates = context.dates
    first_day, last_day = dates[0].astype(date), dates[-1].astype(date)
    model_id = "stub" if arguments.smoke else MODEL_ID
    store = CallStore(directory=calls_dir(root=root), log_path=root / "scoring_misses.jsonl")
    guard = BudgetGuard(ceiling_usd=1e9, max_calls=10**9, max_input_tokens=10**12)
    model = CachedOnlyModel(model_id=model_id)

    def answer_function(
        *,
        prompt: Prompt,
        sample_index: int,
        label: str,
        plant: str,
        first_day: date,
        last_day: date,
        evidence: Evidence,
    ) -> Answer:
        del evidence
        return labelled_answer(
            store=store,
            model=model,
            guard=guard,
            prompt=prompt,
            sample_index=sample_index,
            label=label,
            plant=plant,
            first_day=first_day,
            last_day=last_day,
        )

    answers: dict[tuple[str, VersionType, RecordIdType, int, int], Answer] = {}
    notes: list[dict[str, object]] = []
    missing: list[str] = []

    def collect(*, arm: str, version: VersionType, record_id: RecordIdType, samples: int) -> None:
        kind = "tools" if arm == "tools_llm" else "raw"
        for plant in range(plants):
            case = context.hindsight_case(
                record_id=record_id, plant=plant, kind=kind, version=version
            )
            for sample in range(samples):
                try:
                    answers[(arm, version, record_id, plant, sample)] = answer_function(
                        prompt=case.prompt,
                        sample_index=sample,
                        label="",
                        plant=context.sites[plant],
                        first_day=case.first_day,
                        last_day=case.last_day,
                        evidence=case.evidence,
                    )
                except ModelError, AnswerError:
                    missing.append(f"{arm}/{version}/{record_id}/{plant}/{sample}")
                    continue
                reply = store.get(
                    key=cache_key(model=model_id, prompt=case.prompt, sample_index=sample)
                )
                if (
                    version == "v2"
                    and answers[(arm, version, record_id, plant, sample)].no_change
                    and reply
                ):
                    notes.append(
                        {
                            "record": record_id,
                            "plant": context.sites[plant],
                            "sample": sample,
                            "note": note_of(text=reply.text),
                        }
                    )

    for record_id in (*NEW_DRAWS, "original"):
        collect(arm="tools_llm", version="v2", record_id=record_id, samples=SAMPLES)
    for record_id in ("original", *NEW_DRAWS):
        collect(arm="tools_llm", version="v1", record_id=record_id, samples=SAMPLES)
        collect(arm="raw_llm", version="v1", record_id=record_id, samples=1)
    for record_id in OLD_DRAWS:
        collect(arm="tools_llm", version="v1", record_id=record_id, samples=1)
    for record_id in ROUND2_CONTROL_RECORDS:
        collect(arm="tools_llm", version="v2", record_id=record_id, samples=1)
    for record_id in CONTROL_RECORDS:
        collect(arm="tools_llm", version="v1", record_id=record_id, samples=1)

    def tools_only(*, version: VersionType, record_id: RecordIdType) -> dict[int, Answer]:
        rule = tools_only_answer if version == "v1" else tools_only_v2_answer
        return {
            p: rule(
                evidence=context.hindsight_case(
                    record_id=record_id, plant=p, kind="tools"
                ).evidence,
                first_day=first_day,
                last_day=last_day,
            )
            for p in range(plants)
        }

    def llm(
        *, arm: str, version: VersionType, record_id: RecordIdType, sample: int, after: bool
    ) -> dict[int, Answer] | None:
        found = {
            p: answers[(arm, version, record_id, p, sample)]
            for p in range(plants)
            if (arm, version, record_id, p, sample) in answers
        }
        if len(found) != plants:
            return None
        if not after:
            return found
        return {
            p: apply_hindsight(
                answer=a,
                evidence=context.hindsight_case(
                    record_id=record_id, plant=p, kind="tools"
                ).evidence,
            ).answer
            for p, a in found.items()
        }

    def arm_answers(
        *, arm: str, record_id: RecordIdType, sample: int, after: bool
    ) -> dict[int, Answer] | None:
        name, _, version = arm.rpartition("_")
        if name == "tools_only":
            return tools_only(version=version, record_id=record_id)  # ty: ignore[invalid-argument-type]
        return llm(arm=name, version=version, record_id=record_id, sample=sample, after=after)  # ty: ignore[invalid-argument-type]

    arms = {
        "tools_only_v1": 1,
        "tools_only_v2": 1,
        "tools_llm_v1": SAMPLES,
        "tools_llm_v2": SAMPLES,
        "raw_llm_v1": 1,
    }
    original_record = context.record(record_id="original")
    errors: dict[tuple[str, int, bool], np.ndarray] = {}
    score_rows: list[pl.DataFrame] = []
    for after in (False, True):
        for arm, samples in arms.items():
            for sample in range(samples):
                per_draw = []
                for record_id in NEW_DRAWS:
                    planted = arm_answers(arm=arm, record_id=record_id, sample=sample, after=after)
                    original = arm_answers(
                        arm=arm, record_id="original", sample=sample, after=after
                    )
                    if planted is None or original is None:
                        per_draw = []
                        break
                    events = events_of(root=root, record_id=record_id)
                    capacity = score_events(
                        events=events,
                        planted=answers_to_scale(answers=planted, record=original_record),
                        original=answers_to_scale(answers=original, record=original_record),
                        days=len(dates),
                    )
                    detection = score_draw(
                        events=events, planted=planted, original=original, record=original_record
                    )
                    score_rows.append(
                        detection.with_columns(
                            capacity_error=pl.Series(capacity["error"]),
                            arm=pl.lit(arm),
                            sample=pl.lit(sample),
                            draw=pl.lit(record_id),
                            after_checks=pl.lit(after),
                        )
                    )
                    per_draw.append(capacity["error"].to_numpy())
                if per_draw:
                    errors[(arm, sample, after)] = np.concatenate(per_draw) * 100.0
    if score_rows:
        pl.concat(score_rows).write_parquet(out / "event_scores.parquet")
    np.savez_compressed(
        out / "errors.npz",
        months=np.array(
            [
                str(dates[e.first_day].astype("datetime64[M]"))
                for record_id in NEW_DRAWS
                for e in events_of(root=root, record_id=record_id)
            ]
        ),
        **{
            f"{arm}|{sample}|{int(after)}": values
            for (arm, sample, after), values in errors.items()
        },  # ty: ignore[invalid-argument-type]
    )

    def matrix(*, arm: str, after: bool) -> np.ndarray | None:
        samples = arms[arm]
        if any((arm, s, after) not in errors for s in range(samples)):
            return None
        return np.stack([errors[(arm, s, after)] for s in range(samples)], axis=1)

    contrast_rows = []
    pairs = [
        ("v2 tools only minus v1 tools only", "tools_only_v2", "tools_only_v1"),
        ("v2 tools+LLM minus v1 tools+LLM", "tools_llm_v2", "tools_llm_v1"),
        ("v2 tools+LLM minus v2 tools only", "tools_llm_v2", "tools_only_v2"),
        ("v1 tools+LLM minus v1 tools only", "tools_llm_v1", "tools_only_v1"),
        ("v2 tools+LLM minus LLM on raw summaries", "tools_llm_v2", "raw_llm_v1"),
    ]
    for after in (False, True):
        for name, left_arm, right_arm in pairs:
            left, right = matrix(arm=left_arm, after=after), matrix(arm=right_arm, after=after)
            if left is None or right is None:
                continue
            width = max(left.shape[1], right.shape[1])
            difference = np.repeat(left, width // left.shape[1], axis=1) - np.repeat(
                right, width // right.shape[1], axis=1
            )
            mean, lower, upper = event_bootstrap_interval(
                differences=difference, confidence=CONTRAST_CONFIDENCE, seed=TUNING_SEED
            )
            contrast_rows.append(
                {
                    "label": "post hoc",
                    "contrast": name,
                    "after_checks": after,
                    "events": difference.shape[0],
                    "mean_points": mean,
                    "lower": lower,
                    "upper": upper,
                }
            )
    pl.DataFrame(
        contrast_rows,
        schema={
            "label": pl.String,
            "contrast": pl.String,
            "after_checks": pl.Boolean,
            "events": pl.Int64,
            "mean_points": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
        },
    ).write_parquet(out / "contrasts.parquet")

    # Controls.
    control_rows = []
    old_control = events_of(root=root, record_id="control")
    merged = merge_adjacent_events(events=old_control)
    for arm in ("tools_only_v1", "tools_only_v2", "tools_llm_v1", "tools_llm_v2"):
        for record_id, events, label in (
            ("control", old_control, "round 1 control, per event"),
            ("control", merged, "round 1 control, per run of adjacent months (post hoc)"),
            (
                "control2",
                events_of(root=root, record_id="control2"),
                "round 2 control, non-adjacent",
            ),
        ):
            planted = arm_answers(arm=arm, record_id=record_id, sample=0, after=False)
            original = arm_answers(arm=arm, record_id="original", sample=0, after=False)
            if planted is None or original is None:
                continue
            table = score_draw(
                events=events, planted=planted, original=original, record=original_record
            )
            control_rows.append(
                {
                    "control": label,
                    "arm": arm,
                    "events": table.height,
                    "hit_rate": float(table["hit"].cast(pl.Float64).mean()),  # ty: ignore[invalid-argument-type]
                    "false_alarms": false_alarms_per_plant(scores=table),
                }
            )
        null = arm_answers(arm=arm, record_id="null", sample=0, after=False)
        if null is not None:
            steps = sum(
                1
                for p in range(plants)
                for s in segments_of(knots=null[p].knots, dates=dates)
                if abs(s.change) >= 0.03
            )
            control_rows.append(
                {
                    "control": "null record",
                    "arm": arm,
                    "events": 0,
                    "hit_rate": None,
                    "false_alarms": steps,
                }
            )
    pl.DataFrame(
        control_rows,
        schema={
            "control": pl.String,
            "arm": pl.String,
            "events": pl.Int64,
            "hit_rate": pl.Float64,
            "false_alarms": pl.Int64,
        },
    ).write_parquet(out / "controls.parquet")

    # False movement on the 23 quiet plant-months, original record.
    movement = []
    windows = context.first_events.false_movement_windows
    for arm, samples in arms.items():
        per_sample = []
        for sample in range(samples):
            found = arm_answers(arm=arm, record_id="original", sample=sample, after=False)
            if found is not None:
                per_sample.append(
                    movement_by_window(
                        windows=windows,
                        scale_mw=answers_to_scale(answers=found, record=original_record).scale_mw,
                    )
                )
        if per_sample:
            share, lower, upper = movement_interval(
                per_window=np.mean(per_sample, axis=0),
                confidence=CONTRAST_CONFIDENCE,
                seed=TUNING_SEED,
            )
            movement.append(
                {
                    "arm": arm,
                    "false_movement": share,
                    "lower": lower,
                    "upper": upper,
                    "gate": FALSE_MOVEMENT_GATE,
                    "within_gate": share <= FALSE_MOVEMENT_GATE,
                }
            )
    pl.DataFrame(
        movement,
        schema={
            "arm": pl.String,
            "false_movement": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "gate": pl.Float64,
            "within_gate": pl.Boolean,
        },
    ).write_parquet(out / "false_movement.parquet")

    # Round 1's matching-candidate cases, with the candidate lines and the answer.
    cases = []
    coverage: list[dict[str, object]] = []
    for record_id in OLD_DRAWS:
        for event in events_of(root=root, record_id=record_id):
            case = context.hindsight_case(record_id=record_id, plant=event.plant, kind="tools")
            near = [
                c
                for c in case.evidence.changepoints
                if c.direction == "down" and abs(c.day_index - event.first_day) <= 14
            ]
            answer = (
                answers.get(("tools_llm", "v1", record_id, event.plant, 0))
                if (record_id in HINDSIGHT_RECORDS)
                else None
            )
            if answer is None:
                continue
            marked = [
                s
                for s in segments_of(knots=answer.knots, dates=dates)
                if abs(s.change) > 0.03
                and min(abs(s.start - event.first_day), abs(s.end - event.first_day)) <= 14
            ]
            end_near = [
                c
                for c in case.evidence.changepoints
                if c.direction == "up"
                and abs(c.day_index - (event.first_day + event.month_days)) <= 14
            ]
            tools_hit = score_event(
                event=event,
                segments=segments_of(
                    knots=tools_only(version="v1", record_id=record_id)[event.plant].knots,
                    dates=dates,
                ),
            ).hit
            coverage.append(
                {
                    "events": 1,
                    "down_candidate_near_start": bool(near),
                    "up_candidate_near_end": bool(end_near),
                    "model_marked": bool(marked),
                    "tools_only_hit": tools_hit,
                }
            )
            if near:
                cases.append(
                    {
                        "record": record_id,
                        "plant": context.sites[event.plant],
                        "shape": event.shape,
                        "size": round(event.size, 2),
                        "candidates_near_start": "; ".join(
                            f"{c.day} {c.size:+.0%} p={c.p_value:.3f}" for c in near
                        ),
                        "model_marked": bool(marked),
                        "tools_only_hit": tools_hit,
                        "answer": "no change" if answer.no_change else f"{len(answer.knots)} knots",
                    }
                )
    pl.DataFrame(
        cases,
        schema={
            "record": pl.String,
            "plant": pl.String,
            "shape": pl.String,
            "size": pl.Float64,
            "candidates_near_start": pl.String,
            "model_marked": pl.Boolean,
            "tools_only_hit": pl.Boolean,
            "answer": pl.String,
        },
    ).write_parquet(out / "matching_candidate_cases.parquet")
    pl.DataFrame(coverage).sum().write_parquet(out / "candidate_coverage.parquet")
    pl.DataFrame(
        notes,
        schema={"record": pl.String, "plant": pl.String, "sample": pl.Int64, "note": pl.String},
    ).write_parquet(out / "round2_no_change_notes.parquet")

    # Capacity series for the charts: v1 and v2 tools-only and model (sample 0), causal v2.
    series: dict[str, np.ndarray] = {}
    for record_id in ("original", *NEW_DRAWS):
        for arm, tag in (
            ("tools_only_v1", "tools_only_v1"),
            ("tools_only_v2", "tools_only_v2"),
            ("tools_llm_v1", "llm_v1"),
            ("tools_llm_v2", "llm_v2"),
        ):
            found = arm_answers(arm=arm, record_id=record_id, sample=0, after=True)
            if found is not None:
                series[f"{record_id}/{tag}"] = answers_to_scale(
                    answers=found, record=original_record
                ).scale_mw
    for record_id in ROUND2_CAUSAL_RECORDS:
        stitched = np.full((plants, len(dates)), np.nan)
        complete = True
        for plant in range(plants):
            try:
                steps = run_causal_chain(
                    context=context,
                    record_id=record_id,
                    plant=plant,
                    answer_function=answer_function,
                    version="v2",
                )
            except ModelError, AnswerError:
                complete = False
                continue
            from itertools import pairwise

            bounds = [context.day_index(day=d) for d in (*CAUSAL_ISSUE_DAYS, date(2026, 7, 1))]
            for step, (a, b) in zip(steps, pairwise(bounds), strict=True):
                curve = capacity_series(
                    knots=step.outcome.answer.knots,
                    dates=dates,
                    static_p99_mw=float(original_record.static_p99_mw[plant]),
                )
                stitched[plant, a + 1 : b + 1] = curve[a + 1 : b + 1]
        if complete:
            series[f"{record_id}/llm_causal_v2"] = stitched
    np.savez_compressed(out / "capacity_series.npz", **series)  # ty: ignore[invalid-argument-type]
    write_json_atomic(path=out / "missing.json", payload={"missing_answers": missing})
    _LOG.info("scored round 2; %d answers missing", len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
