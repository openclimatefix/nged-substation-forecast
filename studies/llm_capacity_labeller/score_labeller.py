"""Score the labeller's cached answers: planned contrasts, controls, and exploratory checks.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**This script makes no model call.** It rebuilds every prompt (`labeller_jobs`) and reads each
answer back from the cache, so a missing answer (a run stopped by the budget guard) is counted and
its events are left out of the contrasts that need it.

**Planned contrasts (written in the plan before any result):**

- C1: tools plus model against tools only, mean absolute capacity error on the two fresh draws, in
  points of capacity, sample `i` on the planted record paired with sample `i` on the original.
- C2: tools plus model against the model on raw weekly summaries.
- C3: the share of days on the 23 quiet plant-months that the capacity moves more than 5% from its
  trailing median, against the 10% gate, with HYST2 as the reference.

C1 and C2 use a 97.5% bootstrap over (event, sample) pairs. The primary reading scores the answers
before the hard checks; the secondary reading scores them after. The positive control must be found
at 80% or more for C1 to C3 to count as tested.

Outputs under `<root>/tables/`: `contrasts.parquet`, `event_scores.parquet`, `controls.parquet`,
`false_movement.parquet`, `agreement.parquet`, `real_low_months.parquet`, `causal.parquet`,
`anm_spot_check.parquet`, `noise_floor.json`, and `capacity_series.npz` (what the charts draw).
"""

import argparse
import logging
import sys
from datetime import date
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl
from labeller_jobs import (
    CAUSAL_ISSUE_DAYS,
    CAUSAL_RECORDS,
    CONTROL_RECORDS,
    HINDSIGHT_RECORDS,
    TOOLS_SAMPLES,
    HindsightJob,
    LabellerContext,
    run_causal_chain,
    run_hindsight_job,
)
from labeller_paths import (
    TUNING_SEED,
    RecordIdType,
    calls_dir,
    claim_root,
    events_of,
    load_first_record,
    tables_dir,
    write_json_atomic,
)
from studies.capacity_estimators import CapacityRecord, IssueScale
from studies.capacity_evidence import Evidence, build_evidence
from studies.capacity_labeller import (
    Answer,
    AnswerError,
    Prompt,
    capacity_series,
    tools_only_answer,
)
from studies.capacity_llm import (
    MODEL_ID,
    BudgetGuard,
    CallStore,
    LabellerModel,
    ModelError,
    ModelReply,
    labelled_answer,
)
from studies.capacity_recovery import (
    CONTRAST_CONFIDENCE,
    FALSE_MOVEMENT_GATE,
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
    segments_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("score_labeller")

VALUE_THRESHOLD_POINTS: Final[float] = -1.0
"""The model adds value at this change in mean absolute capacity error or lower, with the interval
wholly below zero."""

CONTROL_HIT_GATE: Final[float] = 0.8
"""The share of the positive control's steps the tools-plus-model arm must find."""

FRESH_RECORDS: Final[tuple[RecordIdType, ...]] = ("draw01", "draw02")
"""The two fresh draws."""


class CachedOnlyModel:
    """A model that answers nothing: a prompt missing from the cache raises `ModelError`."""

    def __init__(self, *, model_id: str) -> None:
        """Name the model whose cached answers are read.

        Args:
            model_id: The identifier in the cache key.
        """
        self.model_id = model_id

    def complete(self, *, prompt: Prompt) -> ModelReply:
        """Raise, because scoring never calls a model."""
        del prompt
        msg = "the answer is not in the cache"
        raise ModelError(msg)


def checked_hindsight(*, answer: Answer, evidence: Evidence) -> Answer:
    """Return the hindsight answer after the hard checks drop the failing knots."""
    return apply_hindsight(answer=answer, evidence=evidence).answer


def scale_of(*, answers: dict[int, Answer], record: CapacityRecord) -> IssueScale:
    """Return the capacity scale of per-plant answers (a missing plant keeps the static power)."""
    return answers_to_scale(answers=answers, record=record)


def main() -> int:  # noqa: PLR0912, PLR0915
    """Score every arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    context = LabellerContext(root=root)
    plants = len(context.sites)
    model: LabellerModel = CachedOnlyModel(model_id="stub" if arguments.smoke else MODEL_ID)
    store = CallStore(directory=calls_dir(root=root), log_path=root / "scoring_misses.jsonl")
    guard = BudgetGuard(ceiling_usd=1e9, max_calls=10**9, max_input_tokens=10**12)

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
            store=store, model=model, guard=guard, prompt=prompt, sample_index=sample_index,
            label=label, plant=plant, first_day=first_day, last_day=last_day,
        )  # fmt: skip

    # Collect every answer.
    raw_answers: dict[tuple[str, RecordIdType, int, int], Answer] = {}
    evidences: dict[tuple[RecordIdType, int], Evidence] = {}
    missing: list[str] = []
    for record_id in (*HINDSIGHT_RECORDS, *CONTROL_RECORDS):
        for plant in range(plants):
            evidences[(record_id, plant)] = context.hindsight_case(
                record_id=record_id, plant=plant, kind="tools"
            ).evidence
            samples = TOOLS_SAMPLES if record_id in HINDSIGHT_RECORDS else 1
            arms = [("tools_llm", s) for s in range(samples)]
            if record_id in HINDSIGHT_RECORDS:
                arms.append(("raw_llm", 0))
            for arm, sample in arms:
                job = HindsightJob(arm, record_id, plant, sample)  # ty: ignore[invalid-argument-type]
                try:
                    answer, _ = run_hindsight_job(
                        context=context, job=job, answer_function=answer_function
                    )
                except ModelError, AnswerError:
                    missing.append(job.label)
                    continue
                raw_answers[(arm, record_id, plant, sample)] = answer
    first = context.first_events
    days = len(context.dates)
    original_record = context.record(record_id="original")

    # Tools-only answers (no model) and the checked versions of the model's answers.
    tools_only = {
        (record_id, plant): tools_only_answer(
            evidence=evidences[(record_id, plant)],
            first_day=context.dates[0].astype(date),
            last_day=context.dates[-1].astype(date),
        )
        for record_id in (*HINDSIGHT_RECORDS, *CONTROL_RECORDS)
        for plant in range(plants)
    }
    checked = {
        key: checked_hindsight(answer=answer, evidence=evidences[(key[1], key[2])])
        for key, answer in raw_answers.items()
    }

    def answers_for(
        *, arm: str, record_id: RecordIdType, sample: int, after: bool
    ) -> dict[int, Answer] | None:
        source = checked if after else raw_answers
        if arm == "tools_only":
            return {p: tools_only[(record_id, p)] for p in range(plants)}
        found = {
            p: source[(arm, record_id, p, sample)]
            for p in range(plants)
            if (arm, record_id, p, sample) in source
        }
        return found if len(found) == plants else None

    def scale_for(
        *, arm: str, record_id: RecordIdType, sample: int, after: bool
    ) -> IssueScale | None:
        answers = answers_for(arm=arm, record_id=record_id, sample=sample, after=after)
        return (
            None
            if answers is None
            else scale_of(answers=answers, record=context.record(record_id=record_id))
        )

    # Event scores and the planned contrasts C1 and C2.
    score_tables: list[pl.DataFrame] = []
    errors: dict[tuple[str, str, int, bool], np.ndarray] = {}
    for after in (False, True):
        for arm, samples in (
            ("tools_only", [0]),
            ("tools_llm", list(range(TOOLS_SAMPLES))),
            ("raw_llm", [0]),
        ):
            for sample in samples:
                per_draw = []
                for record_id in FRESH_RECORDS:
                    planted = scale_for(arm=arm, record_id=record_id, sample=sample, after=after)
                    original = scale_for(arm=arm, record_id="original", sample=sample, after=after)
                    if planted is None or original is None:
                        per_draw = []
                        break
                    events = events_of(root=root, record_id=record_id)
                    capacity_error = score_events(
                        events=events, planted=planted, original=original, days=days
                    )
                    answers_p = answers_for(
                        arm=arm, record_id=record_id, sample=sample, after=after
                    )
                    answers_o = answers_for(
                        arm=arm, record_id="original", sample=sample, after=after
                    )
                    assert answers_p is not None
                    assert answers_o is not None
                    detection = score_draw(
                        events=events, planted=answers_p, original=answers_o, record=original_record
                    )
                    table = detection.with_columns(
                        capacity_error=pl.Series(capacity_error["error"]),
                        arm=pl.lit(arm),
                        sample=pl.lit(sample),
                        draw=pl.lit(record_id),
                        after_checks=pl.lit(after),
                    )
                    score_tables.append(table)
                    per_draw.append(capacity_error["error"].to_numpy())
                if per_draw:
                    errors[(arm, "fresh", sample, after)] = np.concatenate(per_draw) * 100.0
    event_scores = pl.concat(score_tables) if score_tables else pl.DataFrame()
    if event_scores.height:
        event_scores.write_parquet(tables_dir(root=root) / "event_scores.parquet")
    contrasts = []
    for after in (False, True):
        tools_errors = {
            s: errors.get(("tools_llm", "fresh", s, after)) for s in range(TOOLS_SAMPLES)
        }
        baseline = errors.get(("tools_only", "fresh", 0, after))
        raw = errors.get(("raw_llm", "fresh", 0, after))
        if baseline is not None and all(v is not None for v in tools_errors.values()):
            matrix = np.stack([tools_errors[s] - baseline for s in range(TOOLS_SAMPLES)], axis=1)
            contrasts.append(("C1 tools+LLM minus tools only", after, matrix))
        if raw is not None and all(v is not None for v in tools_errors.values()):
            matrix = np.stack([tools_errors[s] - raw for s in range(TOOLS_SAMPLES)], axis=1)
            contrasts.append(("C2 tools+LLM minus LLM on raw summaries", after, matrix))
    rows = []
    for name, after, matrix in contrasts:
        mean, lower, upper = event_bootstrap_interval(
            differences=matrix, confidence=CONTRAST_CONFIDENCE, seed=TUNING_SEED
        )
        verdict = (
            "adds value"
            if mean <= VALUE_THRESHOLD_POINTS and upper < 0
            else "worse"
            if lower > 0
            else "not shown"
        )
        rows.append(
            {
                "contrast": name,
                "after_checks": after,
                "events": matrix.shape[0],
                "mean_points": mean,
                "lower": lower,
                "upper": upper,
                "verdict": verdict,
            }
        )
    pl.DataFrame(
        rows,
        schema={
            "contrast": pl.String,
            "after_checks": pl.Boolean,
            "events": pl.Int64,
            "mean_points": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "verdict": pl.String,
        },
    ).write_parquet(tables_dir(root=root) / "contrasts.parquet")

    # Noise floor: sample 0 against sample 1 on the original record, fresh draws' scored days.
    floor = None
    s0 = scale_for(arm="tools_llm", record_id="original", sample=0, after=False)
    s1 = scale_for(arm="tools_llm", record_id="original", sample=1, after=False)
    if s0 is not None and s1 is not None:
        values = []
        for record_id in FRESH_RECORDS:
            for event in events_of(root=root, record_id=record_id):
                start, stop = event.scored_days
                values.append(
                    np.mean(
                        np.abs(
                            s0.scale_mw[event.plant, start:stop]
                            / s1.scale_mw[event.plant, start:stop]
                            - 1.0
                        )
                    )
                )
        floor = float(np.mean(values)) * 100.0
    write_json_atomic(
        path=tables_dir(root=root) / "noise_floor.json",
        payload={"mean_abs_difference_points": floor, "missing_answers": missing},
    )

    # C3 and the controls.
    movement_rows = []
    windows = first.false_movement_windows
    for arm in ("tools_only", "tools_llm"):
        per_sample = []
        for sample in range(TOOLS_SAMPLES) if arm == "tools_llm" else [0]:
            scale = scale_for(arm=arm, record_id="original", sample=sample, after=False)
            if scale is not None:
                per_sample.append(movement_by_window(windows=windows, scale_mw=scale.scale_mw))
        if per_sample:
            share, lower, upper = movement_interval(
                per_window=np.mean(per_sample, axis=0),
                confidence=CONTRAST_CONFIDENCE,
                seed=TUNING_SEED,
            )
            movement_rows.append(
                {
                    "arm": arm,
                    "false_movement": share,
                    "lower": lower,
                    "upper": upper,
                    "gate": FALSE_MOVEMENT_GATE,
                    "within_gate": share <= FALSE_MOVEMENT_GATE,
                }
            )
    part1 = root / "part1"
    if (part1 / "summary.parquet").exists():
        movement_rows.extend(
            {
                "arm": row["estimator"],
                "false_movement": row["false_movement"],
                "lower": None,
                "upper": None,
                "gate": FALSE_MOVEMENT_GATE,
                "within_gate": row["within_gate"],
            }
            for row in pl.read_parquet(part1 / "summary.parquet").iter_rows(named=True)
        )
    pl.DataFrame(
        movement_rows,
        schema={
            "arm": pl.String,
            "false_movement": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "gate": pl.Float64,
            "within_gate": pl.Boolean,
        },
    ).write_parquet(tables_dir(root=root) / "false_movement.parquet")
    control_rows = []
    for arm in ("tools_only", "tools_llm"):
        planted = answers_for(arm=arm, record_id="control", sample=0, after=False)
        original = answers_for(arm=arm, record_id="original", sample=0, after=False)
        if planted is not None and original is not None:
            table = score_draw(
                events=events_of(root=root, record_id="control"),
                planted=planted,
                original=original,
                record=original_record,
            )
            control_rows.append(
                {
                    "control": "positive (30% steps)",
                    "arm": arm,
                    "events": table.height,
                    "hit_rate": float(table["hit"].cast(pl.Float64).mean()),  # ty: ignore[invalid-argument-type]
                    "false_alarms": false_alarms_per_plant(scores=table),
                }
            )
        null = answers_for(arm=arm, record_id="null", sample=0, after=False)
        if null is not None:
            steps = sum(
                1
                for p in range(plants)
                for s in segments_of(knots=null[p].knots, dates=context.dates)
                if abs(s.change) >= 0.03
            )
            control_rows.append(
                {
                    "control": "null (no change)",
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
    ).write_parquet(tables_dir(root=root) / "controls.parquet")

    # Capacity series for the charts and the real-data comparisons.
    series: dict[str, np.ndarray] = {}
    part1_original = (
        np.load(part1 / "scales_original.npz") if (part1 / "scales_original.npz").exists() else {}
    )
    for record_id in ("original", *FRESH_RECORDS):
        part1_scales = (
            part1_original
            if record_id == "original"
            else (
                np.load(part1 / f"scales_{record_id}.npz")
                if (part1 / f"scales_{record_id}.npz").exists()
                else {}
            )
        )
        for name in part1_scales:
            series[f"{record_id}/{name}"] = np.asarray(part1_scales[name])
        series[f"{record_id}/tools_only"] = scale_for(
            arm="tools_only", record_id=record_id, sample=0, after=False
        ).scale_mw  # ty: ignore[unresolved-attribute]
        for after, tag in ((False, "llm_hindsight"), (True, "llm_hindsight_checked")):
            scale = scale_for(arm="tools_llm", record_id=record_id, sample=0, after=after)
            if scale is not None:
                series[f"{record_id}/{tag}"] = scale.scale_mw
    causal_rows = []
    for record_id in CAUSAL_RECORDS:
        stitched = np.full((plants, days), np.nan)
        complete = True
        for plant in range(plants):
            try:
                steps = run_causal_chain(
                    context=context,
                    record_id=record_id,
                    plant=plant,
                    answer_function=answer_function,
                )
            except ModelError, AnswerError:
                complete = False
                continue
            bounds = [context.day_index(day=d) for d in (*CAUSAL_ISSUE_DAYS, date(2026, 7, 1))]
            for step, (a, b) in zip(steps, pairwise(bounds), strict=True):
                curve = capacity_series(
                    knots=step.outcome.answer.knots,
                    dates=context.dates,
                    static_p99_mw=float(original_record.static_p99_mw[plant]),
                )
                stitched[plant, a + 1 : b + 1] = curve[a + 1 : b + 1]
            away = np.nanmean(
                np.abs(
                    stitched[plant, bounds[0] + 1 :] / original_record.static_p99_mw[plant] - 1.0
                )
                > 0.05
            )
            causal_rows.append(
                {
                    "record": record_id,
                    "plant": plant,
                    "share_of_days_more_than_5pct_from_static": float(away),
                    "calls_failing_a_check": sum(s.outcome.fell_back for s in steps),
                    "calls_with_knots": sum(bool(s.answer.knots) for s in steps),
                }
            )
        if complete:
            series[f"{record_id}/llm_causal"] = stitched
    pl.DataFrame(
        causal_rows,
        schema={
            "record": pl.String,
            "plant": pl.Int64,
            "share_of_days_more_than_5pct_from_static": pl.Float64,
            "calls_failing_a_check": pl.Int64,
            "calls_with_knots": pl.Int64,
        },
    ).write_parquet(tables_dir(root=root) / "causal.parquet")
    np.savez_compressed(tables_dir(root=root) / "capacity_series.npz", **series)  # ty: ignore[invalid-argument-type]

    # Real data: agreement with the model's hindsight labels, and the real low months.
    agreement, low_rows = [], []
    reference = series.get("original/llm_hindsight")
    for key in sorted(
        k for k in series if k.startswith("original/") and k != "original/llm_hindsight"
    ):
        if reference is None:
            break
        agreement.append(
            {
                "estimator": key.split("/")[1],
                "mean_abs_difference_share": float(
                    np.nanmean(np.abs(series[key] / reference - 1.0))
                ),
            }
        )
    pl.DataFrame(
        agreement, schema={"estimator": pl.String, "mean_abs_difference_share": pl.Float64}
    ).write_parquet(tables_dir(root=root) / "agreement.parquet")
    from studies.capacity_recovery import calendar_month_spans

    spans = calendar_month_spans(record=original_record)
    for month in first.low_months:
        start, length = spans[month]
        for key in sorted(k for k in series if k.startswith("original/")):
            value = (
                series[key][first.low_plant, start : start + length]
                / original_record.static_p99_mw[first.low_plant]
            )
            low_rows.append(
                {
                    "estimator": key.split("/")[1],
                    "low_month_index": first.low_months.index(month) + 1,
                    "mean_capacity_share_of_static": float(np.nanmean(value)),
                }
            )
    pl.DataFrame(
        low_rows,
        schema={
            "estimator": pl.String,
            "low_month_index": pl.Int64,
            "mean_capacity_share_of_static": pl.Float64,
        },
    ).write_parquet(tables_dir(root=root) / "real_low_months.parquet")

    # ANM plant on the record with the capped hours left in: tools only.
    blind = load_first_record(anm_blind=True)
    plant = first.anm_plant
    evidence = build_evidence(
        record=blind,
        plant=plant,
        last_day=days - 1,
        fit=context.fits[plant],
        cap=context.caps[plant],
    )
    answer = tools_only_answer(
        evidence=evidence,
        first_day=context.dates[0].astype(date),
        last_day=context.dates[-1].astype(date),
    )
    causes = [knot.cause for knot in answer.knots[1:]]
    pl.DataFrame(
        [
            {
                "tools_only_knots": len(answer.knots),
                "knots_naming_curtailment": causes.count("curtailment"),
                "cap_binding_days": int(np.nansum(evidence.cap_binding_share >= 0.5))
                if evidence.cap_binding_share is not None
                else 0,
            }
        ]
    ).write_parquet(tables_dir(root=root) / "anm_spot_check.parquet")
    _LOG.info("scored; %d answers missing", len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
