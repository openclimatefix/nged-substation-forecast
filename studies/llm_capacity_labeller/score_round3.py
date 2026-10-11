"""Score round 3 and the free re-scores of rounds 1 and 2 (post hoc, except where marked).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. The
script makes no model call: it rebuilds each prompt and reads the answer from the cache.

**What it adds to rounds 1 and 2** (outputs under `<root>/tables_round3/`):

- `hit_rates.parquet`: every arm's hit rate with a Wilson interval, the number of events with at
  least one hit (the two samples of an event are correlated), by round and by check (`none`,
  `original`, `amended`).
- `placebo.parquet`: the hit rate of each arm's answers on the null record (nothing changes)
  scored against the event dates of each draw. This is the rate chance gives.
- `size_bands.parquet`: the real and the placebo hit rates by planted size.
- `contrasts_robust.parquet`: round 2's post hoc contrasts with intervals that resample whole
  events (both samples together) and whole calendar months, with the spread of one event's
  difference and the number of events a one-point effect would need.
- `noise_floor.parquet`, `notes_tally.json`, and `real_low_months_round2.parquet`.

**Round 3 (post hoc; the only evaluation of the amended check).** Round 2's tools, prompt, and
samples are frozen. The amended step check (`check_steps(local_scan_candidates=True)`) was written
down and tested before any round 3 call. It is evaluated on fresh draws 05 and 06. Replaying draws
03 and 04 through it is exploratory. The easy case (a 60% loss for 90 days from a random day, two
records) asks whether the model follows a clear signal at all.
"""

import argparse
import logging
import sys
from datetime import date
from typing import Final

import numpy as np
import polars as pl
from labeller_jobs import LabellerContext, VersionType
from labeller_paths import (
    TUNING_SEED,
    RecordIdType,
    calls_dir,
    claim_root,
    events_of,
    write_json_atomic,
)
from score_labeller import CachedOnlyModel
from studies.capacity_labeller import (
    Answer,
    AnswerError,
    tools_only_answer,
    tools_only_v2_answer,
)
from studies.capacity_llm import MODEL_ID, BudgetGuard, CallStore, ModelError, labelled_answer
from studies.capacity_recovery import CONTRAST_CONFIDENCE, score_events
from studies.capacity_sanity import apply_hindsight
from studies.labeller_scoring import (
    answers_to_scale,
    events_bootstrap_interval,
    events_needed,
    placebo_hits,
    score_draw,
    size_band,
    wilson_interval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("score_round3")

ROUND1_DRAWS: Final[tuple[RecordIdType, ...]] = ("draw01", "draw02")
ROUND2_DRAWS: Final[tuple[RecordIdType, ...]] = ("draw03", "draw04")
ROUND3_DRAWS: Final[tuple[RecordIdType, ...]] = ("draw05", "draw06")
EASY_RECORDS: Final[tuple[RecordIdType, ...]] = ("easy01", "easy02")
ROUND4_DRAWS: Final[tuple[RecordIdType, ...]] = tuple(f"draw{n:02d}" for n in range(7, 15))  # ty: ignore[invalid-assignment]
GRID_30_90: Final[tuple[RecordIdType, ...]] = ("grid01", "grid02")
GRID_60_30: Final[tuple[RecordIdType, ...]] = ("grid03", "grid04")
"""The records each round is scored on."""

SCAN_NOISE_WORDS: Final[tuple[str, ...]] = ("seasonal", "weather", "artefact", "artifact", "noise")
"""A round 2 note counts as citing the scan as noise when it names the scan and one of these."""


def _mean_hit(*, frame: pl.DataFrame) -> float | None:
    """Return the share of rows that are hits, or `None` for no rows."""
    return float(np.mean(frame["hit"].to_numpy())) if frame.height else None


def main() -> int:  # noqa: PLR0912, PLR0915
    """Score round 3 and the re-scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    out = root / "tables_round3"
    out.mkdir(exist_ok=True)
    context = LabellerContext(root=root)
    plants, dates = len(context.sites), context.dates
    first_day, last_day = dates[0].astype(date), dates[-1].astype(date)
    model_id = "stub" if arguments.smoke else MODEL_ID
    store = CallStore(directory=calls_dir(root=root), log_path=root / "scoring_misses.jsonl")
    guard = BudgetGuard(ceiling_usd=1e9, max_calls=10**9, max_input_tokens=10**12)
    model = CachedOnlyModel(model_id=model_id)
    original_record = context.record(record_id="original")
    cache: dict[tuple[str, VersionType, RecordIdType, int, int], Answer | None] = {}

    def llm_answer(
        *, arm: str, version: VersionType, record_id: RecordIdType, plant: int, sample: int
    ) -> Answer | None:
        key = (arm, version, record_id, plant, sample)
        if key not in cache:
            case = context.hindsight_case(
                record_id=record_id,
                plant=plant,
                kind="tools" if arm == "tools_llm" else "raw",
                version=version,
            )
            try:
                cache[key] = labelled_answer(
                    store=store,
                    model=model,
                    guard=guard,
                    prompt=case.prompt,
                    sample_index=sample,
                    label="",
                    plant=context.sites[plant],
                    first_day=case.first_day,
                    last_day=case.last_day,
                )
            except ModelError, AnswerError:
                cache[key] = None
        return cache[key]

    def book(
        *, arm: str, record_id: RecordIdType, sample: int, check: str
    ) -> dict[int, Answer] | None:
        """Return every plant's answer of an arm (`tools_only_v1`, `tools_llm_v2`, ...)."""
        name, _, version = arm.rpartition("_")
        if name == "static":
            return {
                p: Answer(
                    plant=context.sites[p],
                    first_day=first_day,
                    last_day=last_day,
                    no_change=True,
                    knots=(),
                )
                for p in range(plants)
            }
        if name == "tools_only":
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
        found = {
            p: llm_answer(arm=name, version=version, record_id=record_id, plant=p, sample=sample)  # ty: ignore[invalid-argument-type]
            for p in range(plants)
        }
        if any(a is None for a in found.values()):
            return None
        answers = {p: a for p, a in found.items() if a is not None}
        if check == "none":
            return answers
        return {
            p: apply_hindsight(
                answer=a,
                evidence=context.hindsight_case(
                    record_id=record_id,
                    plant=p,
                    kind="tools",
                    version=version,
                ).evidence,
                local_scan_candidates=check == "amended",
            ).answer
            for p, a in answers.items()
        }

    def scored(
        *, arm: str, records: tuple[RecordIdType, ...], samples: int, check: str
    ) -> pl.DataFrame | None:
        """Score an arm's answers on the records' events, one row per event and sample."""
        frames = []
        for record_id in records:
            events = events_of(root=root, record_id=record_id)
            for sample in range(samples):
                planted = book(arm=arm, record_id=record_id, sample=sample, check=check)
                original = book(arm=arm, record_id="original", sample=sample, check=check)
                if planted is None or original is None:
                    return None
                table = score_draw(
                    events=events, planted=planted, original=original, record=original_record
                )
                capacity = score_events(
                    events=events,
                    planted=answers_to_scale(answers=planted, record=original_record),
                    original=answers_to_scale(answers=original, record=original_record),
                    days=len(dates),
                )
                frames.append(
                    table.with_columns(
                        capacity_error=pl.Series(capacity["error"]) * 100.0,
                        record=pl.lit(record_id),
                        sample=pl.lit(sample),
                        event=pl.Series(range(table.height)),
                    )
                )
        return pl.concat(frames)

    def hit_row(
        *, round_name: str, label: str, arm: str, check: str, table: pl.DataFrame
    ) -> dict[str, object]:
        hits = int(table["hit"].sum())
        lower, upper = wilson_interval(successes=hits, trials=table.height)
        any_hit = int(table.group_by("record", "event").agg(h=pl.col("hit").any())["h"].sum())
        events = table.select("record", "event").unique().height
        event_lower, event_upper = wilson_interval(successes=any_hit, trials=events)
        return {
            "round": round_name,
            "label": label,
            "arm": arm,
            "check": check,
            "events_times_samples": table.height,
            "hits": hits,
            "hit_rate": hits / table.height,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
            "events": events,
            "events_with_a_hit": any_hit,
            "event_hit_rate": any_hit / events,
            "event_wilson_95_lower": event_lower,
            "event_wilson_95_upper": event_upper,
            "capacity_error_points": float(table["capacity_error"].mean()),  # ty: ignore[invalid-argument-type]
        }

    specs = [
        (
            "round 1",
            "planned",
            ROUND1_DRAWS,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v1", 1, ("none",)),
                ("tools_llm_v1", 2, ("none", "original")),
                ("raw_llm_v1", 1, ("none",)),
            ],
        ),
        (
            "round 2",
            "post hoc",
            ROUND2_DRAWS,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v1", 2, ("none", "original")),
                ("tools_llm_v2", 2, ("none", "original", "amended")),
                ("raw_llm_v1", 1, ("none",)),
            ],
        ),
        (
            "round 3",
            "post hoc",
            ROUND3_DRAWS,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, ("none", "original", "amended")),
            ],
        ),
        (
            "easy case",
            "post hoc",
            EASY_RECORDS,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, ("none", "original", "amended")),
            ],
        ),
    ]
    full = tuple(
        r
        for r in ROUND4_DRAWS
        if all(
            book(arm="tools_llm_v2", record_id=r, sample=k, check="none") is not None
            for k in range(2)
        )
    )
    model_checks = ("none", "original", "amended")
    last_draw = full[-1][-2:] if full else "06"
    specs += [
        (
            f"draws 07 to {last_draw}",
            "post hoc",
            full,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, model_checks),
            ],
        ),
        (
            f"draws 05 to {last_draw} pooled",
            "post hoc, round 3 pre-registered",
            (*ROUND3_DRAWS, *full),
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, model_checks),
            ],
        ),
        (
            "grid 30% for 90 days",
            "post hoc",
            GRID_30_90,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, model_checks),
            ],
        ),
        (
            "grid 60% for 30 days",
            "post hoc",
            GRID_60_30,
            [
                ("static_v1", 1, ("none",)),
                ("tools_only_v2", 1, ("none",)),
                ("tools_llm_v2", 2, model_checks),
            ],
        ),
    ]
    rows = []
    tables: dict[tuple[str, str, str], pl.DataFrame] = {}
    for round_name, label, records, arms in specs:
        for arm, samples, checks in arms:
            for check in checks:
                table = scored(arm=arm, records=records, samples=samples, check=check)
                if table is None:
                    _LOG.warning("missing answers: %s %s %s", round_name, arm, check)
                    continue
                tables[(round_name, arm, check)] = table
                rows.append(
                    hit_row(round_name=round_name, label=label, arm=arm, check=check, table=table)
                )
    pl.DataFrame(rows).write_parquet(out / "hit_rates.parquet")
    if any(t.height for t in tables.values()):
        pl.concat(
            [
                t.with_columns(round=pl.lit(r), arm=pl.lit(a), check=pl.lit(c))
                for (r, a, c), t in tables.items()
            ]
        ).write_parquet(out / "event_scores.parquet")

    # Candidate coverage: how many planted losses the tools list near the start.
    coverage_rows = []
    for name, records in (
        ("draws 01 and 02 (round 1)", ROUND1_DRAWS),
        ("draws 03 and 04 (round 2)", ROUND2_DRAWS),
        (f"draws 05 to {last_draw}", (*ROUND3_DRAWS, *full)),
        ("easy case (60% for 90 days)", EASY_RECORDS),
        ("grid 30% for 90 days", GRID_30_90),
        ("grid 60% for 30 days", GRID_60_30),
    ):
        total = changepoint = scan = either = 0
        for record_id in records:
            for event in events_of(root=root, record_id=record_id):
                found = context.hindsight_case(
                    record_id=record_id, plant=event.plant, kind="tools"
                ).evidence
                near_change = any(
                    c.direction == "down" and abs(c.day_index - event.first_day) <= 14
                    for c in found.changepoints
                )
                near_scan = any(
                    e.direction == "down" and abs(e.first_day - event.first_day) <= 14
                    for e in found.local_episodes
                )
                total += 1
                changepoint += near_change
                scan += near_scan
                either += near_change or near_scan
        coverage_rows.append(
            {
                "records": name,
                "events": total,
                "down_changepoint_candidate_within_14_days_of_start": changepoint,
                "down_local_scan_episode_starting_within_14_days": scan,
                "either": either,
            }
        )
    pl.DataFrame(coverage_rows).write_parquet(out / "candidate_coverage_all.parquet")

    # Placebo: null-record answers scored against each draw's event dates.
    placebo_rows, band_rows = [], []
    for arm, checks in (
        ("tools_only_v1", ("none",)),
        ("tools_only_v2", ("none",)),
        ("tools_llm_v1", ("none",)),
        ("tools_llm_v2", ("none", "amended")),
    ):
        for check in checks:
            null = book(arm=arm, record_id="null", sample=0, check=check)
            if null is None:
                continue
            for round_name, records in (
                ("round 1 dates", ROUND1_DRAWS),
                ("round 2 dates", ROUND2_DRAWS),
                ("round 3 dates", ROUND3_DRAWS),
                (f"draws 05 to {last_draw} dates", (*ROUND3_DRAWS, *full)),
            ):
                parts = [
                    placebo_hits(
                        events=events_of(root=root, record_id=r), answers=null, dates=dates
                    )
                    for r in records
                ]
                table = pl.concat(parts)
                lower, upper = wilson_interval(
                    successes=int(table["hit"].sum()), trials=table.height
                )
                placebo_rows.append(
                    {
                        "arm": arm,
                        "check": check,
                        "event_dates_of": round_name,
                        "events": table.height,
                        "placebo_hits": int(table["hit"].sum()),
                        "placebo_hit_rate": _mean_hit(frame=table),
                        "wilson_95_lower": lower,
                        "wilson_95_upper": upper,
                    }
                )
                if round_name in ("round 2 dates", f"draws 05 to {last_draw} dates"):
                    real = tables.get(
                        (
                            "round 2"
                            if round_name == "round 2 dates"
                            else f"draws 05 to {last_draw} pooled",
                            arm,
                            check,
                        )
                    )
                    for band in ("under 15%", "15% to 25%", "25% and over"):
                        mask = pl.Series([size_band(size=s) == band for s in table["size"]])
                        real_band = (
                            None
                            if real is None
                            else real.filter(
                                pl.Series([size_band(size=s) == band for s in real["size"]])
                            )
                        )
                        band_rows.append(
                            {
                                "arm": arm,
                                "check": check,
                                "event_dates_of": round_name,
                                "size_band": band,
                                "events": int(mask.sum()),
                                "placebo_hit_rate": _mean_hit(frame=table.filter(mask)),
                                "real_hit_rate": None
                                if real_band is None
                                else _mean_hit(frame=real_band),
                            }
                        )
    pl.DataFrame(placebo_rows).write_parquet(out / "placebo.parquet")
    pl.DataFrame(band_rows).write_parquet(out / "size_bands.parquet")

    # Robust intervals for round 2's post hoc contrasts, and the powering of round 1.
    robust = []
    errors_path = root / "tables_round2" / "errors.npz"
    if errors_path.exists():
        with np.load(errors_path) as stored:
            errors = {key: stored[key] for key in stored.files}
        months = errors.pop("months")
        pairs = [
            ("v2 tools only minus v1 tools only", "tools_only_v2", 1, "tools_only_v1", 1),
            ("v2 tools+LLM minus v1 tools+LLM", "tools_llm_v2", 2, "tools_llm_v1", 2),
            ("v2 tools+LLM minus v2 tools only", "tools_llm_v2", 2, "tools_only_v2", 1),
            ("v1 tools+LLM minus v1 tools only", "tools_llm_v1", 2, "tools_only_v1", 1),
            (
                "v2 tools+LLM minus LLM on raw summaries (confounded: raw arm kept the v1 prompt)",
                "tools_llm_v2",
                2,
                "raw_llm_v1",
                1,
            ),
        ]
        for after in (0, 1):
            for name, left, left_n, right, right_n in pairs:
                try:
                    left_matrix = np.stack(
                        [errors[f"{left}|{s}|{after}"] for s in range(left_n)], axis=1
                    )
                    right_matrix = np.stack(
                        [errors[f"{right}|{s}|{after}"] for s in range(right_n)], axis=1
                    )
                except KeyError:
                    continue
                difference = left_matrix.mean(axis=1, keepdims=True) - right_matrix.mean(
                    axis=1, keepdims=True
                )
                for resampled, clusters in (("events", None), ("calendar months", months)):
                    mean, lower, upper = events_bootstrap_interval(
                        differences=difference,
                        confidence=CONTRAST_CONFIDENCE,
                        seed=TUNING_SEED,
                        clusters=clusters,
                    )
                    robust.append(
                        {
                            "label": "post hoc",
                            "contrast": name,
                            "after_original_checks": bool(after),
                            "resampled": resampled,
                            "events": difference.shape[0],
                            "mean_points": mean,
                            "lower": lower,
                            "upper": upper,
                            "sd_of_one_event_points": float(difference.std(ddof=1)),
                            "events_needed_for_minus_1_point": events_needed(
                                sd_points=float(difference.std(ddof=1)), effect_points=1.0
                            ),
                        }
                    )
    round1_scores = root / "tables" / "event_scores.parquet"
    if round1_scores.exists():
        r1 = pl.read_parquet(round1_scores).filter(~pl.col("after_checks"))
        per = r1.group_by("arm", "draw", "plant", "shape", "size").agg(
            e=pl.col("capacity_error").mean()
        )
        wide = per.pivot(
            on="arm", index=["draw", "plant", "shape", "size"], values="e"
        ).drop_nulls()
        month_of = {
            (record_id, event.plant, event.shape, event.size): str(
                dates[event.first_day].astype("datetime64[M]")
            )
            for record_id in ROUND1_DRAWS
            for event in events_of(root=root, record_id=record_id)
        }
        wide_months = np.array(
            [
                month_of[(d, p, sh, sz)]
                for d, p, sh, sz in zip(
                    wide["draw"], wide["plant"], wide["shape"], wide["size"], strict=True
                )
            ]
        )
        for left, right in (("tools_llm", "tools_only"), ("tools_llm", "raw_llm")):
            diff = (wide[left] - wide[right]).to_numpy() * 100.0
            sd = float(diff.std(ddof=1))
            for resampled, clusters in (("events", None), ("calendar months", wide_months)):
                mean, lower, upper = events_bootstrap_interval(
                    differences=diff.reshape(-1, 1),
                    confidence=CONTRAST_CONFIDENCE,
                    seed=TUNING_SEED,
                    clusters=clusters,
                )
                robust.append(
                    {
                        "label": "planned (round 1; the plan's resampling of (event, sample) "
                        "pairs gave the intervals on the page's table)",
                        "contrast": f"{left} minus {right}",
                        "after_original_checks": False,
                        "resampled": resampled,
                        "events": len(diff),
                        "mean_points": mean,
                        "lower": lower,
                        "upper": upper,
                        "sd_of_one_event_points": sd,
                        "events_needed_for_minus_1_point": events_needed(
                            sd_points=sd, effect_points=1.0
                        ),
                    }
                )
    pl.DataFrame(
        robust,
        schema={
            "label": pl.String,
            "contrast": pl.String,
            "after_original_checks": pl.Boolean,
            "resampled": pl.String,
            "events": pl.Int64,
            "mean_points": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "sd_of_one_event_points": pl.Float64,
            "events_needed_for_minus_1_point": pl.Int64,
        },
    ).write_parquet(out / "contrasts_robust.parquet")

    # Round 3 contrast on the fresh draws, and the noise floor.
    round3 = []
    left = tables.get(("round 3", "tools_llm_v2", "amended"))
    right = tables.get(("round 3", "tools_only_v2", "none"))
    if left is not None and right is not None:
        left_matrix = left.pivot(
            on="sample", index=["record", "event"], values="capacity_error"
        ).sort("record", "event")
        left_error = left_matrix.drop("record", "event").to_numpy().mean(axis=1)
        right_error = right.sort("record", "event")["capacity_error"].to_numpy()
        difference = (left_error - right_error).reshape(-1, 1)
        months = np.array(
            [
                str(dates[e.first_day].astype("datetime64[M]"))
                for r in ROUND3_DRAWS
                for e in events_of(root=root, record_id=r)
            ]
        )
        for resampled, clusters in (("events", None), ("calendar months", months)):
            mean, lower, upper = events_bootstrap_interval(
                differences=difference,
                confidence=CONTRAST_CONFIDENCE,
                seed=TUNING_SEED,
                clusters=clusters,
            )
            round3.append(
                {
                    "label": "post hoc (round 3, amended check)",
                    "contrast": "tools+LLM v2 (amended checks) minus tools only v2",
                    "resampled": resampled,
                    "events": difference.shape[0],
                    "mean_points": mean,
                    "lower": lower,
                    "upper": upper,
                }
            )
    pl.DataFrame(
        round3,
        schema={
            "label": pl.String,
            "contrast": pl.String,
            "resampled": pl.String,
            "events": pl.Int64,
            "mean_points": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
        },
    ).write_parquet(out / "round3_contrast.parquet")
    floor = []
    for version in ("v1", "v2"):
        for records in (ROUND2_DRAWS, ROUND3_DRAWS, full):
            if not records:
                continue
            first = book(arm=f"tools_llm_{version}", record_id="original", sample=0, check="none")
            second = book(arm=f"tools_llm_{version}", record_id="original", sample=1, check="none")
            if first is None or second is None:
                continue
            one = answers_to_scale(answers=first, record=original_record).scale_mw
            two = answers_to_scale(answers=second, record=original_record).scale_mw
            values = []
            for record_id in records:
                for event in events_of(root=root, record_id=record_id):
                    start, stop = event.scored_days
                    values.append(
                        np.mean(
                            np.abs(
                                one[event.plant, start:stop] / two[event.plant, start:stop] - 1.0
                            )
                        )
                    )
            floor.append(
                {
                    "version": version,
                    "scored_days_of": ", ".join(records),
                    "mean_abs_difference_between_samples_points": float(np.mean(values)) * 100.0,
                }
            )
    pl.DataFrame(floor).write_parquet(out / "noise_floor.parquet")

    # Tally of round 2's reasons for no change.
    notes_path = root / "tables_round2" / "round2_no_change_notes.parquet"
    if notes_path.exists():
        notes = pl.read_parquet(notes_path)["note"].to_list()
        cites = [n for n in notes if ("local-scan" in n.lower() or "local scan" in n.lower())]
        noise = [n for n in cites if any(w in n.lower() for w in SCAN_NOISE_WORDS)]
        draw_notes = [
            n
            for n, r in zip(
                pl.read_parquet(notes_path)["note"].to_list(),
                pl.read_parquet(notes_path)["record"].to_list(),
                strict=True,
            )
            if r in ROUND2_DRAWS
        ]
        draw_cites = [
            n for n in draw_notes if "local-scan" in n.lower() or "local scan" in n.lower()
        ]
        draw_noise = [n for n in draw_cites if any(w in n.lower() for w in SCAN_NOISE_WORDS)]
        write_json_atomic(
            path=out / "notes_tally.json",
            payload={
                "notes": len(notes),
                "naming_the_local_scan": len(cites),
                "naming_the_scan_and_a_noise_word": len(noise),
                "draws_03_and_04_only": {
                    "notes": len(draw_notes),
                    "naming_the_local_scan": len(draw_cites),
                    "naming_the_scan_and_a_noise_word": len(draw_noise),
                },
                "rule": "a note names the scan ('local-scan' or 'local scan') and one of "
                + ", ".join(SCAN_NOISE_WORDS),
            },
        )

    # Real low months: round 2's series as a share of static power (opinion, not truth).
    series_path = root / "tables_round2" / "capacity_series.npz"
    low_rows = []
    if series_path.exists():
        from studies.capacity_recovery import calendar_month_spans

        spans = calendar_month_spans(record=original_record)
        with np.load(series_path) as stored:
            for tag in ("tools_only_v1", "tools_only_v2", "llm_v1", "llm_v2"):
                if f"original/{tag}" not in stored.files:
                    continue
                for index, month in enumerate(context.first_events.low_months, start=1):
                    start, length = spans[month]
                    share = (
                        stored[f"original/{tag}"][
                            context.first_events.low_plant, start : start + length
                        ]
                        / original_record.static_p99_mw[context.first_events.low_plant]
                    )
                    low_rows.append(
                        {
                            "series": tag,
                            "low_month": index,
                            "mean_capacity_share_of_static": float(np.nanmean(share)),
                        }
                    )
    pl.DataFrame(
        low_rows,
        schema={
            "series": pl.String,
            "low_month": pl.Int64,
            "mean_capacity_share_of_static": pl.Float64,
        },
    ).write_parquet(out / "real_low_months_round2.parquet")
    _LOG.info("scored round 3 and the re-scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
