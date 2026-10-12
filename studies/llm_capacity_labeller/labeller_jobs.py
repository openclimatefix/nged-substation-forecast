"""Which labeller calls the study makes, and how each call's prompt is built.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
`run_labeller.py` makes the calls and `score_labeller.py` reads the same calls back from the cache,
so both build every prompt through this module and cannot disagree about what was asked.

**The calls, in the order they run (138 in all):**

1. Hindsight, tools plus language model: 3 records (the original and the two fresh draws), 6 plants,
   2 samples each = 36.
2. Hindsight, raw weekly summaries only: 3 records, 6 plants, 1 sample = 18.
3. Controls: the synthetic null record and the positive control, 6 plants each, 1 sample = 12.
4. Causal: 2 records (the original and the first fresh draw), 6 plants, 6 monthly issue dates,
   1 sample = 72. Each call carries the estimate the previous call left, after the hard checks.

The calls that decide the planned contrasts come first, so a budget stop cuts the causal role.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal

import numpy as np
from labeller_paths import (
    RecordIdType,
    load_cap_overlays,
    load_first_events,
    load_plant_fits,
    record_for,
)
from studies.capacity_estimators import CapacityRecord
from studies.capacity_evidence import CapOverlay, Evidence, PlantFit, build_evidence
from studies.capacity_labeller import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    Answer,
    CaseKindType,
    Knot,
    Prompt,
    allowed_window,
    build_prompt,
    case_file_text,
    chart_pngs,
)
from studies.capacity_sanity import SanityOutcome, apply_causal

VersionType = Literal["v1", "v2"]
"""Round 1 (`v1`, planned) or round 2 (`v2`, post hoc: the local scan and the softer prompt)."""

ArmType = Literal["tools_llm", "raw_llm"]
"""The two arms that call the language model: with the six tools, and with weekly summaries only."""

HINDSIGHT_RECORDS: Final[tuple[RecordIdType, ...]] = ("original", "draw01", "draw02")
"""The records the hindsight arms read."""

CONTROL_RECORDS: Final[tuple[RecordIdType, ...]] = ("null", "control")
"""The synthetic null record and the positive control."""

CAUSAL_RECORDS: Final[tuple[RecordIdType, ...]] = ("original", "draw01")
"""The records the causal role reads."""

TOOLS_SAMPLES: Final[int] = 2
"""Samples per plant and record in the arm with the tools."""

CAUSAL_ISSUE_DAYS: Final[tuple[date, ...]] = tuple(date(2026, month, 1) for month in range(1, 7))
"""The monthly issue dates of the causal role: the last 6 months of the record."""

CAUSAL_PERIOD_FIRST: Final[date] = date(2024, 7, 1)
"""The first date a causal knot may take."""

CAUSAL_SHOWN_MONTHS: Final[int] = 18
"""How many trailing months the causal case file's tables list."""

READABLE_LAG_DAYS: Final[int] = 2
"""The latest day a causal case file reads is the issue day minus this (the satellite delay)."""

AnswerFunction = Callable[..., Answer]
"""Maps a prompt and its call details to a validated answer: `(prompt=, sample_index=, label=,
plant=, first_day=, last_day=, evidence=)`."""


@dataclass(frozen=True)
class CaseFile:
    """One case file, ready to send.

    Attributes:
        prompt: The system prompt, the text, and the chart images.
        evidence: The evidence the text was written from.
        first_day: The first date a knot may take.
        last_day: The last date a knot may take.
    """

    prompt: Prompt
    evidence: Evidence
    first_day: date
    last_day: date


@dataclass(frozen=True)
class HindsightJob:
    """One hindsight call.

    Attributes:
        arm: Which arm.
        record_id: The record read.
        plant: The plant's index.
        sample_index: Which sample of the same prompt.
        version: The round: `v1` (planned) or `v2` (post hoc).
    """

    arm: ArmType
    record_id: RecordIdType
    plant: int
    sample_index: int
    version: VersionType = "v1"

    @property
    def label(self) -> str:
        """A description for the log."""
        prefix = "hindsight" if self.version == "v1" else "r2/hindsight"
        return f"{prefix}/{self.arm}/{self.record_id}/plant{self.plant}/s{self.sample_index}"


def hindsight_jobs(*, plants: int, tools_samples: int = TOOLS_SAMPLES) -> list[HindsightJob]:
    """List the hindsight and control calls in the order they run.

    Args:
        plants: The number of plants.
        tools_samples: Samples per plant and record in the arm with the tools.

    Returns:
        The tools-plus-model calls on the three records, the raw-summary calls, then the controls.
    """
    jobs = [
        HindsightJob("tools_llm", record_id, plant, sample)
        for record_id in HINDSIGHT_RECORDS
        for plant in range(plants)
        for sample in range(tools_samples)
    ]
    jobs += [
        HindsightJob("raw_llm", record_id, plant, 0)
        for record_id in HINDSIGHT_RECORDS
        for plant in range(plants)
    ]
    jobs += [
        HindsightJob("tools_llm", record_id, plant, 0)
        for record_id in CONTROL_RECORDS
        for plant in range(plants)
    ]
    return jobs


class LabellerContext:
    """The records, fits, and cap overlays, and a cache of evidence and case files."""

    def __init__(self, *, root: Path) -> None:
        """Load what every case file needs.

        Args:
            root: The output root.
        """
        self.root = root
        self.first_events = load_first_events()
        original = record_for(root=root, record_id="original")
        self.fits: list[PlantFit] = load_plant_fits(record=original)
        self.caps: dict[int, CapOverlay] = load_cap_overlays(root=root)
        self.sites = original.sites
        self.dates = original.dates

    @functools.cache  # noqa: B019 - the context lives for one run
    def record(self, *, record_id: RecordIdType) -> CapacityRecord:
        """Return a record, built once."""
        return record_for(root=self.root, record_id=record_id)

    @functools.cache  # noqa: B019
    def evidence(self, *, record_id: RecordIdType, plant: int, last_day: int) -> Evidence:
        """Return one plant's evidence on a record, read up to `last_day`, built once."""
        return build_evidence(
            record=self.record(record_id=record_id),
            plant=plant,
            last_day=last_day,
            fit=self.fits[plant],
            cap=self.caps.get(plant),
        )

    def day_index(self, *, day: date) -> int:
        """Return a date's index in the record."""
        return int((np.datetime64(day) - self.dates[0]).astype(int))

    @functools.cache  # noqa: B019
    def hindsight_case(
        self,
        *,
        record_id: RecordIdType,
        plant: int,
        kind: CaseKindType,
        version: VersionType = "v1",
    ) -> CaseFile:
        """Return the whole-record case file of one plant, built once.

        Args:
            record_id: The record read.
            plant: The plant's index.
            kind: `tools` for the six tools and the two charts, `raw` for weekly summaries only.
            version: `v2` adds the local-scan table and uses the round-2 system prompt for `tools`.

        Returns:
            The case file, whose knots may take any date of the record.
        """
        evidence = self.evidence(record_id=record_id, plant=plant, last_day=len(self.dates) - 1)
        scan = version == "v2" and kind == "tools"
        text = case_file_text(evidence=evidence, kind=kind, shown_months=None, local_scan=scan)
        images = chart_pngs(evidence=evidence, shown_months=None) if kind == "tools" else ()
        first_day, last_day = allowed_window(text=text)
        return CaseFile(
            prompt=build_prompt(
                text=text, images=images, system=SYSTEM_PROMPT_V2 if scan else SYSTEM_PROMPT
            ),
            evidence=evidence,
            first_day=first_day,
            last_day=last_day,
        )

    def causal_case(
        self,
        *,
        record_id: RecordIdType,
        plant: int,
        issue_day: date,
        current_knots: tuple[Knot, ...],
        version: VersionType = "v1",
    ) -> CaseFile:
        """Return the case file of one plant on one issue date.

        Args:
            record_id: The record read.
            plant: The plant's index.
            issue_day: The day the labeller is asked.
            current_knots: The estimate the previous call left.
            version: `v2` adds the local-scan table and uses the round-2 system prompt.

        Returns:
            The case file, cut at the issue day minus `READABLE_LAG_DAYS`.
        """
        last_index = self.day_index(day=issue_day) - READABLE_LAG_DAYS
        evidence = self.evidence(record_id=record_id, plant=plant, last_day=last_index)
        text = case_file_text(
            evidence=evidence,
            kind="tools",
            shown_months=CAUSAL_SHOWN_MONTHS,
            current_knots=current_knots,
            issue_day=issue_day,
            period_first=CAUSAL_PERIOD_FIRST,
            local_scan=version == "v2",
        )
        images = chart_pngs(evidence=evidence, shown_months=CAUSAL_SHOWN_MONTHS)
        first_day, last_day = allowed_window(text=text)
        return CaseFile(
            prompt=build_prompt(
                text=text,
                images=images,
                system=SYSTEM_PROMPT_V2 if version == "v2" else SYSTEM_PROMPT,
            ),
            evidence=evidence,
            first_day=first_day,
            last_day=last_day,
        )


def run_hindsight_job(
    *, context: LabellerContext, job: HindsightJob, answer_function: AnswerFunction
) -> tuple[Answer, Evidence]:
    """Make (or read back) one hindsight call.

    Args:
        context: The context.
        job: The call.
        answer_function: Turns the case file into a validated answer.

    Returns:
        The model's answer, before the hard checks, and the evidence it was written from.
    """
    case = context.hindsight_case(
        record_id=job.record_id,
        plant=job.plant,
        kind="tools" if job.arm == "tools_llm" else "raw",
        version=job.version,
    )
    answer = answer_function(
        prompt=case.prompt,
        sample_index=job.sample_index,
        label=job.label,
        plant=context.sites[job.plant],
        first_day=case.first_day,
        last_day=case.last_day,
        evidence=case.evidence,
    )
    return answer, case.evidence


@dataclass(frozen=True)
class CausalStep:
    """One causal call.

    Attributes:
        issue_day: The day the labeller was asked.
        answer: The model's answer.
        outcome: The answer after the hard checks (the static power if any failed).
        evidence: The evidence the case file was written from.
    """

    issue_day: date
    answer: Answer
    outcome: SanityOutcome
    evidence: Evidence


def run_causal_chain(
    *,
    context: LabellerContext,
    record_id: RecordIdType,
    plant: int,
    answer_function: AnswerFunction,
    version: VersionType = "v1",
) -> list[CausalStep]:
    """Make (or read back) the six causal calls of one plant on one record, in date order.

    Args:
        context: The context.
        record_id: The record read.
        plant: The plant's index.
        answer_function: Turns the case file into a validated answer.
        version: The round.

    Returns:
        One step per issue date; each call is shown the knots the previous step kept.
    """
    steps: list[CausalStep] = []
    current: tuple[Knot, ...] = ()
    prefix = "causal" if version == "v1" else "r2/causal"
    for issue_day in CAUSAL_ISSUE_DAYS:
        case = context.causal_case(
            record_id=record_id,
            plant=plant,
            issue_day=issue_day,
            current_knots=current,
            version=version,
        )
        answer = answer_function(
            prompt=case.prompt,
            sample_index=0,
            label=f"{prefix}/{record_id}/plant{plant}/{issue_day}",
            plant=context.sites[plant],
            first_day=case.first_day,
            last_day=case.last_day,
            evidence=case.evidence,
        )
        outcome = apply_causal(answer=answer, evidence=case.evidence)
        steps.append(
            CausalStep(issue_day=issue_day, answer=answer, outcome=outcome, evidence=case.evidence)
        )
        current = outcome.answer.knots
    return steps


ROUND2_TOOLS_RECORDS: Final[tuple[RecordIdType, ...]] = ("draw03", "draw04", "original")
"""The records round 2's tools-plus-model arm reads: the two new draws and the original."""

ROUND2_V1_RECORDS: Final[tuple[RecordIdType, ...]] = ("draw03", "draw04")
"""The new draws on which round 1's tools-plus-model arm is re-run, so v1 and v2 are paired."""

ROUND2_CONTROL_RECORDS: Final[tuple[RecordIdType, ...]] = ("control2", "null")
"""Round 2's non-adjacent positive control and the synthetic null record."""

ROUND2_CAUSAL_RECORDS: Final[tuple[RecordIdType, ...]] = ("original", "draw03")
"""The records round 2's causal role reads."""


def round2_jobs(*, plants: int, tools_samples: int = TOOLS_SAMPLES) -> list[HindsightJob]:
    """List round 2's hindsight and control calls in the order they run.

    Args:
        plants: The number of plants.
        tools_samples: Samples per plant and record in the arms with the tools.

    Returns:
        v2 tools plus model on the two new draws and the original, v1 tools plus model on the two
        new draws, the raw-summary arm on the two new draws, then the v2 controls: 36 + 24 + 12 + 12
        calls at 6 plants and 2 samples.
    """
    jobs = [
        HindsightJob("tools_llm", record_id, plant, sample, "v2")
        for record_id in ROUND2_TOOLS_RECORDS
        for plant in range(plants)
        for sample in range(tools_samples)
    ]
    jobs += [
        HindsightJob("tools_llm", record_id, plant, sample, "v1")
        for record_id in ROUND2_V1_RECORDS
        for plant in range(plants)
        for sample in range(tools_samples)
    ]
    jobs += [
        HindsightJob("raw_llm", record_id, plant, 0, "v1")
        for record_id in ROUND2_V1_RECORDS
        for plant in range(plants)
    ]
    jobs += [
        HindsightJob("tools_llm", record_id, plant, 0, "v2")
        for record_id in ROUND2_CONTROL_RECORDS
        for plant in range(plants)
    ]
    return jobs


ROUND3_EASY_RECORDS: Final[tuple[RecordIdType, ...]] = ("easy01", "easy02")
"""The two easy-case records (post hoc): a 60% loss lasting 90 days at every plant."""

ROUND3_DRAW_RECORDS: Final[tuple[RecordIdType, ...]] = ("draw05", "draw06")
"""The fresh draws on which the amended step check is evaluated once (post hoc)."""


def round3_jobs(*, plants: int, samples: int = TOOLS_SAMPLES) -> list[HindsightJob]:
    """List round 3's calls in the order they run: the easy case, then the fresh draws.

    Args:
        plants: The number of plants.
        samples: Samples per plant and record.

    Returns:
        Round 2's tools-plus-model arm (version `v2`, frozen) on the two easy records and the two
        fresh draws: 24 + 24 calls at 6 plants and 2 samples. The original record's answers are
        round 2's, already cached.
    """
    return [
        HindsightJob("tools_llm", record_id, plant, sample, "v2")
        for record_id in (*ROUND3_EASY_RECORDS, *ROUND3_DRAW_RECORDS)
        for plant in range(plants)
        for sample in range(samples)
    ]


ROUND4_GRID_RECORDS: Final[tuple[RecordIdType, ...]] = ("grid01", "grid02", "grid03", "grid04")
"""The mid-difficulty grid: 30% for 90 days (`grid01`, `grid02`) and 60% for 30 days (the rest)."""

ROUND4_DRAW_RECORDS: Final[tuple[RecordIdType, ...]] = (
    "draw07",
    "draw08",
    "draw09",
    "draw10",
    "draw11",
    "draw12",
    "draw13",
    "draw14",
)
"""The eight further fresh draws, pooled with round 3's for the headline."""


def round4_jobs(*, plants: int, samples: int = TOOLS_SAMPLES) -> list[HindsightJob]:
    """List round 4's calls in the order they run: the grid, then the eight draws.

    Args:
        plants: The number of plants.
        samples: Samples per plant and record.

    Returns:
        Round 2's tools-plus-model arm (version `v2`, frozen) on the four grid records and then the
        eight draws: 48 + 96 calls at 6 plants and 2 samples.
    """
    return [
        HindsightJob("tools_llm", record_id, plant, sample, "v2")
        for record_id in (*ROUND4_GRID_RECORDS, *ROUND4_DRAW_RECORDS)
        for plant in range(plants)
        for sample in range(samples)
    ]
