"""Make the labeller's calls: the budget estimate, the causal probe, and the checkpointed batches.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**A real run spends money, so it needs `--confirm-spend`, and `--smoke` never spends any.** A smoke
run writes under `_smoke/`, answers every prompt with a stub (the tools-only rule, in the model's
answer format) and touches no network. A real run calls `claude -p` (see `studies.capacity_llm`).

**Phases:**

- `estimate`: builds every prompt, estimates the input tokens, the calls, and the cost from them
  and from the probe's measured price, and writes `tables/budget_estimate.json`. It exits with an
  error if the estimate exceeds the ceiling. The cut order is: skip the causal role (do not run
  `--phase causal`), then run `--tools-samples 1`.
- `hindsight`: the hindsight calls and the controls (see `labeller_jobs`), in the order that puts
  the planned contrasts first.
- `causal`: the six monthly issue dates of each plant on the original record and the first fresh
  draw. It runs after the causal probe has passed on both records.

**Every call is cached by the hash of its prompt, so a stopped run resumes where it stopped, and
every prompt and answer is appended to `labeller_log.jsonl` under `data/`.** The budget guard starts
from the cost already in the log, so a resumed run cannot overspend, and a stop on the guard exits
with status 2.

Run it with `uv run python studies/llm_capacity_labeller/run_labeller.py --phase estimate`, then
`--phase hindsight --confirm-spend`, then `--phase causal --confirm-spend`.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from typing import Final

import numpy as np
from labeller_jobs import (
    CAUSAL_ISSUE_DAYS,
    CAUSAL_PERIOD_FIRST,
    CAUSAL_RECORDS,
    CAUSAL_SHOWN_MONTHS,
    READABLE_LAG_DAYS,
    TOOLS_SAMPLES,
    AnswerFunction,
    LabellerContext,
    VersionType,
    hindsight_jobs,
    run_causal_chain,
    run_hindsight_job,
)
from labeller_paths import (
    RecordIdType,
    calls_dir,
    claim_root,
    log_path,
    tables_dir,
    write_json_atomic,
)
from studies.capacity_estimators import scramble_from_day
from studies.capacity_evidence import CapOverlay, Evidence, PlantFit, build_evidence
from studies.capacity_labeller import (
    SYSTEM_PROMPT_V2,
    Answer,
    Prompt,
    answer_to_json,
    case_file_text,
    chart_pngs,
    tools_only_answer,
    tools_only_v2_answer,
)
from studies.capacity_llm import (
    COST_CEILING_USD,
    MAX_CALLS,
    BudgetExceeded,
    BudgetGuard,
    CallStore,
    ClaudeCliModel,
    LabellerModel,
    StubModel,
    estimate_input_tokens,
    labelled_answer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_labeller")

PRICE_PER_INPUT_TOKEN_USD: Final[float] = 4.0e-6
"""The probe measured about US$4 per million input tokens (the command-line tool bills a prompt as
cache-creation input)."""

PRICE_PER_OUTPUT_TOKEN_USD: Final[float] = 15.0e-6
"""An assumed output price, on the high side of the probe's fit (about US$8 per million)."""

OUTPUT_TOKENS_PER_CALL: Final[int] = 4000
"""The output tokens assumed per call, thinking included."""

PROBE_SEED: Final[int] = 11470
"""The seed of the noise the causal probe writes after the cut-off day."""


def estimate_budget(
    *, context: LabellerContext, tools_samples: int, with_causal: bool
) -> dict[str, object]:
    """Estimate the calls, tokens, and cost of the run from the prompts it would send.

    Args:
        context: The context.
        tools_samples: Samples per plant and record in the arm with the tools.
        with_causal: Whether the causal role is included.

    Returns:
        The estimate, by group, and in total.
    """
    plants = len(context.sites)
    groups: dict[str, list[int]] = {}
    for job in hindsight_jobs(plants=plants, tools_samples=tools_samples):
        case = context.hindsight_case(
            record_id=job.record_id,
            plant=job.plant,
            kind="tools" if job.arm == "tools_llm" else "raw",
        )
        key = f"hindsight/{job.arm}/{job.record_id}"
        groups.setdefault(key, []).append(estimate_input_tokens(prompt=case.prompt))
    if with_causal:
        for record_id in CAUSAL_RECORDS:
            for plant in range(plants):
                for issue_day in CAUSAL_ISSUE_DAYS:
                    case = context.causal_case(
                        record_id=record_id, plant=plant, issue_day=issue_day, current_knots=()
                    )
                    groups.setdefault(f"causal/{record_id}", []).append(
                        estimate_input_tokens(prompt=case.prompt)
                    )
    calls = sum(len(tokens) for tokens in groups.values())
    input_tokens = sum(sum(tokens) for tokens in groups.values())
    cost = (
        input_tokens * PRICE_PER_INPUT_TOKEN_USD
        + calls * OUTPUT_TOKENS_PER_CALL * PRICE_PER_OUTPUT_TOKEN_USD
    )
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens_assumed": calls * OUTPUT_TOKENS_PER_CALL,
        "cost_usd": round(cost, 2),
        "ceiling_usd": COST_CEILING_USD,
        "call_cap": MAX_CALLS,
        "tools_samples": tools_samples,
        "with_causal": with_causal,
        "groups": {
            key: {
                "calls": len(tokens),
                "mean_input_tokens": int(np.mean(tokens)),
                "max_input_tokens": max(tokens),
            }
            for key, tokens in groups.items()
        },
    }


def assert_causal_prompts_ignore_the_future(
    *, context: LabellerContext, record_id: RecordIdType, version: VersionType = "v1"
) -> None:
    """Raise if any causal prompt changes when every input after its cut-off day is noise.

    The probe replaces power, satellite irradiance, modelled output, and the export-cap overlay
    after the cut-off day with noise, and replaces each plant's static power and fitted quantities
    (which read the whole record) too, then compares the prompt text and the bytes of both charts,
    which covers the charts' axis domains.

    Args:
        context: The context.
        record_id: The record to probe.
        version: The round whose prompts are probed.

    Raises:
        ValueError: Naming the plant, issue date, and what changed.
    """
    record = context.record(record_id=record_id)
    rng = np.random.default_rng(PROBE_SEED)
    for plant in sorted({0, context.first_events.anm_plant, len(context.sites) - 1}):
        for issue_day in CAUSAL_ISSUE_DAYS:
            last_index = context.day_index(day=issue_day) - READABLE_LAG_DAYS
            kept = context.causal_case(
                record_id=record_id,
                plant=plant,
                issue_day=issue_day,
                current_knots=(),
                version=version,
            )
            noisy = scramble_from_day(
                record=record,
                first_day=last_index + 1,
                seed=PROBE_SEED + plant,
                scramble_static=True,
            )
            fit = context.fits[plant]
            noisy_fit = PlantFit(
                ac_capacity_mw=fit.ac_capacity_mw * 2,
                dc_capacity_mw=fit.dc_capacity_mw * 3,
                tilt_deg=fit.tilt_deg + 7,
                azimuth_deg=fit.azimuth_deg + 11,
            )
            cap = context.caps.get(plant)
            noisy_cap = None
            if cap is not None:
                later = np.arange(cap.cap_mw.shape[0]) > last_index
                noisy_cap = CapOverlay(
                    cap_mw=np.where(
                        later[:, None], rng.uniform(0, 9, cap.cap_mw.shape), cap.cap_mw
                    ),
                    binding=np.where(
                        later[:, None], rng.random(cap.binding.shape) > 0.5, cap.binding
                    ),
                )
            evidence = build_evidence(
                record=noisy,
                plant=plant,
                last_day=last_index,
                fit=noisy_fit,
                cap=noisy_cap,
            )
            text = case_file_text(
                evidence=evidence,
                kind="tools",
                shown_months=CAUSAL_SHOWN_MONTHS,
                current_knots=(),
                issue_day=issue_day,
                period_first=CAUSAL_PERIOD_FIRST,
                local_scan=version == "v2",
            )
            images = chart_pngs(evidence=evidence, shown_months=CAUSAL_SHOWN_MONTHS)
            if text != kept.prompt.text or images != kept.prompt.images:
                msg = (
                    f"the causal prompt for plant {context.sites[plant]} on {issue_day} changed "
                    "when inputs after its cut-off day were replaced by noise"
                )
                raise ValueError(msg)
    _LOG.info("causal probe passed on %s", record_id)


def make_answer_function(
    *, store: CallStore, model: LabellerModel, guard: BudgetGuard, lookup: dict[str, str] | None
) -> AnswerFunction:
    """Build the function that turns a case file into a validated answer through the cache.

    Args:
        store: The call store.
        model: The model.
        guard: The budget guard.
        lookup: For a stub run, the dictionary the stub answers from; each prompt's tools-only
            answer is added to it before the call. `None` for a real run.

    Returns:
        The answer function (see `labeller_jobs.AnswerFunction`).
    """

    def answer(
        *,
        prompt: Prompt,
        sample_index: int,
        label: str,
        plant: str,
        first_day: date,
        last_day: date,
        evidence: Evidence,
    ) -> Answer:
        if lookup is not None:
            rule = tools_only_v2_answer if prompt.system == SYSTEM_PROMPT_V2 else tools_only_answer
            lookup[prompt.text] = answer_to_json(
                answer=rule(evidence=evidence, first_day=first_day, last_day=last_day)
            )
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

    return answer


def cli_version() -> str:
    """Return the command-line tool's version string."""
    done = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False)
    return done.stdout.strip()


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("estimate", "hindsight", "causal", "all"), required=True
    )
    parser.add_argument("--smoke", action="store_true", help="Stub model under _smoke/; no spend.")
    parser.add_argument("--confirm-spend", action="store_true", help="Required for a real run.")
    parser.add_argument("--tools-samples", type=int, default=TOOLS_SAMPLES)
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    context = LabellerContext(root=root)
    with_causal = arguments.phase in ("causal", "all", "estimate")
    if arguments.phase == "estimate":
        estimate = estimate_budget(
            context=context, tools_samples=arguments.tools_samples, with_causal=True
        )
        write_json_atomic(path=tables_dir(root=root) / "budget_estimate.json", payload=estimate)
        print(json.dumps(estimate, indent=2))
        cost = float(str(estimate["cost_usd"]))
        if cost > COST_CEILING_USD:
            _LOG.error(
                "estimated cost US$%.2f is over the US$%.2f ceiling; cut the causal role first",
                cost,
                COST_CEILING_USD,
            )
            return 1
        return 0
    if not arguments.smoke and not arguments.confirm_spend:
        _LOG.error("a real run spends money; pass --confirm-spend (or --smoke for a stub run)")
        return 1
    lookup: dict[str, str] | None = {} if arguments.smoke else None
    model: LabellerModel = (
        StubModel(respond=lambda prompt: (lookup or {})[prompt.text])
        if arguments.smoke
        else ClaudeCliModel()
    )
    store = CallStore(directory=calls_dir(root=root), log_path=log_path(root=root))
    guard = BudgetGuard.from_log(log_path=log_path(root=root))
    answer_function = make_answer_function(store=store, model=model, guard=guard, lookup=lookup)
    write_json_atomic(
        path=root / "run_manifest.json",
        payload={
            "model": model.model_id,
            "cli_version": None if arguments.smoke else cli_version(),
            "smoke": arguments.smoke,
        },
    )
    try:
        if arguments.phase in ("hindsight", "all"):
            jobs = hindsight_jobs(plants=len(context.sites), tools_samples=arguments.tools_samples)
            for index, job in enumerate(jobs, start=1):
                run_hindsight_job(context=context, job=job, answer_function=answer_function)
                _LOG.info("%d/%d %s (spent US$%.2f)", index, len(jobs), job.label, guard.spent_usd)
        if with_causal:
            for record_id in CAUSAL_RECORDS:
                assert_causal_prompts_ignore_the_future(context=context, record_id=record_id)
            for record_id in CAUSAL_RECORDS:
                for plant in range(len(context.sites)):
                    run_causal_chain(
                        context=context,
                        record_id=record_id,
                        plant=plant,
                        answer_function=answer_function,
                    )
                    _LOG.info(
                        "causal %s plant %d done (spent US$%.2f)", record_id, plant, guard.spent_usd
                    )
    except BudgetExceeded as error:
        _LOG.error("stopped by the budget guard: %s", error)
        return 2
    _LOG.info("finished: %d calls, US$%.2f", guard.calls, guard.spent_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
