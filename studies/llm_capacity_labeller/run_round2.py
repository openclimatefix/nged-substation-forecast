"""Make round 2's calls: the budget estimate and the checkpointed batches (post hoc).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**Round 2 is post hoc.** Round 1 (`run_labeller.py`) ran as planned and stays unchanged. Round 2
adds a local scan as a seventh evidence table, softens the system prompt for a loss that lasts a
month, and is judged on two new draws of planted losses that nobody tuned on. It also re-runs
round 1's tools-plus-model arm on those draws, so that the two versions are paired.

**Budget.** The guard counts every call already in the log and gives round 2 at most `ROUND_USD`
(US$6) on top, inside the study's US$15 ceiling. A real run needs `--confirm-spend`; `--smoke` uses
the stub and spends nothing.

**Phases:** `estimate` (prompts built, uncached calls and cost estimated), `hindsight` (see
`labeller_jobs.round2_jobs`), `causal` (the original and the third draw, after the causal probe),
and `all`.

Run it with `uv run python studies/llm_capacity_labeller/run_round2.py --phase estimate`.
"""

import argparse
import json
import logging
import sys
from typing import Final

import numpy as np
from labeller_jobs import (
    CAUSAL_ISSUE_DAYS,
    ROUND2_CAUSAL_RECORDS,
    TOOLS_SAMPLES,
    LabellerContext,
    round2_jobs,
    run_causal_chain,
    run_hindsight_job,
)
from labeller_paths import calls_dir, claim_root, log_path, tables_dir, write_json_atomic
from run_labeller import (
    assert_causal_prompts_ignore_the_future,
    cli_version,
    make_answer_function,
)
from studies.capacity_labeller import cache_key
from studies.capacity_llm import (
    BudgetExceeded,
    BudgetGuard,
    CallStore,
    ClaudeCliModel,
    LabellerModel,
    StubModel,
    estimate_input_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_round2")

ROUND_USD: Final[float] = 6.0
"""The most round 2 may spend, on top of what round 1 spent."""

ROUND_CALLS: Final[int] = 200
"""The most calls round 2 may make, retries included."""

ROUND_INPUT_TOKENS: Final[int] = 2_500_000
"""The most input tokens round 2 may send."""

PRICE_PER_INPUT_TOKEN_USD: Final[float] = 4.0e-6
"""The probe's measured price of input."""

MEASURED_COST_PER_CALL_USD: Final[float] = 0.022
"""Round 1's measured cost per call (US$3.04 over 138 calls), the output share being small."""


def estimate(*, context: LabellerContext, store: CallStore, model_id: str) -> dict[str, object]:
    """Estimate round 2's uncached calls and cost from the prompts it would send.

    Args:
        context: The context.
        store: The call store, whose cached calls cost nothing.
        model_id: The model whose cache keys are checked.

    Returns:
        The estimate: calls, input tokens, and cost two ways (the probe's input price with an
        assumed 4,000 output tokens, and round 1's measured cost per call).
    """
    plants = len(context.sites)
    groups: dict[str, list[int]] = {}
    for job in round2_jobs(plants=plants):
        case = context.hindsight_case(
            record_id=job.record_id,
            plant=job.plant,
            kind="tools" if job.arm == "tools_llm" else "raw",
            version=job.version,
        )
        key = cache_key(model=model_id, prompt=case.prompt, sample_index=job.sample_index)
        if store.get(key=key) is None:
            name = f"hindsight/{job.version}/{job.arm}/{job.record_id}"
            groups.setdefault(name, []).append(estimate_input_tokens(prompt=case.prompt))
    for record_id in ROUND2_CAUSAL_RECORDS:
        for plant in range(plants):
            for issue_day in CAUSAL_ISSUE_DAYS:
                case = context.causal_case(
                    record_id=record_id,
                    plant=plant,
                    issue_day=issue_day,
                    current_knots=(),
                    version="v2",
                )
                groups.setdefault(f"causal/v2/{record_id}", []).append(
                    estimate_input_tokens(prompt=case.prompt)
                )
    calls = sum(len(tokens) for tokens in groups.values())
    tokens = sum(sum(tokens) for tokens in groups.values())
    return {
        "uncached_calls": calls,
        "input_tokens": tokens,
        "cost_usd_from_input_price_and_4000_output_tokens": round(
            tokens * PRICE_PER_INPUT_TOKEN_USD + calls * 4000 * 15.0e-6, 2
        ),
        "cost_usd_from_round1_measured_cost_per_call": round(calls * MEASURED_COST_PER_CALL_USD, 2),
        "round_allowance_usd": ROUND_USD,
        "groups": {
            key: {"calls": len(values), "mean_input_tokens": int(np.mean(values))}
            for key, values in groups.items()
        },
    }


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("estimate", "hindsight", "causal", "all"), required=True
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--tools-samples", type=int, default=TOOLS_SAMPLES)
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    context = LabellerContext(root=root)
    store = CallStore(directory=calls_dir(root=root), log_path=log_path(root=root))
    model_id = "stub" if arguments.smoke else ClaudeCliModel().model_id
    if arguments.phase == "estimate":
        result = estimate(context=context, store=store, model_id=model_id)
        write_json_atomic(
            path=tables_dir(root=root) / "round2_budget_estimate.json", payload=result
        )
        print(json.dumps(result, indent=2))
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
    guard = BudgetGuard.for_round(
        log_path=log_path(root=root),
        round_usd=ROUND_USD,
        round_calls=ROUND_CALLS,
        round_input_tokens=ROUND_INPUT_TOKENS,
    )
    _LOG.info(
        "the log already holds %d calls and US$%.2f; the ceiling is US$%.2f",
        guard.calls,
        guard.spent_usd,
        guard.ceiling_usd,
    )
    answer_function = make_answer_function(store=store, model=model, guard=guard, lookup=lookup)
    write_json_atomic(
        path=root / "run_manifest_round2.json",
        payload={
            "model": model.model_id,
            "cli_version": None if arguments.smoke else cli_version(),
            "smoke": arguments.smoke,
        },
    )
    try:
        if arguments.phase in ("hindsight", "all"):
            jobs = round2_jobs(plants=len(context.sites), tools_samples=arguments.tools_samples)
            for index, job in enumerate(jobs, start=1):
                run_hindsight_job(context=context, job=job, answer_function=answer_function)
                _LOG.info("%d/%d %s (spent US$%.2f)", index, len(jobs), job.label, guard.spent_usd)
        if arguments.phase in ("causal", "all"):
            for record_id in ROUND2_CAUSAL_RECORDS:
                assert_causal_prompts_ignore_the_future(
                    context=context, record_id=record_id, version="v2"
                )
            for record_id in ROUND2_CAUSAL_RECORDS:
                for plant in range(len(context.sites)):
                    run_causal_chain(
                        context=context,
                        record_id=record_id,
                        plant=plant,
                        answer_function=answer_function,
                        version="v2",
                    )
                    _LOG.info(
                        "causal %s plant %d done (spent US$%.2f)", record_id, plant, guard.spent_usd
                    )
    except BudgetExceeded as error:
        _LOG.error("stopped by the budget guard: %s", error)
        return 2
    _LOG.info("finished: %d calls in all, US$%.2f in all", guard.calls, guard.spent_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
