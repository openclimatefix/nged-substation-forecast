"""Make round 3's calls: the easy case and the amended-check draws (post hoc).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
Round 3 freezes round 2's tools, prompt, and samples. It adds (a) an easy case, a 60% loss for 90
days at every plant from a random day, to see whether the model follows a clear signal, and (b) two
fresh draws, 05 and 06, on which the amended step check is evaluated once. **The amended check was
written down before this script ran** (`studies.capacity_sanity.check_steps`, with its tests).

**Budget.** The guard counts every call already in the log and allows `ROUND_USD` (US$3) more,
inside the US$15 ceiling. A real run needs `--confirm-spend`; `--smoke` uses the stub.

Run it with `uv run python studies/llm_capacity_labeller/run_round3.py --phase estimate`, then
`--phase run --confirm-spend`.
"""

import argparse
import json
import logging
import sys
from typing import Final

import numpy as np
from labeller_jobs import LabellerContext, round3_jobs, run_hindsight_job
from labeller_paths import calls_dir, claim_root, log_path, tables_dir, write_json_atomic
from run_labeller import cli_version, make_answer_function
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
_LOG: Final[logging.Logger] = logging.getLogger("run_round3")

ROUND_USD: Final[float] = 3.0
"""The most round 3 may spend, on top of what rounds 1 and 2 spent."""

ROUND_CALLS: Final[int] = 80
ROUND_INPUT_TOKENS: Final[int] = 1_200_000
"""The most calls and input tokens round 3 may use."""

MEASURED_COST_PER_CALL_USD: Final[float] = 0.03
"""Round 2's measured cost per call (US$3.93 over 156 calls, about 0.025)."""


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("estimate", "run"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    context = LabellerContext(root=root)
    store = CallStore(directory=calls_dir(root=root), log_path=log_path(root=root))
    model_id = "stub" if arguments.smoke else ClaudeCliModel().model_id
    jobs = round3_jobs(plants=len(context.sites))
    if arguments.phase == "estimate":
        tokens = []
        for job in jobs:
            case = context.hindsight_case(
                record_id=job.record_id, plant=job.plant, kind="tools", version="v2"
            )
            key = cache_key(model=model_id, prompt=case.prompt, sample_index=job.sample_index)
            if store.get(key=key) is None:
                tokens.append(estimate_input_tokens(prompt=case.prompt))
        result = {
            "uncached_calls": len(tokens),
            "mean_input_tokens": int(np.mean(tokens)) if tokens else 0,
            "cost_usd_from_measured_cost_per_call": round(
                len(tokens) * MEASURED_COST_PER_CALL_USD, 2
            ),
            "round_allowance_usd": ROUND_USD,
        }
        write_json_atomic(
            path=tables_dir(root=root) / "round3_budget_estimate.json", payload=result
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
        "the log holds %d calls and US$%.2f; the ceiling is US$%.2f",
        guard.calls,
        guard.spent_usd,
        guard.ceiling_usd,
    )
    answer_function = make_answer_function(store=store, model=model, guard=guard, lookup=lookup)
    write_json_atomic(
        path=root / "run_manifest_round3.json",
        payload={
            "model": model.model_id,
            "cli_version": None if arguments.smoke else cli_version(),
            "smoke": arguments.smoke,
        },
    )
    try:
        for index, job in enumerate(jobs, start=1):
            run_hindsight_job(context=context, job=job, answer_function=answer_function)
            _LOG.info("%d/%d %s (spent US$%.2f)", index, len(jobs), job.label, guard.spent_usd)
    except BudgetExceeded as error:
        _LOG.error("stopped by the budget guard: %s", error)
        return 2
    _LOG.info("finished: %d calls in all, US$%.2f in all", guard.calls, guard.spent_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
