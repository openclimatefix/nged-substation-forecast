# Can a language model, given deterministic evidence tools, label a solar plant's capacity changes?

Study for [issue #1147](https://github.com/openclimatefix/nged-substation-forecast/issues/1147), the
follow-up to the [first capacity
study](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/effective-capacity/).
The design is in `plans/study-llm-capacity-labeller-1147.md` in git history (pull request #1148), and
the results are on [the study
page](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/llm-capacity-labeller/).

**Every script takes `--smoke`, which reads and writes under `<study directory>/_smoke/`, answers
with a stub model, and is not a result.** `labeller_paths.claim_root` marks each root `real` or
`smoke` and raises if a run of the other kind points at it. A real call to the model needs
`--confirm-spend`, and the budget guard stops the run before a call that could reach US$15.

## The scripts

| Script | What it does |
|---|---|
| `labeller_paths.py`, `labeller_jobs.py` | Imported by the others: paths, stamps, loaders, and the list of calls with the way each prompt is built. |
| `build_labeller_inputs.py` | Draws the two fresh sets of planted losses (seeds 11441 and 11442), builds the synthetic null record, and reads the export-cap overlay. |
| `estimate_part1.py` | CPU. Tunes HYST2 and HM2 on the first study's tuning draw, runs CX2S at the first study's tuned setting, and scores all three on the fresh draws and the 23 quiet plant-months. |
| `probe_claude_cli.py` | The 2-call probe of `claude -p`. `--summarise` prints the saved results with no call. |
| `run_labeller.py` | `--phase estimate` estimates the cost from the built prompts. `--phase hindsight` and `--phase causal` make the calls, cached by the hash of the prompt, resumable, and logged under `data/`. |
| `score_labeller.py` | Reads the cached answers, applies the hard checks, and writes the planned contrasts, the controls, the event scores, and the capacity series the charts draw. Makes no call. |
| `build_round2_inputs.py`, `run_round2.py`, `score_round2.py` | **Round 2, post hoc.** Two new draws (seeds 11443 and 11444) and a non-adjacent positive control; the calls of the local-scan tools and the softer prompt, with the guard counting round 1's log and allowing US$6 more; the post hoc contrasts, the controls, and the model's quoted reasons for no change. |
| `build_round3_inputs.py`, `run_round3.py`, `score_round3.py` | **Round 3, post hoc.** Two fresh draws (05 and 06, seeds 11445 and 11446) and an easy case (a 60% loss for 90 days from a random day); the amended step check is evaluated once; `score_round3.py` also holds the free re-scores of rounds 1 and 2 (Wilson intervals, the placebo hit rate, hits by planted size, intervals that resample events and months, the noise floor, the powering, and the tally of the notes). |
| `fit_forecast_arms.py` | GPU. The exploratory forecast-level check on the lag study's rows. |
| `llm_capacity_labeller_charts.py` | Draws each estimator's capacity over a plant's daily maximum power in four figures of seven panels; `--round 2` draws round 2's four figures of eight panels. |
| `report_llm_capacity_labeller.py` | Prints every table the page quotes into `report.md`. |

## Run order

```bash
uv run python studies/llm_capacity_labeller/build_labeller_inputs.py
uv run python studies/llm_capacity_labeller/estimate_part1.py          # CPU, a few minutes
uv run python studies/llm_capacity_labeller/run_labeller.py --phase estimate
uv run python studies/llm_capacity_labeller/run_labeller.py --phase hindsight --confirm-spend
uv run python studies/llm_capacity_labeller/run_labeller.py --phase causal --confirm-spend
uv run python studies/llm_capacity_labeller/score_labeller.py
uv run python studies/llm_capacity_labeller/fit_forecast_arms.py --device cuda --max-workers 4
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py
uv run python studies/llm_capacity_labeller/report_llm_capacity_labeller.py
```

The smoke pipeline is the same commands with `--smoke` (and no `--confirm-spend`).

## What the files hold

All under `data/studies/per_study/llm_capacity_labeller/` (or its `_smoke/`):

- `built/`: `fresh_draws.json`, `null_power.npz`, `cap_overlay.npz`, and `manifest.json`.
- `part1/`: the tuning table, the chosen settings, the scores, and each estimator's capacity.
- `calls/` and `labeller_log.jsonl`: one file per call, and the log of every prompt and answer. **The
  log pairs plant letters with dates, so it stays under `data/` and is never committed.**
- `tables/`, `figures/`, and `report.md`.
