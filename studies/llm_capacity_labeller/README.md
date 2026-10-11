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
| `build_round4_inputs.py`, `run_round4.py` | **Round 4, post hoc.** A mid-difficulty grid (30% for 90 days, 60% for 30 days) and eight further fresh draws (07 to 14), run with round 2's frozen arm and the amended check; the guard counts earlier rounds, projects each call at a US$0.25 cap, and stopped the run during draw 13. |
| `fit_forecast_arms.py` | GPU. The exploratory forecast-level check on the lag study's rows. |
| `llm_capacity_labeller_charts.py` | Draws each estimator's capacity over a plant's daily maximum power in four figures of seven panels; `--round 2` draws round 2's four figures of eight panels. |
| `report_llm_capacity_labeller.py` | Prints every table the page quotes into `report.md`. |

## Design details

The page keeps its text short, so the details live here.

**Records and planted losses.** The record is the first study's: 6 solar plants, 2,495 days, power
from each plant's own start. A planted loss multiplies one plant's power for one calendar month by
a step, a ramp, or an intermittent pattern of 5% to 40%, with the first study's footprint rules.
Draws 01 and 02 (seeds 11441 and 11442) carry the planned round; draws 03 and 04 (11443 and 11444)
round 2; draws 05 and 06 (11445 and 11446) round 3; draws 07 to 14 (11447 to 11454) round 4. The
positive control plants 17 steps of 30% (round 1) or 18 non-adjacent steps (round 2). The easy case
(60% for 90 days) and the grid (30% for 90 days, 60% for 30 days) start on random days.

**The six tools and the case file.** The clear-sky ratio with data-quality masks; the seasonal
correction (the other plants' same-day median plus the plant's own day-of-year profile from earlier
years) with the 7-day neighbour ratio; the hour-of-day shape with the power envelope; the export-cap
overlay for the one plant that has one; changepoint candidates with permutation p-values; and a
tools-only rule that writes a knot at each strong changepoint. Round 2 adds a seventh table, the
local scan (30-day windows against the 30 days either side, with a 15% threshold). A case file is
text tables plus two small charts. It is a function of one plant's record up to a cut-off day, and
the builder has no argument that names a planted event.

**The model's answer.** A knot is a date, a capacity in megawatts, and one of six causes. Capacity is
linear between knots and constant outside them, so a step is two knots one day apart, and an empty
list means the static power (the plant's 99th-percentile output). The hindsight role reads the whole
record once per plant. The causal role reads the record up to two days before each of six monthly
issue dates. A causal case file prints the static power of the readings up to the cut-off and omits
the plant's fitted tilt, azimuth, and capacity, because those were fitted on the whole record.

**The four hard checks.** Bounds (0.2 to 1.1 times the static power); a step that a changepoint
candidate supports (round 3's amended check also accepts a scan episode starting or ending within 14
days, with no size test); a variance of the corrected ratio that the estimate does not raise; and
consistency with the export cap. A causal estimate that fails any check becomes the static power; a
hindsight estimate loses only the failing knots.

**The planned contrasts.** C1 (tools plus model against tools only) and C2 (against the model on
weekly summaries) are mean absolute capacity errors in points with 97.5% bootstrap intervals; the
model adds value at −1.0 point with the interval wholly below zero. C3 is false movement on the 23
quiet plant-months against a 10% gate. If the positive control finds under 80%, C1 to C3 are
untested. The planned bootstrap resamples (event, sample) pairs; the post hoc intervals resample
events and calendar months.

**The model route and the spend guard.** `claude -p` runs from an empty temporary directory with
every tool disabled, no settings, memory, skills, or MCP servers, and a fixed system prompt. It needs
no API key and has no temperature flag. A two-call probe cost US$0.054 (an image read correctly; an
18,016-character numeric table cost 12,559 input tokens). The guard counts every logged call,
including failures, and projects each next call at a per-call cap or the largest cost seen. **The cap
was US$1 for rounds 1 to 3 and US$0.25 for round 4,** because the dearest of the 342 earlier calls
cost US$0.07 and a US$1 cap would have stopped round 4's US$4 allowance after US$3. Round 4 stopped
during draw 13 (the plan pooled draws 05 to 14), and the partial draw 13 answers are excluded.

## Run order

```bash
uv run python studies/llm_capacity_labeller/build_labeller_inputs.py
uv run python studies/llm_capacity_labeller/estimate_part1.py          # CPU, a few minutes
uv run python studies/llm_capacity_labeller/run_labeller.py --phase estimate
uv run python studies/llm_capacity_labeller/run_labeller.py --phase hindsight --confirm-spend
uv run python studies/llm_capacity_labeller/run_labeller.py --phase causal --confirm-spend
uv run python studies/llm_capacity_labeller/score_labeller.py
uv run python studies/llm_capacity_labeller/build_round2_inputs.py
uv run python studies/llm_capacity_labeller/run_round2.py --phase all --confirm-spend
uv run python studies/llm_capacity_labeller/score_round2.py
uv run python studies/llm_capacity_labeller/build_round3_inputs.py
uv run python studies/llm_capacity_labeller/run_round3.py --phase run --confirm-spend
uv run python studies/llm_capacity_labeller/build_round4_inputs.py
uv run python studies/llm_capacity_labeller/run_round4.py --phase run --confirm-spend
uv run python studies/llm_capacity_labeller/score_round3.py
uv run python studies/llm_capacity_labeller/fit_forecast_arms.py --device cuda --max-workers 4
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py --round 2
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
