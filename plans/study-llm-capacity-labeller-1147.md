# Plan: can an LLM, given deterministic evidence tools, label a solar plant's capacity changes? (#1147)

**The problem.** The [first capacity study](../docs/studies/forecasts/effective-capacity.md) (issue
#1143) found that no daily capacity estimate lowered day-ahead error detectably at 6 solar plants,
that tuning on recovery chose agile estimators whose raw level carries a winter bias of the physics
model, and that HYST, a hold-at-static detector designed after the results, did least harm. The
study also had no ground truth for real capacity changes: only 3 of 106 plant-months are a real
low-output loss, all at one plant (`derated_plan.json`: 2024-12, 2025-03, 2026-04).

**The planned solution.** A new study with four parts. Part 1 reruns HYST, HM, and CX2 with the
seasonal correction built in. Part 2 builds six deterministic evidence tools that write a case file
per plant and period. Part 3 has Claude Sonnet 5.5 read the case files and write dated capacity
knots with a stated cause, as a hindsight labeller and as a causal estimator. Part 4 adds four hard
checks that reject an implausible estimate. Scoring uses blind planted losses on fresh draws,
spot checks on the real low months and the plant under active network management (ANM), and one
exploratory forecast-level check.

## Verdict and size

**Verdict: worth doing as the issue describes.** The maintainer fixed the four-part design.

**Size: complex.** The five trigger answers:

1. **What gets stored:** yes. A published page, checkpoint tables, a prompt and answer log, and new
   code in `packages/studies/`.
2. **Production serving path:** no. Only `studies/`, `packages/studies/`, and `docs/studies/` change.
3. **A degradation rule:** no. The sanity layer is a study prototype of a rule a production
   estimator would need and changes nothing in `inherent-stability.md`.
4. **More than one defensible design:** yes (how the model is called, how knots are scored, how Part
   1 is tuned, the causal cadence).
5. **Callers not nameable without searching:** no for new code; yes for the reused first-study
   machinery, so the plan names each reused symbol.

**Reviews bought:** both plan reviews, both Opus science reviews, a fresh review of every script
before its first run, both diff reviews, and a mutation pass because `packages/studies/` changes.
The diff stays inside the autonomous-merge list if no dependency changes.

**Departures from the issue:** fresh planted draws (the first study's 10 selection draws were seen
in full), and a causal run limited to the last 6 months of the record (call budget).

## Design

### Records, draws, and blindness

**The record is the first study's `CapacityRecord`** (`studies.capacity_estimators`; 6 plants, 2,495
days, 2019-09-01 to 2026-06-30 on the CAMS side, power from each plant's own start, read from
`built/record.npz`). **Planted losses use `capacity_recovery.draw_events` and `plant_losses`** with
the first study's shapes (step, ramp, intermittent), sizes (5% to 40%), one-month duration, and
footprint rules, so Part 1 stays comparable with the first page. **Two fresh draws** (seeds 11441
and 11442, never run) carry every planned contrast. The old draws re-score Part 1 only, labelled
exploratory.

**Blindness is structural.** The case-file builder takes a `CapacityRecord` and an issue day, and
has no event argument. The prompt never mentions planted losses. The labeller sees planted records
and the original record under the same anonymous labels.

**A knot series is scored with the first study's ratio.** The score is
`capacity_recovery.relative_capacity_error` of (the labeller's series on the planted record) over
(its series on the original record). The original-record run is also the real-data run, so the
"without" half costs no extra calls, and real changes and seasonal bias cancel as before.

### Part 1: seasonal correction built in

**Each estimator divides the daily physics-model ratio by a seasonal reference before detecting.**
The ratio is power over `cams_shape` output. The reference combines the other 5 plants' same-day
median ratio (the 6 plants share one 25 km box, so the winter bias is common) with the plant's own
profile from earlier years where one exists (`capacity_posthoc.deseasonalise_causally`). Both are
causal. The cross-plant reference would hide a loss hitting all 6 plants together, which the planted
losses (one plant per event) do not test; the page says so.

**HYST2 and HM2 are tuned, CX2S is not.** HYST2 (`hysteresis_level`) and HM2 (`estimate_hm`) are
cheap, so each setting on its grid minimises `recovery_error + 2 x false_movement_share` on the
tuning draw (draw 0, seed 11430, as before), subject to false movement at most
`FALSE_MOVEMENT_GATE` (0.10). The weight 2 is a plan choice. CX2S reruns the CX2 solver at the first
study's tuned settings on the corrected ratio, because the solver is the expensive part. Scoring
uses `score_events` and `false_movement_share` on the 23 quiet plant-months and the 2 fresh draws.

### Part 2: six evidence tools and the case file

**A case file is a deterministic function of (plant, period, issue day).** The causal run reads days
up to the issue day minus 2 (the first study's `ISSUE_LAG_DAYS`, which contains the 1-day CAMS
delay), exactly what the first study's issue-time scales saw.

| Tool | What it measures | Cause it separates | Causal timing |
|---|---|---|---|
| Clear-sky ratio, with data-quality masks | Daily mean power over `cams_shape` output on hours with clear-sky index above 0.8, excluding ANM-capped hours, gap days, stuck runs of 6 or more hours, and the commissioning ramp (`studies.commissioning`); masked days are listed as flags | any loss against none; data fault against physical change | through T-2 |
| Seasonal correction and neighbour ratio | The ratio minus the plant's earlier-year profile and minus the other plants' same-day median ratio, as two columns, plus the 7-day neighbour ratio (`capacity_estimators.neighbour_ratio`) | model bias and shared weather against a local loss | profile from earlier years; others through T-2 |
| Hour-of-day shape, with envelope and clipping columns | The corrected ratio by hour ending per month; per month the 95th-percentile power in 8 irradiance bins, the 99th percentile of power, and the share of hours within 2% of it | a morning or evening hole (failed string, shading) against flat soiling | through T-2 |
| ANM cap-change overlay | The cap in megawatts and the share of daylight hours where it binds, for the ANM plant; `none recorded` for the 5 plants NGED confirmed ran free | curtailment against capacity loss | through T-2 (the live service has the feed) |
| Changepoint candidates | CUSUM and a two-segment least-squares fit on the corrected ratio: date, direction, size, permutation p-value; a candidate within 14 days of the issue day is flagged unconfirmed | candidate knots | through T-2 |
| Tools-only labeller | A fixed rule that writes a knot at each changepoint with p below 0.01 and size above 8% and takes the cause from a decision table (ANM overlap, curtailment; a stuck-run flag, data fault; an hour-of-day hole, string failure; otherwise degradation) | the baseline for the ablation | as the others |

**The case file is text tables plus 2 small PNG charts** (Altair with `vl_convert`, both importable
in the study environment): the corrected ratio against date with changepoints and ANM cap days
marked, and the hour-of-day-by-month heatmap. No raw half-hourly or hourly data enter a prompt.
There is no envelope chart. **The envelope and clipping columns cannot be tested for a DC-side
against an AC-side loss with planted scalings, because a planted loss scales all of a day's power,
clipping plateau included.** Only the 3 real low months and the ANM plant can speak to that
distinction, and the page says so as a limitation.

### Part 3: the LLM

**Model: Claude Sonnet 5.5 (`claude-sonnet-5-5`), temperature 1.0.** A Fable or Opus labeller would
probably read a case file better; the page states that as a limitation only.

**Output schema** (JSON, validated by a Pydantic model that also enforces knots in strictly
increasing date order and inside the record):

```json
{"plant": "A", "period": ["2025-01-01", "2025-06-30"], "no_change": false,
 "knots": [{"date": "2025-03-04", "capacity_mw": 17.9,
            "cause": "string_failure|soiling|degradation|curtailment|model_bias|data_fault",
            "evidence": ["changepoint 2025-03-03 size -12% p=0.002"]}]}
```

**The default is no change**: the prompt says that an empty `knots` list is the expected answer for
most periods. Knots are linearly interpolated, held at the first knot's level before it and at the
last knot's after it; with no knots the capacity is the static 99th-percentile power (static P99).

**Two roles, both kept.**

- **Hindsight labeller:** one call per plant per record, with the whole-record monthly case file.
- **Causal estimator:** one call per plant per monthly issue date over the last 6 months of the
  record, on the original record and one planted draw (fresh draw 1), 1 sample. The prompt carries
  the previous call's knots as "your current estimate". The case file ends at the issue day minus 2.

**Arms** (hindsight, same records): tools only; tools plus LLM, 2 samples; LLM on raw daily
summaries (daily power and CAMS totals plus the data flags, no other tool output), 1 sample.

### Part 4: four hard checks (`capacity_sanity.py`)

**Policy, not a check:** an estimate that fails any hard check is replaced by the static P99, and
the series records the failing check. Knot order is enforced by the schema, not a check.

1. **Bounds:** 0.2 x static P99 <= capacity <= 1.3 x the fitted DC capacity
   (`built/orientation.json`; the 1.3 is a plan choice), and capacity above the fitted AC limit is
   clipped to it.
2. **Step within the evidence:** each step lies within 5 points or 40% of the size of a changepoint
   candidate within 14 days, whichever is larger; a step with no such candidate fails.
3. **Reproduce the power:** dividing the measured power by the estimate must not raise the variance
   of the normalised clear-sky ratio over the period.
4. **ANM consistency:** a knot whose cause is not `curtailment` must not sit inside a run of 3 or
   more cap-binding days, and a `curtailment` knot needs a cap-binding day within 14 days.

**The sanity layer is scored by the share of planted events lost to rejection** (a hit before the
checks that becomes a static-P99 series after them), reported beside the hit rate.

### Evaluation

**Planned contrasts (written before any result; thresholds are plan choices):**

- **C1: tools plus LLM against tools only,** paired difference in mean absolute capacity error on the
  2 fresh draws. The LLM adds value at -1.0 point or lower with the 97.5% event-bootstrap interval
  wholly below zero. (The first study's best recoverer beat doing nothing by 3.2 points.)
- **C2: tools plus LLM against the LLM on raw daily summaries,** same metric and threshold. This
  answers whether the tools matter.
- **C3: false movement on the 23 quiet plant-months,** tools plus LLM against the 10% gate (share of
  days the capacity is more than 5% from its 90-day median). HYST2 is the reference.

**Exploratory (labelled so):** the causal estimator against the best Part 1 estimator on the last 6
months; the forecast-level check; the real low months and the ANM plant; the 2-sample spread; hit,
false-alarm, date and size scores; the rejection share; the old-draw Part 1 re-score; per-cause
breakdown. Anything designed after a first result is post hoc.

**Intervals.** Stage A contrasts use a 97.5% bootstrap over events; events within a draw share
weather, so the interval covers which month, shape, and size were drawn and not differences between
plants. The forecast check resamples whole calendar months and one of 3 seeds
(`studies.bootstrap.bootstrap_difference`, 2,000 resamples).

**Controls.**

- **Negative:** the labeller reads a month-deranged record (`capacity_scaling.derange_months`); the
  right answer is no change, and the false-alarm count is the floor.
- **Positive:** the 30% step control (`CONTROL_SIZE`, `draw_control_steps`, 17 plant-months). If
  tools plus LLM finds fewer than 80% of them, C1 to C3 are reported as untested, not null.

**Scores:** hits (a knot within 14 days of the event start, correct sign, size within 50%), false
alarms, date error, size error, and capacity error. A date error is reported beside every hit rate,
because a 14-day window flatters a labeller against HYST's 7-day delay.

**Real data.** The LLM's hindsight labels are a reference for scoring KF, HYST, HM, CX2, and the Part
1 designs on the real record. The labels are an expert opinion and not ground truth, and they use
the same tools as the estimators, so agreement is not independent evidence. The 3 real low months and
the ANM plant appear as case-file figures with the labeller's knots. No figure puts a plant letter
next to the ANM plant, a curtailment date, or a low-output date.

**Forecast-level check (exploratory, no decision rule).** The hindsight knots become issue-time
scales (`capacity_estimators.scale_for_rows`) and run through the first study's Stage B machinery
(`capacity_scaling.scaled_frame`; B0's 7 columns; the 29,445 lag-study rows), at the primary XGBoost
setting only, scored in megawatts over the fixed static P99, after the constant-scale and
refit-reproduces-B0 assertions. The causal series covers 6 months and is compared with B0 on those
months.

## Auditability and leakage

**Every prompt and answer is logged** to `labeller_log.jsonl` (prompt, image hashes, model id,
sampling parameters, answer, token counts, timestamp). **Answers are cached** by the SHA-256 of the
(model id, system prompt, prompt, image bytes, sample index). **`--smoke`** uses a stub labeller and
no network, with the first study's `claim_root` and `RUN_KIND.txt` separation. **A causal probe**
replaces every input after the issue day minus 2 with noise and asserts that prompt text and image
bytes are unchanged, run once per record before any real call (the pattern is
`capacity_estimators.assert_scale_ignores_the_issue_day_onward`).

**How the model is called.** `claude -p --model claude-sonnet-5-5 --output-format json` (installed
at `/home/jack/.local/bin/claude`) or the Anthropic SDK run as `uv run --with anthropic`; neither
changes `pyproject.toml`, and `anthropic` is not in `uv.lock` (checked). Whether `claude -p` accepts
images and reports token counts is **unverified**; the first step is a 2-call probe.

## Code layout and tests

**Package modules take a `CapacityRecord` (already in `studies.capacity_estimators`) and need
nothing from the first study's folder.** `studies/effective_capacity/capacity_paths.py` stays where
it is, unchanged.

| Where | Holds |
|---|---|
| `packages/studies/src/studies/capacity_evidence.py` | the tools and case-file builder, with the issue-day cut |
| `packages/studies/src/studies/capacity_labeller.py` | knot schema and interpolation, the tools-only labeller, prompt builder, cache key, case-file text and charts |
| `packages/studies/src/studies/capacity_sanity.py` | the four checks and the static-P99 fallback |
| `packages/studies/src/studies/capacity_seasonal.py` | the seasonal reference and the HYST2, HM2, CX2S wrappers |
| `packages/studies/src/studies/labeller_scoring.py` | hits, false alarms, date and size error, event bootstrap |
| `studies/llm_capacity_labeller/labeller_paths.py` | paths, seeds, `load_record` and `claim_root` equivalents, the fingerprint helpers (small copies, not imports from the other study folder) |
| `studies/llm_capacity_labeller/` scripts | `build_draws.py`, `estimate_part1.py`, `build_case_files.py`, `run_labeller.py` (calls, cache, log, budget guard), `score_labeller.py`, `fit_forecast_arms.py`, `report_llm_capacity_labeller.py`, `llm_capacity_labeller_charts.py`, `README.md` |

**Tests** (under `packages/studies/tests/`, run with `--run-studies`; each fails on the bug named):

- **Blind builder:** no event argument in the signature; output on a planted record equals output on
  a copied record with identical power.
- **Causal probe per tool:** noise after the issue day minus 2 leaves each tool unchanged (fails on a
  centred rolling window).
- **Seasonal correction:** a December dip shared by all 6 plants corrects to within 2% of flat; a
  one-plant 25% step survives.
- **Clear-sky ratio:** a known 20% loss recovers 0.8 within 1%; cloudy-hour changes do not move it;
  a gap, a 6-hour stuck run, and a ramp are each masked and flagged.
- **Hour-of-day shape and clipping columns:** a morning-only loss and a flat loss give different
  residual rows; a lowered plateau changes the plateau column and a falling slope does not.
- **Changepoints:** a planted step is found within 3 days with direction and size; a flat noisy
  series gives p above 0.05; two steps give two candidates.
- **ANM overlay:** the binding share equals a hand count; the 5 free plants return `none recorded`.
- **Schema and interpolation:** linear between knots, constant outside, unordered or out-of-record
  knots rejected.
- **Cache key:** one changed prompt word, image byte, or sample index changes the key.
- **Each of the four checks,** with a failing and a passing input: bounds (including the AC clip),
  a 20% step against a 5% changepoint, an estimate that adds a spurious step (variance rises), and a
  degradation knot inside a cap run; plus the fallback records the failing check.
- **Scoring:** a hand-built knot series against a hand-built event; a wrong-sign knot is a miss and
  a false alarm.
- **Budget guard:** the runner raises before the 141st call and before 1.4 million input tokens.
- **Boundaries:** `test_study_boundaries.py` passes.
- **Mutation pass** on `capacity_evidence.py`, `capacity_sanity.py`, and `labeller_scoring.py`.

## Docs

- **`docs/studies/forecasts/llm-capacity-labeller.md`:** a figure-led page per the study skill, with
  a short section on what this iteration implies for the next iteration and for wind, linking back
  to the first page; the `mkdocs.yml` nav gains the entry.
- **`docs/studies/forecasts/effective-capacity.md`:** one forward link in its Discussion;
  **`docs/roadmap/capacity-estimation.md`:** a pointer in its status text.
- **`studies/llm_capacity_labeller/README.md`** and the `studies/README.md` index row.
- **Figure order:** Figure 1 the paired contrasts C1 to C3 with intervals; the corrected ratio at the
  6 plants before and after correction; one worked case file; the labeller's knots against the truth
  for 3 planted events; the controls; the ablation; the rejections; the exploratory causal and
  forecast checks last.

## Verification commands

From `.github/workflows/ci.yml` and `studies_tests.yml`:

```bash
uv run ruff check . && uv run ruff format --check .
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run ty check
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict --site-dir /tmp/ci-site
uv run python scripts/lint/check_docs_links.py
uv run pytest
uv run pytest -n auto --run-studies packages/studies
uv run pre-commit run --all-files --show-diff-on-failure
uv run python studies/llm_capacity_labeller/run_labeller.py --smoke
```

## Compute and token budget

**Calls: 138 planned, hard stop at 140.** Records: the original plus 2 fresh draws, so 3.

- Hindsight, tools plus LLM: 3 records x 6 plants x 2 samples = 36.
- Hindsight, raw-summary ablation: 3 x 6 x 1 = 18.
- Causal: 2 records x 6 plants x 6 monthly issue dates x 1 = 72.
- Controls: 1 deranged record x 6 plants + 1 positive-control record x 6 plants = 12.

A schema-invalid answer is retried once and the retry counts toward the 140.

**Tokens.** Hindsight with tools: about 10,000 input (tables 6,000, 2 charts 4,000, a chart-cost
estimate) and 1,500 output, for 36 + 12 calls. Raw ablation: about 5,000 input and 1,500 output, 18
calls. Causal: about 7,000 input and 1,000 output, 72 calls. The total is about 0.48M + 0.09M +
0.50M = **1.07M input and 0.16M output tokens, unverified until the probe measures real counts.** The
guard stops at 1.4M input and 0.25M output.

**Cost.** At an assumed US$3 per million input and US$15 per million output tokens (from memory, not
checked against current pricing), the planned run costs about **US$5.6** and the guard ceiling about
**US$8**. The stated ceiling is **US$10**, which needs the maintainer's approval before the first
real call; nothing beyond it is spent without a further decision.

**Compute slots** (from the Study MAIN COORDINATOR): 8 CPU cores briefly for Part 1 tuning and
scoring; 2 cores for case-file building; 1 core while the calls run; then GPU plus 4 CPU cores for
the exploratory forecast fits (3 seeds, 5 folds, primary setting only).

## Open decisions

1. **How to call the model:** `claude -p` if the probe shows images and token counts, otherwise
   `uv run --with anthropic`. Adding `anthropic` to `pyproject.toml` ends the autonomous-merge
   condition.
2. **Spend ceiling:** US$10 stated, about US$5.6 planned. Approval is needed before the first real
   call.
3. **Thresholds** (-1.0 point for C1 and C2, weight 2 for false movement, 1.3 x DC capacity, 14-day
   window) are plan choices. Recommendation: fix now and report sensitivity on the tuning draw only.
4. **Only 2 fresh draws (about 33 events)** give wide intervals, and the 6-month causal run is
   exploratory by design. Recommendation: accept, and say on the page that a null result is a finding
   about sample size.
5. **Planted losses last one month** while real ones persist. Recommendation: keep the first study's
   shapes, and add a persistent step as a post hoc arm only if budget remains.

## Review triage (simplicity review)

- (1) Causal role shrink: changed. Kept (maintainer wants both roles) but cut to 6 monthly issue
  dates, 6 plants, 1 sample, original plus 1 planted draw (72 calls); its contrast is exploratory.
- (2) Tools 8 to 6: changed. Neighbour ratio joined the seasonal tool; data-quality flags became
  masks in the clear-sky tool; the envelope chart was dropped and its numbers became columns of the
  hour-of-day tool; the DC/AC limitation is stated.
- (3) Sanity checks 9 to 4: accepted. Order moved to the schema, the policy is not a check;
  cross-sample agreement, the step-frequency prior, and the corrupted-series control are dropped;
  rejection share is scored.
- (4) Planned contrasts 5 to 3: accepted. The causal contrast and the forecast check are exploratory;
  the forecast check uses one setting.
- (5) `capacity_paths.py` helpers stay in the study folder: accepted. Package modules take a
  `CapacityRecord`; `labeller_paths.py` holds small equivalents; the earlier claim that the package
  needed those helpers was wrong and is removed.
- (6) Part 1 tuning: changed. HYST2 and HM2 are tuned with the false-movement penalty; CX2S runs at
  the first study's settings.
- (7) Fresh draws 4 to 2: accepted.
- (8) Samples 3 to 2 for tools plus LLM: accepted.
- (9) Prose cuts: accepted. Departures and restated risks dropped; only decision-carrying risks stay.
- (10) CI commands: accepted. Read from `.github/workflows/` and written out exactly.
