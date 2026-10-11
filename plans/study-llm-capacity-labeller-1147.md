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

## Design

### Records, draws, and blindness

**The record is the first study's `CapacityRecord`** (`studies.capacity_estimators`; 6 plants, 2,495
days, 2019-09-01 to 2026-06-30 on the CAMS side, power from each plant's own start, read from
`built/record.npz`). **Planted losses use `capacity_recovery.draw_events` and `plant_losses`** with
the first study's shapes (step, ramp, intermittent), sizes (5% to 40%), one-month duration, and
footprint rules, so Part 1 stays comparable with the first page. **Two fresh draws** (seeds 11441
and 11442, never run) carry every planned contrast. The old draws re-score Part 1 only, labelled
exploratory.

**Blindness is structural.** The case-file builder takes a `CapacityRecord`, an issue day, the ANM cap
array (from `studies.export_cap`), and `orientation.json`'s fitted quantities as explicit arguments,
and has no event argument. (`CapacityRecord` carries no cap, and its `power_mw` already has
export-capped hours set to NaN, so curtailment is nearly invisible in it; the ANM-plant spot check
therefore runs on the `power_anm_blind` record that `capacity_paths.save_record` writes.) The prompt never mentions planted losses. The labeller sees planted records
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
cheap, so each is tuned on the tuning draw (draw 0, seed 11430, as before) by lexicographic
selection: minimise recovery error among the settings whose false movement is at most
`FALSE_MOVEMENT_GATE` (0.10). A weighted sum is avoided because recovery error (about 0.02 to 0.05)
and a 0-to-1 share have different units; the report also shows the choice at false-movement weights
0 and 0.5 on the tuning draw. CX2S reruns the CX2 solver at the first
study's tuned settings on the corrected ratio, because the solver is the expensive part. Scoring
uses `score_events` and `false_movement_share` on the 23 quiet plant-months and the 2 fresh draws.

### Part 2: six evidence tools and the case file

**A case file is a deterministic function of (plant, period, issue day).** The causal run reads days
up to the issue day minus 2 (the first study's `ISSUE_LAG_DAYS`, which contains the 1-day CAMS
delay), exactly what the first study's issue-time scales saw.

| Tool | What it measures | Cause it separates | Causal timing |
|---|---|---|---|
| Clear-sky ratio, with data-quality masks | Daily mean power over `cams_shape` output on hours with clear-sky index above 0.8, excluding gap days, stuck runs of 6 or more hours, and the commissioning ramp (`studies.commissioning`); masked days are listed as flags | any loss against none; data fault against physical change | through T-2 |
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

**Model: Claude Sonnet 5.5 (`claude-sonnet-5-5`), at the command-line tool's default sampling.** A
Fable or Opus labeller would probably read a case file better; the page states that as a limitation
only. `claude -p` has no temperature flag, so every arm shares the default noise level, and repeated
samples differ because the model samples (the sample index is part of the cache key, so a second
sample is a second call and not a cache hit). The log records the exact model name, the flags, and
the command-line tool's version.

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

**Leakage limits the page states.** Real calendar dates admit the model's pretraining knowledge of
UK weather (CAMS normalisation mutes it, but does not remove it). CAMS errors are shared by the
labeller and the estimators, so the hindsight reference cannot score `model_bias` (the physics
model's own errors look like capacity to both). The causal probe also covers chart axis domains
computed from the whole series.

**Arms** (hindsight, same records): tools only; tools plus LLM, 2 samples; LLM on raw daily
summaries (daily power and CAMS totals plus the data flags, no other tool output), 1 sample.

### Part 4: four hard checks (`capacity_sanity.py`)

**Policy, not a check:** a causal estimate that fails any hard check is replaced by the static P99
and the series records the failing check. **A hindsight estimate is rejected per knot** (the
failing knot is dropped and recorded), so hits are not correlated within a plant by a whole-series
fallback. Knot order is enforced by the schema, not a check.

1. **Bounds:** 0.2 x static P99 <= capacity <= 1.1 x static P99 (the 1.1 is a plan choice). There is
   no AC clip: the static P99 exceeds the fitted AC capacity at four plants (for one, 5.95 against
   5.71 MW), so a clip would bias the no-change default and every ratio, and a bound of 1.3 x the
   fitted DC capacity could never bind.
2. **Step within the evidence:** a step is the difference between consecutive knots' levels. It
   passes if |LLM step - nearest changepoint candidate within 14 days| <= max(5 points, 0.4 x
   |candidate|); a step with no candidate fails.
3. **Reproduce the power:** on the scored window, the variance of the normalised clear-sky ratio
   after dividing by the estimate must be at most (ties pass) the variance without it. An empty knot
   list passes trivially, which the page says.
4. **ANM consistency:** a knot whose cause is not `curtailment` must not sit inside a run of 3 or
   more cap-binding days, and a `curtailment` knot needs a cap-binding day within 14 days.

**Check 2 stops the LLM from beating its own tools** (a step with no candidate fails, and ramps and
intermittent losses are not representable as candidates). C1 and C2 are therefore scored on the
answers **before** the checks (primary) and after them (secondary). The share of planted events
lost to rejection is reported beside the hit rate.

### Evaluation

**Planned contrasts (written before any result; thresholds are plan choices):**

- **C1: tools plus LLM against tools only,** paired difference in mean absolute capacity error on the
  2 fresh draws. Sample i on the planted record is paired with sample i on the original record, and
  the bootstrap resamples events and the sample index together. A noise floor is reported: the
  error between the 2 samples on the original record over the same scored days. The LLM adds value
  at -1.0 point or lower with the interval wholly below zero. (The first study's best recoverer
  beat doing nothing by 3.2 points.)
- **C2: tools plus LLM against the LLM on raw daily summaries,** same metric and threshold. This
  answers whether the tools matter.
- **C3: false movement on the 23 quiet plant-months,** tools plus LLM against the 10% gate (share of
  days the capacity is more than 5% from its 90-day median), with an interval from resampling the 23
  plant-months. HYST2 is the reference. C3 is read together with C1, because a labeller that never
  writes a knot scores zero false movement.

**Exploratory (labelled so):** the causal estimator against the best Part 1 estimator on the last 6
months; the forecast-level check; the real low months and the ANM plant; the 2-sample spread; hit,
false-alarm, date and size scores; the rejection share; the old-draw Part 1 re-score; per-cause
breakdown. Anything designed after a first result is post hoc.

**Multiplicity:** Bonferroni over C1 and C2 (the 97.5% level); C3 is descriptive.

**Intervals.** Stage A contrasts use a 97.5% bootstrap over events; events within a draw share
weather, so the interval covers which month, shape, and size were drawn and not differences between
plants. The forecast check resamples whole calendar months and one of 3 seeds
(`studies.bootstrap.bootstrap_difference`, 2,000 resamples).

**Controls.**

- **Negative:** a synthetic null record: each plant's power is its median ratio times `cams_shape`,
  plus residuals resampled from the 23 quiet plant-months. The right answer is no change, and the
  false-alarm count is the floor. (`capacity_scaling.derange_months` returns a donor map for forecast
  scales and does not derange a record, so it is not used.)
- **Positive:** the 30% step control (`draw_control_steps`, 17 plant-months; `CONTROL_SIZE` is 0.30
  in `studies/effective_capacity/capacity_paths.py`, copied into `labeller_paths.py` with the source
  stated, not imported). If
  tools plus LLM finds fewer than 80% of them, C1 to C3 are reported as untested, not null.

**Scores.** Every planted one-month loss needs a drop knot within 14 days of its start with the
correct sign and a size within 50%, and a recovery knot within 14 days of its end. Size is defined
per shape: a ramp starts at 0 and the size is its deepest point; an intermittent loss uses the
window-mean factor. A false alarm is a knot with no planted event and no counterpart (within 14 days,
same sample) in the original-record run, so the real low months are not false alarms. Date error,
size error, and capacity error are also reported. A date error is reported beside every hit rate,
because a 14-day window flatters a labeller against HYST's 7-day delay.

**Real data.** The LLM's hindsight labels are a reference for scoring KF, HYST, HM, CX2, and the Part 1 designs on the real record; they are an expert opinion and not ground truth, and share the tools' inputs. The 3 real low months and the ANM plant (on the `power_anm_blind` record) appear as case-file figures. No figure puts a plant letter next to the ANM plant, a curtailment date, or a low-output date.

**Forecast-level check (exploratory, no decision rule).** The hindsight knots become issue-time
scales (`capacity_estimators.scale_for_rows`) and run through the first study's Stage B machinery
(`capacity_scaling.scaled_frame`; B0's 7 columns; the 29,445 lag-study rows), at the primary XGBoost
setting only, scored in megawatts over the fixed static P99, after the constant-scale and
refit-reproduces-B0 assertions. The causal series covers 6 months and is compared with B0 on those
months.

## Auditability and leakage

**Every prompt and answer is logged** to `labeller_log.jsonl` (prompt, image hashes, model id,
sampling parameters, answer, token counts, timestamp). The log pairs plant letters with ANM and
low-output dates, so it is written only under `data/` (a test asserts the path) and never
committed. **Answers are cached** by the SHA-256 of the
(model id, system prompt, prompt, image bytes, sample index). **`--smoke`** uses a stub labeller and
no network, with the first study's `claim_root` and `RUN_KIND.txt` separation. **A causal probe**
replaces every input after the issue day minus 2 with noise and asserts that prompt text and image
bytes are unchanged, run once per record before any real call (the pattern is
`capacity_estimators.assert_scale_ignores_the_issue_day_onward`).

**How the model is called.** The route is `claude -p` (Claude Code's print mode), run from an empty
temporary working directory with every tool disabled (`--tools ""`), no settings sources, no
slash commands or skills, no MCP servers, no session persistence, and a fixed system prompt that
replaces the default one. It needs no API key and no new dependency. The `anthropic` SDK route is
dropped because it would need an API key and a change to `pyproject.toml`, and the maintainer
declined both. Images go in as base64 `image` blocks on the `stream-json` input format. The
command-line tool reports token counts and a cost for every call, and the budget guard reads them.
A test asserts that no prompt contains a path under `data/`, and the empty directory means the
model has no file to open even if it were given a path. The first step is a 2-call probe (done:
see the probe results in the page's methods) that checks images, token counts, the model flag,
and the reported cost.

## Code layout and tests

**Package modules take a `CapacityRecord` (already in `studies.capacity_estimators`) and need
nothing from the first study's folder.** `studies/effective_capacity/capacity_paths.py` stays where
it is, unchanged.

| Where | Holds |
|---|---|
| `packages/studies/src/studies/capacity_evidence.py` | the tools and case-file builder, with the issue-day cut |
| `packages/studies/src/studies/capacity_labeller.py` | knot schema and interpolation, the tools-only labeller, prompt builder, cache key, case-file text and charts |
| `packages/studies/src/studies/capacity_sanity.py` | the four checks, per-knot rejection (hindsight) and the static-P99 fallback (causal) |
| `packages/studies/src/studies/capacity_seasonal.py` | the seasonal reference and the HYST2, HM2, CX2S wrappers |
| `packages/studies/src/studies/labeller_scoring.py` | hits, false alarms, date and size error, event bootstrap |
| `packages/studies/src/studies/capacity_llm.py` | the `LabellerModel` interface, `ClaudeCliModel` and the stub, the budget guard, the call store and log (under `data/`) |
| `packages/studies/src/studies/capacity_estimate_charts.py` | the figure of each estimator's capacity over a plant's daily maximum power: seven panels on one vertical range, loss windows shaded, no calendar date |
| `studies/llm_capacity_labeller/labeller_paths.py`, `labeller_jobs.py` | paths, seeds, loaders, stamps, `CONTROL_SIZE` (a small copy, source stated), and the list of calls with the way each prompt is built |
| `studies/llm_capacity_labeller/` scripts | `build_labeller_inputs.py`, `estimate_part1.py`, `probe_claude_cli.py`, `run_labeller.py` (estimate, causal probe, calls, guard), `score_labeller.py`, `fit_forecast_arms.py`, `llm_capacity_labeller_charts.py`, `report_llm_capacity_labeller.py`, `README.md` |

**Tests** (under `packages/studies/tests/`, run with `--run-studies`; each fails on the bug named):

- **Blind builder:** no event argument in the signature; output on a planted record equals output on
  a copied record with identical power.
- **Causal probe per tool:** noise after the issue day minus 2 leaves each tool and each chart's axis
  domain unchanged (fails on a centred rolling window or a whole-series axis).
- **Seasonal correction:** a December dip shared by all 6 plants corrects to within 2% of flat; a
  one-plant 25% step survives.
- **Clear-sky ratio:** a known 20% loss recovers 0.8 within 1%; cloudy-hour changes do not move it;
  a gap, a 6-hour stuck run, and a ramp are each masked and flagged.
- **Changepoints:** a planted step is found within 3 days with direction and size; a flat noisy
  series gives p above 0.05; two steps give two candidates.
- **Schema and interpolation:** linear between knots, constant outside, unordered or out-of-record
  knots rejected.
- **Cache key:** one changed prompt word, image byte, or sample index changes the key.
- **Prompt and log hygiene:** no prompt contains a path under `data/`; the log path is under `data/`.
- **Each of the four checks,** with a failing and a passing input: bounds (a 1.2 x P99 knot fails),
  a 20% step against a 5% changepoint, an estimate that adds a spurious step (variance rises), and a
  degradation knot inside a cap run; plus the fallback records the failing check.
- **Budget guard:** the runner raises before the 151st call and before 2.0 million input tokens.
- **Estimator figure:** seven panels, one shared vertical range, shading only for the plant's own
  overlapping losses, a note on a panel with no estimate, no calendar date anywhere in the spec, and
  no accessibility text on any mark (the stage list's chart script draws it from the stub pipeline
  too, so the layout is checked before any real run).
- **Boundaries:** `test_study_boundaries.py` passes.
- **Mutation pass** on `capacity_evidence.py`, `capacity_sanity.py`, and `labeller_scoring.py`.

## Docs

- **`docs/studies/forecasts/llm-capacity-labeller.md`:** a figure-led page per the study skill, with
  a short section on what this iteration implies for the next iteration and for wind, linking back
  to the first page; the `mkdocs.yml` nav gains the entry.
- **`docs/studies/forecasts/effective-capacity.md`:** one forward link in its Discussion. The
  roadmap page `docs/roadmap/capacity-estimation.md` is not part of this PR: the maintainer adds
  any pointer there.
- **The new page ends with two short sections.** A discussion of other places in the NGED project's
  data pipeline where a language model could help, each as one or two sentences with the
  deterministic-tools-first pattern, what could go wrong (non-determinism, leakage, cost), and what
  the live service would need (cached, logged, with a guarded fallback to the non-LLM path, per
  `docs/design-philosophy/inherent-stability.md`); speculation is labelled as such. Then a
  proposed follow-up study, as the study skill asks.
- **`studies/llm_capacity_labeller/README.md`** and the `studies/README.md` index row.

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

**Calls: 138 planned, hard stop at 150** (12 spare for schema-invalid retries, each counting).
Records: the original plus 2 fresh draws, so 3.

- Hindsight, tools plus LLM: 3 records x 6 plants x 2 samples = 36.
- Hindsight, raw-summary ablation: 3 x 6 x 1 = 18.
- Causal: 2 records x 6 plants x 6 monthly issue dates x 1 = 72.
- Controls: 1 synthetic null record x 6 plants + 1 positive-control record x 6 plants = 12.

**The C1 to C3 calls (hindsight and controls) run first**, so a guard stop hits the causal run.

**Cost ceiling: US$15, enforced by a hard guard.** The guard sums the cost the command-line tool
reports for each call and refuses a call when the running total plus the largest single-call cost
so far would reach US$15. It also stops at 150 calls. **The probe measured** (two calls, US$0.054
in total): the command-line tool bills a prompt as cache-creation input at about US$4 per million
tokens, and numeric tables cost about 1.4 characters per token (an 18,016-character table was
12,557 tokens). The planned cost is re-estimated from those figures and a built case file's size
before the first real batch (`run_labeller.py --phase estimate`); **if the re-estimate exceeds
US$15, the causal role is cut first, then the sample counts, and the cut is recorded.**

**Compute slots** (from the Study MAIN COORDINATOR): 8 CPU cores briefly for Part 1 tuning and
scoring; 2 cores for case-file building; 1 core while the calls run; then GPU plus 4 CPU cores for
the exploratory forecast fits (3 seeds, 5 folds, primary setting only).

## Open decisions

1. **Model route and spend (decided by the maintainer):** `claude -p` from an empty directory with
   tools disabled, no API key, no new dependency; ceiling US$15 with a hard guard.
2. **Thresholds** (-1.0 point for C1 and C2, 1.1 x P99 upper bound, 14-day window, 5-point step
   tolerance) are plan choices. Recommendation: fix now and report sensitivity on the tuning draw
   only.
3. **Only 2 fresh draws (about 33 events)** give wide intervals, and the 6-month causal run is
   exploratory. Recommendation: accept, and say that a null result is a finding about sample size.
4. **Planted losses last one month** while real ones persist. Recommendation: keep the first study's
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

## Review triage (correctness review; each item checked, 1 and 4 and 13 against the repo)

- (1) AC clip dropped, upper bound 1.1 x P99: accepted. Checked `orientation.json`: static P99
  exceeds `ac_capacity_mw` at plant A (5.95 against 5.71 MW).
- (2) SDK default, `claude -p` only isolated, no-`data/`-path test, API key an open decision: accepted.
- (3) C1 pairing by sample index, noise floor, joint bootstrap, temperature reason: accepted.
- (4) Deranged-power control dropped for a synthetic null record: accepted. Checked
  `derange_months`: it returns a donor map for scales.
- (5) C1 and C2 scored before checks (primary) and after; step and tolerance defined: accepted.
- (6) Hit, recovery, size, and false-alarm rules: accepted.
- (7) Cap array and orientation as arguments; ANM spot check on `power_anm_blind`: accepted.
- (8) Lexicographic tuning with weights 0 and 0.5 reported: accepted.
- (9) C3 interval, read with C1, Bonferroni over C1 and C2: accepted.
- (10) Budget recomputed (1.72M in, 0.23M out, US$8.6), probe re-estimates, C1 to C3 first, 12 spare
  calls: accepted. The coordinator's quoted 1.074M and 0.171M figures were the earlier estimate and
  are superseded.
- (11) Roadmap edit moved to an open decision: accepted.
- (12) Per-knot rejection for hindsight, whole-series fallback for causal: accepted.
- (13) `CONTROL_SIZE` copied with source: accepted. Checked: it is defined only in the study folder.
- (14) Leakage limitations and axis-domain probe: accepted.
- (15) Check 3 window, tie rule, empty-list note: accepted.
- (16) Log under `data/`, tested, never committed: accepted.
- (17) Notation: accepted; the 0.16M and 1.3 x DC wording are gone.

## Implementation notes (where the build differs from the plan above)

- **Answer schema:** frozen dataclasses with explicit validation (`AnswerError`) replace the Pydantic
  model, because `pydantic` is not a direct dependency of `packages/studies` and adding it would end
  the autonomous-merge condition.
- **Commissioning ramp:** the record is built after `studies.commissioning.drop_commissioning_ramp`,
  so a ramp shows as a gap at the record's start, and the clear-sky tool has no separate ramp mask.
- **Raw-summary arm:** weekly totals (measured and modelled megawatt-hours, gap and stuck days), not
  daily rows, because a daily table would cost about 35,000 tokens.
- **Changepoints:** the CUSUM scan and the two-segment fit are the same statistic, so the code runs
  one binary segmentation with permutation p-values.
- **Tokens:** numeric tables cost 1.44 characters per token in the probe, and a built hindsight case
  file is about 11,400 tokens including the two charts. The re-estimated budget is US$12.87 (see
  `run_labeller.py --phase estimate`), most of it the assumed 4,000 output tokens a call.
- **Estimator figure (added requirement):** `llm_capacity_labeller_charts.py` draws four figures (a
  planted step, a planted ramp, a quiet stretch, and a real low-output month) of seven panels each:
  static power, HYST2, HM2, CX2S, the tools-only rule, the model's hindsight labels, and the model's
  causal estimate.
- **Pre-run review fixes:** the causal case file states the knot dates it allows, and `parse_answer`
  reads the same window from the text; every billed call, failures and timeouts included, is
  recorded in the guard and the log; the guard projects the next call at the per-call cap or the
  largest seen; the probe needs `--confirm-spend`; `parse_stream` requires an assistant event naming
  the model and an init event with no tools, skills, or MCP servers, and the empty directory; a
  strict cut uses its own static power and carries no plant fit (the physics model's output still
  rests on fits made from the whole record); and check 3 divides each series by its own median so a
  correct constant offset passes.
- **Round 2 (post hoc, maintainer decision after round 1):** round 1 ran as planned and stays the
  planned result (0 of 32 hits, a failed positive control, C1 to C3 untested). Round 2 adds a local
  scan as a seventh evidence table, softens the prompt for a loss that lasts a month, asks for a note
  on why there is no change, and is judged on two new draws (seeds 11443 and 11444) with a positive
  control that has no two steps at a plant in adjacent months. Round 1's model arm is re-run on
  those draws so the versions are paired. Spend: US$3.93 for 156 calls, US$6.97 in all.
