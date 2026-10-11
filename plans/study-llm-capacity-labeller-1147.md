# Plan: can an LLM, given deterministic evidence tools, label a solar plant's capacity changes? (#1147)

**The problem.** The [first capacity study](../docs/studies/forecasts/effective-capacity.md) (issue
#1143) found that no daily capacity estimate lowered day-ahead error detectably at 6 solar plants,
that the tuning on recovery chose agile estimators whose raw level carries a winter bias of the
physics model, and that HYST, a slow hold-at-static detector designed after the results, did least
harm. The study also had no ground truth for real capacity changes: only 3 of 106 plant-months are
a real low-output loss, all at one plant (`derated_plan.json`: 2024-12, 2025-03, 2026-04).

**The planned solution.** A new study, with four parts. Part 1 reruns HYST, HM, and CX2 with the
seasonal correction built in and tunes them on recovery and false movement together. Part 2 builds
deterministic evidence tools that write a case file per plant and period. Part 3 has Claude Sonnet
5.5 read the case files and write dated capacity knots with a stated cause, as a hindsight labeller
and as a causal estimator. Part 4 adds deterministic checks that reject or repair an implausible
estimate. Everything is scored on blind planted losses on fresh draws, then spot-checked on the real
low months and the plant under active network management (ANM), and one forecast-level check follows.

## Verdict, size, and departures

**Verdict: worth doing, as the issue describes, with the changes below.** The maintainer has fixed
the four-part design, so the plan does not reopen it.

**Size: complex.** The five trigger answers are:

1. **What gets stored:** yes. A published page, plus new checkpoint tables, the LLM prompt and answer
   log, and new code in `packages/studies/`.
2. **Production serving path:** no. Nothing under `src/` or any package except `packages/studies`
   changes, and no production code imports `studies`.
3. **A degradation rule:** no. The sanity checks (Part 4) are R&D checks that fall back to the
   static 99th-percentile capacity (static P99) and record the reason, and they are a study
   prototype of the rule a production estimator would need, not a change to
   `inherent-stability.md`.
4. **More than one defensible design:** yes. How the LLM is called, how knots are scored against
   steps, ramps and intermittent losses, how Part 1 is tuned, and the causal cadence all admit
   several designs (see "Open decisions").
5. **Callers not nameable without searching:** no for the new code; yes for the reused first-study
   machinery (`capacity_estimators`, `capacity_recovery`, `capacity_scaling`,
   `studies/effective_capacity/capacity_paths.py`), which is why the plan names the symbols reused.

**Reviews bought:** both plan reviews (simplicity first, then correctness), both Opus
scientific-validity reviews, a fresh review of every script before its first run, and both diff
reviews. The mutation pass runs because `packages/studies/` changes. The PR touches only
`studies/`, `packages/studies/`, and `docs/studies/`, so the study skill's autonomous-merge
condition can hold, provided no dependency changes (see "How the LLM is called").

**Departures from the issue body:**

- **The ANM log is a causal input as well as a spot check.** The issue says so; the plan applies it
  to the cap-change overlay tool and the ANM-consistency sanity check. The log exists for one plant
  only (`anm_plant` in `built/events.json`), which the plan treats as a spot check and not a sample.
- **Fresh planted draws.** The first study's 10 selection draws were seen in full, and HYST was
  designed after them. This study plants 4 new draws (seeds 11441 to 11444, never run) for every
  contrast. The old draws are re-scored for Part 1 as an exploratory comparison with the first page.
- **Causal runs cover the last 12 months of the record only.** The call budget cannot cover
  monthly issue over 29 months at 6 plants and 5 records (see "Compute and token budget").

## Design

### Records and what is blind

**The record is the first study's `CapacityRecord`:** 6 plants, 2,495 days (2019-09-01 to
2026-06-30 on the CAMS side; power exists only from each plant's start), 24 hours a day, with
`cams_shape` (modelled output per megawatt of DC capacity, driven by CAMS) and `clear_shape`
(`built/record.npz`, loaded by `capacity_paths.load_record`). **Planted losses use
`capacity_recovery.plant_losses` on draws made by `draw_events`**, with the shapes, sizes (5% to
40%), one-month duration, and footprint rules unchanged, so Part 1 results stay comparable with the
first page.

**Blindness is structural.** The case-file builder takes a `CapacityRecord` and an issue day and
nothing else: it has no parameter that carries events. The prompt never mentions planted losses.
The labeller sees planted records and the original record under the same anonymous labels, so it
cannot tell which is which. A test builds the case file from a planted record and from the original
record and asserts that the builder's signature has no event argument and that its output depends
only on the record (see Tests).

**Scoring a knot series reuses the first study's ratio.** For each event, the score is
`capacity_recovery.relative_capacity_error` of (LLM series on the planted record) divided by (LLM
series on the original record). The original-record run is the real-data run, so the "without"
half costs no extra calls. This cancels the labeller's real-data changes and any seasonal bias, as
the first study's ratio did, and the pairing makes the labeller's score comparable with the
estimators'.

### Part 1: the leading designs with the seasonal correction built in

**The three estimators each divide the daily physics-model ratio by a seasonal reference before
detecting anything.** The ratio is power over `cams_shape`-modelled output. The reference is
(a) the same day's median ratio of the other 5 plants (the 6 plants share one 25 km box, so the
winter bias is common to them, and the reference needs no earlier years), and (b) the plant's own
profile from earlier calendar years where one exists
(`capacity_posthoc.deseasonalise_causally`). Both are causal. HYST2, HM2, and CX2S then run the
first study's `hysteresis_level`, `estimate_hm`, and the CX2 solver on the corrected ratio. The
cross-plant reference would hide a loss that hits all 6 plants together, which the limitations
section says, and which the planted losses (one plant per event) do not test.

**Tuning penalises false movement.** On the tuning draw (draw 0, seed 11430, as before) each
estimator's setting minimises `recovery_error + 2 x false_movement_share` over its grid, after
the hard rule that false movement must not exceed `FALSE_MOVEMENT_GATE` (0.10). The weight of 2 is
a plan choice (a 5-point recovery gain is not worth moving on 3 percentage points more days) and is
stated on the page. HYST2's trigger (10% drop for 7 days) was untuned in the first study, so it
joins the grid.

**Scoring** uses `score_events` for recovery and `false_movement_share` on the same 23 quiet
plant-months, on the 4 fresh draws. The old 10 selection draws are scored too, labelled
exploratory.

### Part 2: the case file

**A case file is a deterministic function of (plant, period, issue day).** Each tool is computed
for the hindsight run from all days, and for the causal run from days up to the issue day minus 2
(the first study's `ISSUE_LAG_DAYS`, which leaves the CAMS delay of 1 day and the 09:00 issue time
inside the bound). The causal estimator therefore sees exactly what the first study's issue-time
scales saw.

| Tool | What it measures | Cause it discriminates | Causal timing |
|---|---|---|---|
| Clear-sky output ratio | Daily mean of power over `cams_shape`-modelled output on hours with clear-sky index above 0.8, cleaned of ANM-capped hours, in the plant's own units and normalised to its first 90 valid days | any loss against none | through day T-2 |
| Seasonal correction | The ratio minus the plant's earlier-year profile and minus the other plants' same-day median ratio, reported as two separate columns | model bias (winter) against capacity loss | earlier years only for the profile; the other plants' ratio through T-2 |
| Neighbour ratio | The plant's 7-day output over the others' (`capacity_estimators.neighbour_ratio`) | local loss against shared weather | through T-2 |
| Hour-of-day residual shape | The ratio by hour ending, per month, as a 24-value row | a failed string or shading (a morning or evening hole) against soiling (flat across hours) | through T-2 |
| Power against irradiance envelope | Per month, the 95th-percentile power in 8 irradiance bins, plus clipping statistics: the 99th percentile of power, the share of hours within 2% of it, and the slope below the plateau | a fixed plateau with a falling slope is a DC-side loss; a lowered plateau is an AC-side limit or curtailment | through T-2 |
| ANM cap-change overlay | The cap (megawatts) and the share of each day's daylight hours where the cap binds, for the ANM plant; a flat `none recorded` for the other 5, which NGED confirmed ran free | curtailment against capacity loss | through T-2 (the live service has the feed) |
| Changepoint candidates | A CUSUM and a two-segment least-squares fit on the corrected ratio: date, direction, size in per cent, and a permutation p-value | candidate knots | through T-2; a changepoint within the last 14 days is flagged as unconfirmed |
| Data-quality flags | Gap days, stuck-value runs (same value 6+ hours), and the commissioning ramp (`studies.commissioning`) | data fault against anything physical | through T-2 |

**The case file is text tables plus 3 small PNG charts**: the corrected ratio against date with
the changepoint candidates and ANM cap days marked, the hour-of-day-by-month residual heatmap, and
the power-against-irradiance envelope by quarter. Charts are drawn with Altair and `vl_convert`
(both import in the study environment) under the `dataviz` rules, with no site identifier.
Half-hourly or hourly raw data never enter a prompt. Daily summaries enter only in the ablation arm
described below.

### Part 3: the LLM

**Model: Claude Sonnet 5.5 (`claude-sonnet-5-5`)**, temperature 1.0 so that repeated samples differ.
Fable or Opus would probably read a case file better; the page states that as a limitation and does
not test it.

**Output schema** (JSON, validated by a Pydantic model before use):

```json
{
  "plant": "A",
  "period": ["2025-01-01", "2025-06-30"],
  "knots": [{"date": "2025-03-04", "capacity_mw": 17.9,
             "cause": "string_failure|soiling|degradation|curtailment|model_bias|data_fault",
             "evidence": ["changepoint 2025-03-03 size -12% p=0.002", "hour-of-day: 07h-10h low"]}],
  "no_change": false,
  "confidence": 0.7
}
```

**The default is no change**: the prompt says that an empty `knots` list with `no_change: true` is
the expected answer for most periods. Knots are linearly interpolated between dates, held at the
first knot's value before it, and held at the last knot's value after it. `capacity_mw` is the
level the knot asserts, in megawatts, and the first knot's level is the starting capacity (static
P99 if the labeller writes none).

**Two roles.**

- **Hindsight labeller:** one call per plant per record, with the whole-record case file.
  Case files are at monthly granularity to fit the context.
- **Causal estimator:** one call per plant per monthly issue date over the last 12 months of the
  record (12 issue dates). The prompt carries the previous call's knots as "your current estimate",
  so the causal estimator revises rather than restarts. The case file ends at issue day minus 2.

**Arms and the ablation** (all hindsight, same records):

- **Tools only (deterministic labeller):** a fixed rule writes a knot at each changepoint with
  p below 0.01 and size above 8%, takes the cause from a decision table (ANM overlap gives
  curtailment, a stuck-run flag gives data fault, an hour-of-day hole gives string failure,
  otherwise degradation), and proposes no knot otherwise. The thresholds are chosen on the tuning
  draw only.
- **Tools plus LLM:** the arm described above, 3 samples.
- **LLM on raw daily summaries:** the same schema and prompt, but the case file holds only the daily
  power total, daily CAMS irradiance total, and the data flags, with no tool outputs.

### Part 4: sanity checks (`capacity_sanity.py`)

**Each check is a function of (knots, case-file evidence) returning pass, fail with a reason, or a
repaired series.** An estimate is rejected if a hard check fails, and the fallback is the static P99
with the reason recorded on the series.

1. **Time order and dates:** knots strictly increasing in date, inside the record. Hard.
2. **Physical bounds:** 0.2 x static P99 <= capacity <= 1.3 x the fitted DC capacity from
   `built/orientation.json` (the 1.3 is a plan choice, stated on the page), and capacity above the
   fitted AC limit is clipped to it. Hard (bounds), repair (AC limit).
3. **Change size within evidence:** each step's size must lie within 5 percentage points or 40% of
   the tools' own changepoint size, whichever is larger. A step with no matching changepoint within
   14 days fails. Hard.
4. **ANM consistency:** a knot whose cause is not `curtailment` must not sit inside a run of 3 or
   more cap-binding days; a `curtailment` knot with no cap-binding day within 14 days fails. Repair:
   the series ignores the knot.
5. **Neighbour agreement:** over the period after a knot, the sign and rough size of the series'
   change must agree with the change in the neighbour ratio within 10 points. Soft: lowers
   confidence and flags, does not reject.
6. **Reproduce the power:** apply the estimate to the measured power; the variance of the
   normalised clear-sky ratio over the period must not exceed the variance without the estimate.
   Hard.
7. **Cross-sample agreement:** the 3 hindsight samples' series must agree within 5 points on 80% of
   days; where not, the plant-period is reported unstable and the median series is used. Soft.
8. **Step-frequency prior:** more than 3 knots in 12 months at one plant fails. Hard, with the
   threshold set from the real record (the 3 real low months are one episode per 3 months at most).
9. **Rejection or repair rule:** hard failure gives the static P99 and a recorded reason; a repair
   is applied and recorded; soft flags are recorded only.

**The sanity layer is scored too:** the share of planted events lost by rejections, and the share of
implausible estimates (a deliberately corrupted knot series: swapped dates, a 2x capacity, a
no-evidence step) that the layer rejects. The second share is the layer's own positive control.

### Evaluation

**Planned contrasts (written before any result; thresholds are plan choices):**

- **C1: tools plus LLM against tools only,** recovery error on the 4 fresh draws, hindsight. The
  LLM adds value if the paired difference in mean absolute capacity error is at most -1.0 points
  (the first study's best recoverer, KF, beat doing nothing by 3.2), with the 97.5% event-bootstrap
  interval wholly below zero.
- **C2: tools plus LLM against the LLM on raw daily summaries,** same metric and threshold. This
  answers whether the tools matter.
- **C3: false movement on the 23 quiet plant-months,** tools plus LLM against the gate of 10%. The
  labeller passes if the share of days its capacity moves more than 5% from its 90-day median is at
  most 0.10. HYST2 is the reference.
- **C4: causal tools plus LLM against the best Part 1 estimator,** recovery error on the last 12
  months of draw 1 and the original record, threshold -1.0 points.
- **C5: forecast-level,** the best hindsight and best causal estimate against B0, at the primary
  XGBoost setting only, with the first study's decision rule (`capacity_scaling.decision_verdict`,
  threshold 0.179 points, month-resampled bootstrap) reused.

**Everything else is exploratory:** the real low months, the ANM plant, the 3-sample spread, the
sanity-check scores, the old-draw re-score of Part 1, the second XGBoost setting, and the per-cause
breakdown. Anything designed after a first result is labelled post hoc.

**Interval method.** Stage A contrasts use a 97.5% bootstrap over events (events within a draw
share weather, so the page says the interval covers which month, shape and size were drawn and
not differences between plants). C5 resamples whole calendar months and one of 3 seeds
(`studies.bootstrap.bootstrap_difference`, 2,000 resamples).

**Controls.**

- **Negative:** the labeller reads case files from the first study's month-deranged records
  (`capacity_scaling.derange_months`); the correct answer is no change, and the false-alarm count
  is the size of the false-positive floor. A second negative control reads a case file from a
  different plant than the one named in the prompt header: the labeller must not hallucinate steps.
- **Positive:** the 30% step control (`CONTROL_SIZE`, `draw_control_steps`) must be found by tools
  plus LLM in at least 80% of its 17 plant-months, or C1 to C4 are reported as untested, not null.
- **Oracle for C5:** the first study's oracle scale (`capacity_scaling.oracle_scale`) on the planted
  control, as before, so the forecast arm's recovery share can be stated.

**Scores per labeller:** hits (a knot within 14 days of the event start with the right sign and a
size within 50%), false alarms (knots in quiet plant-months), date error in days, size error in
points, and the capacity error above. A planted ramp counts as a hit if a knot lies in its first
half.

**Real data spot checks (exploratory, charted).** The LLM's hindsight labels are the reference
against which the deterministic estimators (KF, HYST, HM, CX2, and the Part 1 designs) are scored
on the real record, and the plan states that the labels are an expert opinion and not ground truth:
the reference is partly built from the same tools, so agreement between the tools and the labeller
is not independent evidence. The three real low months and the ANM plant are shown as case-file
figures with the labeller's knots. No figure puts a plant letter next to the ANM plant, a curtailment
date, or a low-output date.

**Forecast-level check.** The knot series become issue-time scales
(`capacity_estimators.scale_for_rows`) and enter the first study's Stage B machinery
(`capacity_scaling.scaled_frame`, `fit_capacity_arms.py`), scored in megawatts over the fixed
static P99, on B0's 7 columns and the lag study's 29,445 rows, with the constant-scale and
refit-reproduces-B0 assertions run first. The causal series covers only 12 months, so the causal
arm is scored on those months and compared with B0 on the same months. This is exploratory in its
conclusions, planned in its single contrast (C5).

## Auditability and leakage

**Every prompt and answer is logged** to `labeller_log.jsonl` (prompt text, image hashes, model id,
sampling parameters, answer, token counts, timestamp). **Answers are cached** by the SHA-256 of the
(model id, system prompt, prompt, image bytes, sample index), so a re-run costs nothing and a
changed prompt cannot reuse a stale answer. **A real/smoke split** reuses `claim_root` and
`RUN_KIND.txt`; `--smoke` uses a stub labeller that returns a fixed answer and makes no network
call. The model name and version are written into each answer row.

**The causal probe.** A test replaces every input after the issue day minus 2 with random noise
and asserts that the prompt text and the image bytes are unchanged (for each tool). The first
study's `capacity_estimators.assert_scale_ignores_the_issue_day_onward` is the pattern. The
causal runner calls the probe once per record before any real call.

## How the LLM is called

**Recommendation: call the model through the `claude` command-line program in print mode
(`claude -p --model claude-sonnet-5-5 --output-format json`), which is installed on the
workstation (`/home/jack/.local/bin/claude`), or through the Anthropic SDK run as
`uv run --with anthropic`.** Both avoid a dependency change, which the autonomous-merge condition
forbids; the `anthropic` package is not in `pyproject.toml` or `uv.lock` (checked). Whether
`claude -p` accepts image input and returns token counts in the form the log needs is
**unverified**; the first implementation step is a 2-call probe that settles which route to use.
See "Open decisions".

## Code layout and tests

### `packages/studies/src/studies/` (tested)

| Module | Holds |
|---|---|
| `capacity_evidence.py` | the Part 2 tools as pure functions on `CapacityRecord`, with the issue-day cut; the case-file builder |
| `capacity_casefile.py` | text and chart rendering of a case file; the case-file hash |
| `capacity_labeller.py` | the knot schema, knot interpolation to a daily series, the deterministic tools-only labeller, the prompt builder, the cache key |
| `capacity_sanity.py` | the nine Part 4 checks and the reject-or-repair rule |
| `capacity_seasonal.py` | the cross-plant reference and HYST2, HM2, CX2S wrappers over the first study's estimators |
| `labeller_scoring.py` | hits, false alarms, date and size error for knots, and the event bootstrap |

### `studies/llm_capacity_labeller/` (not unit-tested; checked through `report.md`)

| Script | What it does |
|---|---|
| `labeller_paths.py` | paths, the fresh draw seeds, imports of the first study's loaders (read-only) |
| `build_draws.py` | the 4 fresh draws, the original record's quiet windows reused |
| `estimate_part1.py` | Part 1 tuning and scoring (CPU, 8 cores) |
| `build_case_files.py` | all case files, hindsight and causal, with the causal probe |
| `run_labeller.py` | the LLM calls, the cache, the log, the budget guard |
| `score_labeller.py` | all Stage A contrasts, the sanity layer, the spot checks |
| `fit_forecast_arms.py` | C5 (GPU, 4 cores), reusing `fit_capacity_arms` helpers |
| `report_llm_capacity_labeller.py`, `llm_capacity_labeller_charts.py` | `report.md`, the figures |
| `README.md` | what each file holds |

The package modules above need the first study's `studies/effective_capacity/capacity_paths.py`
helpers, which sit in a study folder and cannot be imported from another (`test_study_boundaries`).
Whatever the new scripts need from `capacity_paths.py` (`claim_root`, `fingerprint`, `check_stamp`,
`load_record`, `write_json_atomic`) moves into `packages/studies/src/studies/capacity_paths.py`
with its tests, and the first study's scripts import it from there. This is a refactor of a merged
study's imports; the plan verifies that the first study's `report.md` is unchanged after it, bit
for bit on the per-row outputs, as the study skill requires.

### Tests (each can fail on the bug it exists to catch)

All under `packages/studies/tests/`, run with `--run-studies`.

- **Blind builder:** the case-file builder's signature has no event argument, and its output on a
  planted record equals its output on a hand-copied record with identical power. Fails if events
  leak in.
- **Causal probe, per tool:** noise after the issue day minus 2 leaves each tool's output unchanged.
  Fails if a tool reads the future (a centred rolling window is the planted bug).
- **Seasonal correction cancels a shared winter bias:** a synthetic record where all 6 plants share
  a 25% December dip must give a corrected ratio within 2% of flat; a one-plant 25% step must
  survive. Fails if the reference removes real loss or leaves bias.
- **Clear-sky ratio on a synthetic plant** with a known 20% loss from a known day recovers 0.8 to
  within 1%; the clear-sky mask drops cloudy hours (a cloudy-hour-only change must not move it).
- **Hour-of-day shape separates the two causes:** a morning-only loss and a flat loss give different
  residual rows.
- **Envelope and clipping:** a DC-side loss (plateau fixed, slope down) and an AC-side loss
  (plateau lowered) are labelled differently by the statistic.
- **Changepoint tool:** finds a planted step within 3 days, reports direction and size, and returns
  p above 0.05 on a flat noisy series; a series with 2 steps gives 2 candidates.
- **Data-quality flags:** a gap, a stuck run of 6 hours, and a ramp are each flagged.
- **ANM overlay:** binding-hour share on a hand-built cap series equals the hand count; the 5 free
  plants return `none recorded`.
- **Knot interpolation:** linear between knots, constant outside, and unordered knots are rejected.
- **Cache key:** changing one word of the prompt, one image byte, or the sample index changes the
  key; the same inputs give the same key.
- **Sanity checks, one test each,** each with a failing input and a passing input: order, bounds
  (including the AC clip), change size (a 20% step against a 5% changepoint), ANM consistency (a
  degradation knot inside a cap run), neighbour disagreement, variance not reduced (an estimate
  that adds a spurious step), cross-sample spread, step frequency, and the fall-back-with-reason
  rule.
- **Scoring:** hits, false alarms, and date error on a hand-built knot series against a
  hand-built event; a knot of the wrong sign is a false alarm and a miss.
- **Budget guard:** the runner raises before the 321st call and before 4.2 million input tokens.
- **Boundary test:** `test_study_boundaries.py` passes (new scripts import only their own folder,
  `studies.*`, and reviewed packages).
- **Mutation pass** on `capacity_evidence.py`, `capacity_sanity.py`, and `labeller_scoring.py`
  (the `implement-issue` skill, step 7).

## Docs

- **`docs/studies/forecasts/llm-capacity-labeller.md`:** a figure-led study page following the study
  skill's structure, with a short discussion of what this iteration implies for the next iteration
  and for wind, and a link back to the first page. The `mkdocs.yml` nav gains the entry under
  Forecasts.
- **`docs/studies/forecasts/effective-capacity.md`:** one line in its "Discussion" linking forward
  to the follow-up, and the roadmap page `docs/roadmap/capacity-estimation.md` gains a pointer in
  its status text. No other page changes.
- **`studies/llm_capacity_labeller/README.md`** and the `studies/README.md` index row.
- **The page's figure sequence** (planned now, so the page is figure-led): Figure 1 headline: the
  paired contrasts C1 to C4 with intervals. Then the data (the corrected ratio at the 6 plants,
  showing the winter bias and its removal), a worked case file for one planted loss, the labeller's
  knots against the truth for 3 planted events, the controls, the ablation, the sanity layer's
  catches, and the forecast-level check last.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --run-studies -n auto packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
uv run python studies/llm_capacity_labeller/run_labeller.py --smoke
```

Also the pydoclint and docs-link-checker steps CI runs, per the maintainer's note that the skill's
verification set omits them (exact commands unverified; read `.github/workflows/`).

## Compute and token budget

**Calls.** Planned 288, ceiling 320 (a 10% allowance for repair retries):

- Hindsight, tools plus LLM: (4 fresh draws + 1 original) x 6 plants x 3 samples = 90.
- Hindsight, raw-summary ablation: 5 x 6 x 1 sample = 30, plus 2 more samples for one draw and the
  original = 24, giving 54.
- Causal, tools plus LLM: (1 planted draw + the original) x 6 plants x 12 monthly issue dates = 144.

**Tokens.** Hindsight case files about 12,000 input tokens (tables about 6,000, 3 charts about
6,000; the chart cost is an estimate) and 1,500 output; causal about 8,000 input and 1,000 output.
That is 144 x 12,000 + 144 x 8,000 = 2.9 million input tokens and 0.36 million output tokens in
all, **unverified until the 2-call probe measures real counts.** The guard stops the run at 4.2
million input and 0.6 million output tokens.

**Cost.** At an assumed US$3 per million input and US$15 per million output tokens (a figure from
memory, not checked against current pricing), the planned run costs about US$14 and the guard
ceiling about US$22. The plan recommends a ceiling of **US$25** for the study. The maintainer
decides anything beyond that.

**Compute slots** (requested from the Study MAIN COORDINATOR as before): 8 CPU cores briefly for
Part 1 tuning and scoring (the first study's `estimate_capacity.py` pattern), a few CPU cores for
building case files, no compute for the API calls beyond one core, then GPU plus 4 CPU cores for
the C5 XGBoost fits (the first study's `fit_capacity_arms.py`, 3 seeds, 5 folds).

## Risks and open decisions

1. **How to call the model.** Recommendation: `claude -p` if the probe shows it takes images and
   returns token counts, otherwise `uv run --with anthropic`. Adding `anthropic` to `pyproject.toml`
   would end the autonomous-merge condition and needs the maintainer.
2. **Cross-plant seasonal reference.** It cancels a loss shared by all 6 plants and relies on one
   25 km box. Recommendation: use it, state the limitation, and let the profile from earlier years
   stand beside it.
3. **Causal runs cover 12 months, one planted draw.** Recommendation: accept, because
   monthly issue over the full 29-month power record would need about 350 calls (2 records x 6 plants x 29 months) instead of 144. A different cadence (quarterly)
   over the full record is the alternative at the same cost.
4. **Planted losses last one month and then vanish.** Real capacity losses usually persist.
   Recommendation: keep the first study's shapes for comparability; a persistent-step variant is a
   post hoc addition if budget remains.
5. **Circularity of the real-data reference.** The LLM labels use the same tools as the
   deterministic estimators. Recommendation: state it on the page and never describe the labels as
   ground truth.
6. **Thresholds** (-1.0 point for C1, C2, and C4; weight 2 for false movement; 1.3 x DC capacity;
   14-day matching window) are plan choices with no empirical basis yet. Recommendation: fix them
   now and report sensitivity at half and double on the tuning draw only.
7. **Hits scored with a 14-day window** favour the hindsight labeller over HYST's 7-day delay.
   Recommendation: report date error beside the hit rate.
8. **A Fable or Opus labeller** would likely read case files better. Recommendation: a limitation
   only; the maintainer can spend credits on a one-plant repeat if the Sonnet result is borderline.
9. **Moving `capacity_paths.py` helpers into `packages/studies`** touches the merged study's
   imports. Recommendation: do it, because the boundary rule forbids the alternative, and prove the
   first study's outputs unchanged.
10. **The 1-day CAMS delay against `ISSUE_LAG_DAYS = 2`.** The plan keeps 2 for consistency with
    the forecast contrast; the brief's 1-day figure is inside it. Unverified which the live service
    achieves in practice.
