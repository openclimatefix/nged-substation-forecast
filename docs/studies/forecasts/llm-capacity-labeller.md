# With the seasonal bias divided out, only the hold-at-static detector stayed within the false-movement gate, and it recovered no planted loss

> **Status: the language-model runs have not happened yet.** This page holds the design, the probe
> of the model route, the Part 1 results (which need no model), and the closing sections. The
> sections on the planned contrasts and the figures of each estimator over a plant's power are
> written after the runs, from `report.md`. No number below comes from a model.

**Bottom line.** The [first capacity study](effective-capacity.md) found that agile capacity
estimators recover a planted loss but also move when nothing has changed, because the ratio of a
plant's power to the physics model's output carries a winter bias. **With that bias divided out
causally, one of three designs stayed within the 10% false-movement gate (HYST2, a hold-at-static
detector, which moved on 4.1% of days in 23 quiet plant-months), and it did no better than holding
the static power on two fresh draws of planted losses (5.31 against 5.17 points of mean absolute
capacity error).** The two designs that recovered losses (HM2, 3.59 points, and CX2S, 2.39 points)
moved on 13.8% and 59.7% of quiet days. For Flexpectation (this project), the next question is
whether a language model that reads six deterministic evidence tools can label capacity changes
without that trade-off, and this page's runs are set up to answer it. The Part 1 numbers rest on 32
planted losses, so they are descriptive.

## How the study works

**The record, the planted losses, and the score are the first study's.** The record holds 6 solar
plants over 2,495 days. A planted loss multiplies one plant's power for one calendar month by a
step, a ramp, or an intermittent pattern of 5% to 40%. The capacity error of an estimator is the mean
absolute gap between its capacity on the planted record over its capacity on the original record and
the factor the loss applied, over the loss month and the 60 days after it. Two fresh draws (seeds
11441 and 11442, 16 losses each, never run before) carry every planned contrast. The first study's
tuning draw (seed 11430) sets Part 1's settings.

**Part 1 divides the physics model's seasonal bias out before an estimator reads the ratio.** The
reference for one plant on one day is the median of the other five plants' ratio on that day, plus
the plant's own day-of-year profile from earlier calendar years. Both read only days up to that day.
The shared part would hide a loss that hit all six plants on the same day, which the planted losses
(one plant per event) do not test.

**Part 2 gives a labeller six deterministic evidence tools.** They are the clear-sky ratio with
data-quality masks, the seasonal correction with the 7-day neighbour ratio, the hour-of-day shape
with the power envelope, the export-cap overlay for the one plant that has a cap, changepoint
candidates with permutation p-values, and a tools-only rule that writes a knot at each strong
changepoint. A case file is text tables and two small charts. It is a function of one plant's record
up to a cut-off day, and the builder has no argument that names a planted event.

**Part 3 asks Claude Sonnet 5.5 to write capacity knots.** A knot is a date, a capacity in
megawatts, and one of six causes. Capacity is linear between knots and constant outside them, so a
step is two knots one day apart, and an empty list means the static power. The hindsight role
reads the whole record once per plant. The causal role reads the record up to two days before each
of six monthly issue dates and is shown its previous estimate.

**Part 4 rejects an implausible estimate with four hard checks.** The checks are bounds (0.2 to 1.1
times the static power), a step that a changepoint candidate supports, a variance of the corrected
ratio that the estimate does not raise, and consistency with the export cap. A causal estimate that
fails any check is replaced by the static power. A hindsight estimate loses only the failing knots.

**Planned contrasts (fixed in the plan before any result).** C1 compares tools plus model with tools
only, and C2 compares tools plus model with the model reading weekly energy totals alone, both as the
change in mean absolute capacity error in points on the two fresh draws, with a 97.5% bootstrap
over (event, sample) pairs. The model adds value at a change of −1.0 point or lower with the
interval wholly below zero. C3 is the share of days that the capacity moves more than 5% from its
trailing median on the 23 quiet plant-months, against the 10% gate. The positive control (17 steps of
30%) must be found at 80% or more, or C1 to C3 are reported as untested. A synthetic null record, in
which no plant ever changes, gives the false-alarm floor.

## Probe of the model route

**`claude -p` accepts images, reports its token counts and cost, and serves the requested model.**
The study calls Claude Code's print mode from an empty temporary directory with every tool disabled,
no settings, memory, skills, or MCP servers, and a fixed system prompt. It needs no API key. The
probe made two calls for US$0.054 in all. A 200 by 100 pixel line chart passed as a base64 image was
read correctly (the answer was 5), and an 18,016-character numeric table cost 12,559 input tokens, or
1.44 characters per token. The command-line tool bills a prompt as cache-creation input at about
US$4 per million tokens. It has no temperature flag, so every arm shares its default sampling.

**The planned 138 calls are estimated at US$12.87, under a US$15 ceiling that a guard enforces.** The
estimate takes the 1.15 million input tokens of the built prompts at US$4 per million and assumes
4,000 output tokens a call at US$15 per million, which is the larger part of the cost and is unmeasured
until the first batch. The guard sums the cost the tool reports and stops the run before a call that
could reach the ceiling. If the measured cost makes the estimate exceed the ceiling, the causal role
is cut first, then the samples.

## Results

### Part 1: only HYST2 passed the false-movement gate, and it recovered nothing

| Estimator | Mean absolute capacity error, 32 planted losses (points) | Days that moved, 23 quiet plant-months | Mean delay to find a step (days) |
|---|---|---|---|
| Static power (S0) | 5.17 | 0.0% | not applicable |
| HYST2 (drop 15%, 3 days) | 5.31 | 4.1% | 12.4 |
| HM2 (stay 0.95, spread 0.2) | 3.59 | 13.8% | 6.2 |
| CX2S (the first study's tuned CX2) | 2.39 | 59.7% | 2.1 |

**The tuning rule picks the lowest recovery error among settings within the gate, and no HM2
setting was within it.** For HM2 the rule then takes the setting with the least false movement. The
first study's published figures for its own estimators, scored on its ten selection draws, were 4.40
points and 2.6% of days for HYST, 2.9 points and 31.5% for HM, and 2.2 points and 63.4% for CX2,
against 5.00 points for doing nothing. The comparison is loose, because the draws differ. The seasonal
correction cut HM's false movement from 31.5% to 13.8% and left CX2's roughly unchanged.

## Discussion

### Where else a language model could help in the NGED data pipeline

**Each idea below is speculation unless a repository page is cited, and each follows one pattern:
deterministic tools first, the model reads their output.** The pattern is the one this study tests.
The tools produce numbers that can be checked. The model chooses among them and explains the choice.

- **Labelling faulty metering and switching events.** The [requirements](../../background/requirements.md)
  list switching-event detection and faulty-meter detection as objectives, and say that a negative
  result on detecting switching from power data would inform extracting switching records from
  operational systems. A model could read changepoint candidates, stuck-run flags, and neighbouring
  substations' power and propose a label. The risk is a confident label with no evidence behind it.
- **Reading outage notices, curtailment notices, and DNO documents (speculation).** A model could
  turn free text into dated rows for a table the pipeline already reads. The risk is a
  mis-parsed date or substation that enters the data without anyone noticing.
- **Turning data-quality warnings into operator-readable explanations (speculation).** The
  [inherent-stability](../../design-philosophy/inherent-stability.md#three-audiences-three-channels)
  design already sends an attributable event to the developers through Sentry. A model could write
  the one-paragraph explanation from the event's tags. The risk is an explanation that sounds
  plausible and names the wrong cause.
- **Cleaning the substation metadata table (speculation).** A model could propose fixes to
  inconsistent names or coordinates for a human to accept. The risk is a silent edit to a table the
  rest of the system trusts.
- **Writing the daily health summary, and explaining a forecast miss (speculation).** A model could
  read the day's check results and the largest errors against the forecast and write a few sentences.
  The risk is that the explanation attributes a miss to a cause it never tested.

**Every use shares three risks and three requirements.** The risks are non-determinism (the same
input can give a different answer), leakage (a model that has read the real dates of a period
carries what it learned about the weather in them), and cost (this study's probe shows a
case file costs a few US cents, so a use on every series every day needs a budget). A live use would
need the answer cached by the hash of its prompt, every prompt and answer logged, and a guarded
fallback to the path that has no model. That follows the
[degradation rules](../../design-philosophy/inherent-stability.md#the-rules): an absent or invalid
model answer is treated as missing input, widens the uncertainty, and is recorded on the row, and a
warning path never raises. Following the same rules, a capability that could live in the training
loop belongs there, so the model's output would be a training label, never part of the serving path.

### Proposed follow-up study

**If the model's labels beat the tools-only rule, the follow-up would use them as training labels
for a detector that has no model in the serving path.** This study would have learned how often the
labels agree with planted losses, and what a call costs. The follow-up would ask whether a
changepoint detector fitted to the model's labels on the real record finds the next real low-output
month earlier than HYST2 does, with the false-movement gate unchanged. A rough cost is a day of
agent time, no downloads, and under US$15 of model calls (an estimate from this page's budget, not a
measurement). The case for it is that 3 real low-output months are the only real losses in the
record, so a labeller is the only way to get more. The case against is that a labeller shares the
tools' inputs, so its labels are an expert opinion and not ground truth, and a null result on 32
planted losses would not justify more spend. The proposal commits the project to nothing.

## Limitations

**The numbers on this page depend on a small sample and a shared input.** Part 1 rests on 32 planted
losses, so the intervals the planned contrasts will give are wide, and a null result there is a
finding about sample size. The planted losses last one month while a real loss persists. The six
plants share one 25 km satellite box, which is why the shared correction works, and the model's labels
share the tools' inputs, so they cannot score the physics model's own errors. Real calendar dates
let a model draw on what it learned about UK weather during training. The satellite normalisation
mutes that and does not remove it. A Fable or Opus model would probably read a case file better than
Sonnet 5.5, and the study tests only Sonnet 5.5.

## Scope

**This page says nothing about wind, about plants outside the 6 studied, or about a forecast.** The
exploratory forecast-level check (`fit_forecast_arms.py`) divides the lag study's baseline forecast by
each capacity series at one XGBoost setting, has no decision rule, and is not reported until the
runs have happened.

## Data and code availability

**The satellite irradiance and the lag-features study's rows are inputs that the project holds.**
NGED's power data are private, and no plant's name, identifier, or coordinates appear on this page.
The prompt and answer log pairs plant letters with dates, so it stays in the private data store and is
never committed. The scripts are in
[`studies/llm_capacity_labeller/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/llm_capacity_labeller),
at the commit that merged this page, and the shared code is in `packages/studies/`. The model is
Claude Sonnet 5.5 through `claude -p` (version 2.1.296).

## Reproducing the figures

```bash
uv run python studies/llm_capacity_labeller/build_labeller_inputs.py
uv run python studies/llm_capacity_labeller/estimate_part1.py
uv run python studies/llm_capacity_labeller/probe_claude_cli.py --summarise
uv run python studies/llm_capacity_labeller/run_labeller.py --phase estimate
uv run python studies/llm_capacity_labeller/run_labeller.py --phase hindsight --confirm-spend
uv run python studies/llm_capacity_labeller/run_labeller.py --phase causal --confirm-spend
uv run python studies/llm_capacity_labeller/score_labeller.py
uv run python studies/llm_capacity_labeller/fit_forecast_arms.py --device cuda --max-workers 4
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py
uv run python studies/llm_capacity_labeller/report_llm_capacity_labeller.py
```
