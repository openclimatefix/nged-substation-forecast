# Claude Sonnet 5.5 found none of 32 planted solar losses with six evidence tools, and a post hoc local scan lifted it to about 1 in 8 of 34 new losses

**Bottom line.** A language model that reads deterministic evidence tools did not label a solar
plant's capacity changes usefully. **In the planned round, Claude Sonnet 5.5 found 0 of 32 planted
one-month losses, the fixed tools-only rule found 1, and the 30% positive control (2 of 17 found)
failed its 80% gate, so the planned contrasts C1 to C3 are untested and not null.** The tools'
changepoint list had a matching candidate for only 3 of the 32 losses, and the model left all 3
unmarked. A post hoc round 2 added a local scan to the tools and softened the prompt, and on 34 new
planted losses the tools-only rule found 5 (15%) and the model with the tools found 8 of 68
(12%), against 0 for round 1's model, but neither passed the 80% control gate. For Flexpectation
(this project), a capacity labeller is not ready to use as a reference, and the cheap lesson is that
the tools, not the model, set the ceiling: a one-month loss in a noisy record needs a detector built
for one month. The 32 and 34 losses are small samples, so the intervals are wide.

![Figure 1: Round 1 (planned). Each estimator on a planted step loss at one plant, with the daily
maximum power as grey dots and the planted month
shaded.](../assets/llm_capacity_labeller_estimators_planted_step.svg)

> **Status: the round 1 and round 2 runs are complete and scored; the page has had no science
> review yet.** Round 1 was planned. Round 2 is post hoc, and every round 2 number is labelled so.

**Costs.** The 138 round 1 calls cost US$3.04, and the 156 round 2 calls cost US$3.93, so the study
spent US$6.97 of its US$15 ceiling. The probe's two calls are included.

## How the study works

**The record, the planted losses, and the score are the first study's.** The record holds 6 solar
plants over 2,495 days. A planted loss multiplies one plant's power for one calendar month by a
step, a ramp, or an intermittent pattern of 5% to 40%. The capacity error of an estimator is the
mean
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
of six monthly issue dates and is shown its previous estimate. A causal case file states the
static power and the knot dates it allows, and the static power is the 99th percentile of the
readings up to the cut-off day. The plants' fitted tilts, azimuths, and capacities are left out of
causal case files because they were fitted on the whole record. The physics model's output was
built from those fits, so one look-ahead remains in the modelled output.

**Part 4 rejects an implausible estimate with four hard checks.** The checks are bounds (0.2 to 1.1
times the static power), a step that a changepoint candidate supports, a variance of the corrected
ratio that the estimate does not raise, and consistency with the export cap. A causal estimate that
fails any check is replaced by the static power. A hindsight estimate loses only the failing knots.

**Planned contrasts (fixed in the plan before any result).** C1 compares tools plus model with tools
only, and C2 compares tools plus model with the model reading weekly energy totals alone, both as
the
change in mean absolute capacity error in points on the two fresh draws, with a 97.5% bootstrap
over (event, sample) pairs. The model adds value at a change of −1.0 point or lower with the
interval wholly below zero. C3 is the share of days that the capacity moves more than 5% from its
trailing median on the 23 quiet plant-months, against the 10% gate. The positive control (17 steps
of
30%) must be found at 80% or more, or C1 to C3 are reported as untested. A synthetic null record, in
which no plant ever changes, gives the false-alarm floor.

## Probe of the model route

**`claude -p` accepts images, reports its token counts and cost, and serves the requested model.**
The study calls Claude Code's print mode from an empty temporary directory with every tool disabled,
no settings, memory, skills, or MCP servers, and a fixed system prompt. It needs no API key. The
probe made two calls for US$0.054 in all. A 200 by 100 pixel line chart passed as a base64 image was
read correctly (the answer was 5), and an 18,016-character numeric table cost 12,559 input tokens,
or
1.44 characters per token. The command-line tool bills a prompt as cache-creation input at about
US$4 per million tokens. It has no temperature flag, so every arm shares its default sampling.

**The planned 138 calls are estimated at US$12.87, under a US$15 ceiling that a guard enforces.**
The
estimate takes the 1.15 million input tokens of the built prompts at US$4 per million and assumes
4,000 output tokens a call at US$15 per million, which is the larger part of the cost and is
unmeasured
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
setting was within it** (the least-moving setting moved on 13.8% of quiet days against the 10%
gate).
The HM2 row above is therefore the least-moving setting, and it is outside the gate. **The 23 quiet
plant-months both tune and score the gate, so "within the gate" is an in-sample result for HYST2.**
The
first study's published figures for its own estimators, scored on its ten selection draws, were 4.40
points and 2.6% of days for HYST, 2.9 points and 31.5% for HM, and 2.2 points and 63.4% for CX2,
against 5.00 points for doing nothing. The comparison is loose, because the draws differ. The
seasonal
correction cut HM's false movement from 31.5% to 13.8% and left CX2's roughly unchanged.

### Round 1 (planned): the model and the tools found almost no planted loss, and the control failed

**The model with the tools hit 0 of 32 planted losses, the model on weekly summaries hit 0, and the
tools-only rule hit 1 (a ramp).** A hit needs a drop segment within 14 days of the loss's start and
a recovery segment within 14 days of its end, each of the right sign and size. The model did write
knots on the real record, so it was not refusing to answer.

**The tools showed the loss in the monthly table but not in the changepoint list.** A planted step
of 31% in one month moved the monthly clear-sky ratio from 99 to 68, but the whole-record
changepoint scan listed a candidate within 14 days of the start for only 3 of the 32 losses. For
those 3 the model wrote no knot. In two cases it answered no change although the list held
`down -20%` and `up +29%` (one plant) or `down -32%` (another), and the third case it marked knots
elsewhere. Round 1's prompt did not ask for a reason, and its system prompt said to write a knot
"only when several tools agree on a change that lasts", which discourages marking a one-month loss.

| Planned contrast (round 1, mean absolute capacity error, points) | Estimate | 97.5% interval |
|---|---|---|
| C1: tools plus model minus tools only | +0.64 | [−0.50, +3.04] |
| C2: tools plus model minus model on weekly summaries | +1.04 | [+0.00, +3.45] |

**The positive control failed, so C1 to C3 are untested.** Of the 17 steps of 30%, the model with
the tools found 2 (11.8%), and the tools-only rule found 2, against the 80% gate. The control plants
steps in consecutive months at three plants, and the model read three consecutive planted months as
one longer loss. Rescored per run of adjacent months (post hoc), the control has 13 runs, and the
model found 5 (38.5%) and the tools-only rule found 3 (23.1%), still under the gate. C3, the false
movement on the 23 quiet plant-months, was 0.0% for the model and 6.7% [0.0%, 15.5%] for the
tools-only rule, but a labeller that marks nothing scores zero, so C3 is read with C1.

### Round 2 (post hoc): a local scan lifted the hit rate, but no arm passed the control

**Round 2 changed the tools and the prompt after seeing draws 01 and 02, so it is post hoc and
judged on two new draws (03 and 04) that nobody tuned on.** The seventh evidence table compares
each 30-day window with the 30 days either side of it, on the corrected clear-sky ratio and on the
all-sky energy ratio. The system prompt says a loss that lasts a month is a legitimate knot pair
when the scan and the monthly table agree, and asks for a one-sentence note when there is no
change. The hard checks are unchanged.

| Arm (new draws 03 and 04, 34 planted losses) | Hit rate | Mean absolute capacity error (points) |
|---|---|---|
| Round 1 tools-only rule | 0% (0 of 34) | 5.4 |
| Round 2 tools-only rule | 15% (5 of 34) | 4.3 |
| Round 1 model with tools (2 samples) | 0% (0 of 68) | 5.3 |
| Round 2 model with tools (2 samples) | 12% (8 of 68) | 4.6 |
| Model on weekly summaries | 0% (0 of 34) | 5.2 |

![Figure 2: Round 2 (post hoc). Each estimator on a planted step loss at one plant.](../assets/llm_capacity_labeller_estimators2_planted_step.svg)

| Post hoc contrast (before the hard checks, points; negative favours the first arm) | Estimate | 97.5% interval |
|---|---|---|
| Round 2 tools only minus round 1 tools only | −1.11 | [−2.28, −0.03] |
| Round 2 model minus round 1 model | −0.70 | [−1.68, +0.03] |
| Round 2 model minus round 2 tools only | +0.25 | [−0.54, +1.13] |
| Round 2 model minus the model on weekly summaries | −0.65 | [−1.64, +0.11] |

**The local scan helped the fixed rule, and the model added nothing detectable to it.** Round 2's
tools-only rule beat round 1's by 1.11 points, with an interval just excluding zero. The model with
the tools did not differ detectably from the round 2 tools-only rule. After the unchanged hard
checks, round 2's model scored worse than round 1's (+1.65 points [+0.33, +3.25]), because check 2
accepts a step only if a changepoint candidate lies within 14 days, and the local scan finds
different dates. **The checks therefore need a version for the local scan before a round 2 labeller
can be judged with them.**

**The positive control still failed in round 2.** On round 2's non-adjacent control (18 steps of
30%), the model with the tools found 10 (55.6%) with 0 false alarms, round 1's tools-only rule
found 10 (55.6%) with 4, and round 2's tools-only rule found 3 (16.7%) with 15. On the null record,
where nothing changes, round 2's tools-only rule marked 40 steps, and the model marked none. The
model's false movement on the quiet plant-months was 4.4% [0.0%, 16.2%] and the tools-only rule's
was 1.2% [0.0%, 4.7%], both within the 10% gate.

**Round 2 asked the model why it answered no change, and its reasons treat the local scan as
noise.** Two of its notes, with no plant named:

- "Apparent falls cluster in each Oct-Jan winter and repeat yearly (shared seasonal correction
  swings, low sun, afternoon shading), the corrected ratio returns to about 100 every year, and the
  hour-of-day table shows no lasting hole, so there is no lasting capacity change."
- "The local-scan episodes are mostly winter/summer seasonal-correction artifacts that recur every
  year, and the 2025-06 to 2025-07 dip is matched by the neighbour ratio (84-85) and a similar dip
in
  June 2023, so it reads as weather."

The scan lists up to 15 episodes per plant at the noisiest plants, and the model reads most of
them as weather. That judgement is partly right, because the early years of the record are noisy.

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
  read the day's check results and the largest errors against the forecast and write a few
sentences.
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

**The follow-up would give the labeller a detector built for one-month losses and a hard check that
accepts the detector's dates, then re-test the model against a fixed rule on fresh draws.** This
study learned that the tools set the ceiling: round 1's changepoint list held a candidate for 3 of
32 losses, and the local scan raised the tools-only hit rate to 15%. It also learned that the hard
checks and the evidence must share one list of candidates, and that the positive control must plant
steps in non-adjacent months. The follow-up would ask whether the model adds detection beyond a
fixed rule when the evidence is good enough that the rule passes the 80% control gate. A rough cost
is a day of agent time, no downloads, and about US$7 of model calls (this study's two rounds cost
US$6.97, a measurement). The case for it is that 3 real low-output months are the only real losses
in the record. The case against is that the model did not detectably beat round 2's fixed rule, so
the effort may be better spent on the detector alone, and 34 planted losses give wide intervals. The
proposal commits the project to nothing.

## Limitations

**The numbers on this page depend on a small sample and a shared input.** Part 1 rests on 32 planted
losses, so the intervals the planned contrasts will give are wide, and a null result there is a
finding about sample size. The planted losses last one month while a real loss persists. The six
plants share one 25 km satellite box, which is why the shared correction works, and the model's
labels
share the tools' inputs, so they cannot score the physics model's own errors. Real calendar dates
let a model draw on what it learned about UK weather during training. The satellite normalisation
mutes that and does not remove it. Round 2 is post hoc: its local scan, its threshold of 15%, and
its softer prompt followed round 1's results on draws 01 and 02, and the scan's list is dense at the
two noisiest plants. Round 2 re-ran the hard checks unchanged, and they mismatch the scan. A Fable
or Opus model would probably read a case file better than
Sonnet 5.5, and the study tests only Sonnet 5.5.

## Scope

**This page says nothing about wind, about plants outside the 6 studied, or about a forecast.** The
exploratory forecast-level check (`fit_forecast_arms.py`) divides the lag study's baseline forecast
by
each capacity series at one XGBoost setting, has no decision rule, and has not been run.

## Data and code availability

**The satellite irradiance and the lag-features study's rows are inputs that the project holds.**
NGED's power data are private, and no plant's name, identifier, or coordinates appear on this page.
The prompt and answer log pairs plant letters with dates, so it stays in the private data store and
is
never committed. The scripts are in
[`studies/llm_capacity_labeller/`](https://github.com/openclimatefix/nged-substation-
forecast/tree/main/studies/llm_capacity_labeller),
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
uv run python studies/llm_capacity_labeller/build_round2_inputs.py
uv run python studies/llm_capacity_labeller/run_round2.py --phase estimate
uv run python studies/llm_capacity_labeller/run_round2.py --phase all --confirm-spend
uv run python studies/llm_capacity_labeller/score_round2.py
uv run python studies/llm_capacity_labeller/fit_forecast_arms.py --device cuda --max-workers 4
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py
uv run python studies/llm_capacity_labeller/llm_capacity_labeller_charts.py --round 2
uv run python studies/llm_capacity_labeller/report_llm_capacity_labeller.py
```
