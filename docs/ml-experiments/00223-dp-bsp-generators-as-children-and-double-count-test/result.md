---
id: '00223'
slug: dp-bsp-generators-as-children-and-double-count-test
theme: hierarchy-and-pooling
also_relevant_to:
- generator-forecasting
- differentiable-physics-network
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00223-dp-bsp-generators-as-children-and-double-count-test/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible (how to test
  whether a generator's output is already in a Primary's reading, and which generator forecast to
  sum), and this plan pre-registers one of each. Callers: this folder's scripts, which read
  experiment 204's stored forecasts and call the reviewed `studies.*` modules without changing them.
hypothesis: >-
  The metered generators that NGED lists under a bulk supply point (BSP) are children of that BSP
  as much as the Primaries are: the BSP's reading is the Primaries' demand plus the unmetered load
  minus the generators' output. If the Primaries' readings already include a generator's output,
  adding that generator to the sum double-counts it. This experiment first tests, from the readings
  alone, which generators the BSP's remainder carries and which a Primary already carries, and then
  asks how much the BSP's forecast improves when the generators the BSP carries are summed as
  children.
prediction: >-
  Written before any score is read. (1) The two high-voltage generators (11 kV, so downstream of a
  Primary) show a remainder coefficient near 0, if they hold any data. (2) The wind generators, the
  biomass generator, and the battery show remainder coefficients with intervals that include 1
  (output carried by the BSP's remainder), because their output swings are large and distinct. (3)
  The solar generators cannot be separated one by one, because their output is near-collinear, so
  they are tested as a group, and the group coefficient's interval includes 1 at the BSP that has
  the larger solar capacity. (4) Observed generator output in place of a forecast (the oracle)
  lowers the BSP's quantile score by 5% to 25% against experiment 204's arm (d) at D17 and by 5% to
  15% at D13. (5) A climatology forecast of the generators lowers it by 0% to 3%.
decision_rule: >-
  Two separate rows, reported apart so experiment 204's verdict rules stay intact. Row A, double
  counting: for each generator and each group, regress the change from one half-hour to the next of
  the BSP's remainder (the BSP's reading minus the weighted sum of its child Primaries' readings,
  weights fixed from the winter of 2023-10 to 2024-03) on the change of the generator's output, over
  half-hours before 2025-07-01, with the interval from resampling whole days (20,000 draws, 97.5%
  level, Bonferroni for the number of generators and groups tested). A generator is `carried by the
  BSP` if its interval for the coefficient on its output (sign such that 1 means the BSP's reading
  falls by the generator's output) includes 1 and excludes 0, `inside a Primary` if the interval
  includes 0 and excludes 1, and `cannot tell` otherwise. The same regression of each Primary's
  reading on each generator, whose coefficient is -1 where the generator sits behind that Primary
  and 0 elsewhere, is a second check, and a disagreement between the two is reported. Row B, forecast
  gain: the C1 contrast of experiment 204 (bottom-up minus top-down at each BSP, month-resampled
  paired bootstrap, 97.5% level) for the arm that sums the generators classified `carried by the
  BSP`, with the oracle and the climatology forecast as two arms. Row B is `supported` for an arm
  if its interval at one BSP lies wholly below minus 5% of the top-down score and the same arm's
  contrast against experiment 204's arm (d) has a point estimate below zero. It is `inconclusive`
  otherwise, and `refuted` if both BSPs' intervals lie wholly above zero. A failed known-answer
  check caps both rows at `inconclusive`.
benchmark: >-
  Top-down (arm D's chain on the BSP alone) and experiment 204's arm (d); climatology and
  manual_heuristic for absolute skill
row_set_id: >-
  The shared rows of experiment 204 (`bsp_full_v1`) restricted to rows where every generator in the
  arm has a reading
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  the BSP's effective capacity over leads 0.5 to 168 hours; the mean absolute error of the mean
  forecast; coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00223-dp-bsp-generators-as-children-and-double-count-test/
---

# Experiment 223: are the metered generators under a bulk supply point children of the sum?

This page is the plan. Results are added after the run.

## Question

[Experiment 204](../00204-dp-bsp-full-bottom-up-with-signed-mva-children/result.md) summed every
child Primary of D13 (Skegness) and D17 (Boston) and found no detectable gain. NGED's hierarchy file
also lists 13 metered generators under the two BSPs. The file gives each generator its BSP but no
Primary, so a generator may connect straight to the BSP or through a Primary. A generator behind a
Primary is already in that Primary's reading, and summing it again would count its output twice.
This experiment tests which generators the BSP's remainder carries, from the readings, and then
adds those generators to the bottom-up sum.

## The generators

| Group | Count | Unit | Voltage evidence | Treated as |
|---|---|---|---|---|
| Wind | 3 (labels W1 to W3 in other pages) | 2 in MVA, 1 in MW | 33 kV (EHV) | tested |
| Solar | 7 | MW | 6 at 33 kV, 1 at 11 kV | 6 tested as a group; the 11 kV one is Primary-side |
| Biomass | 1 | MW | 33 kV | tested |
| Battery | 1 | MW | 33 kV | tested |
| Other generation | 1 | MW | 11 kV | Primary-side |

The generators are labelled G1 to G13 in a private table; no page names one or gives an identifier.
Of the 13, one 11 kV solar series holds no readings and the other 11 kV series has under 4 months, so
neither can be tested. The generators' readings are in the cleaned power table in MW (generation
positive) except the two MVA wind series, which are magnitudes of apparent power and so are read
positive. A battery's reading is signed.

**Primary-side by voltage.** An 11 kV generator must connect downstream of a Primary, so it is
already inside that Primary's reading. The plan treats the two 11 kV generators as Primary-side and
leaves them out of every sum. The 11 EHV generators could connect at the BSP's 33 kV busbar or by a
33 kV feeder to a Primary, so voltage cannot decide them and the readings must.

## Row A: does the remainder carry the generator?

Under the hypothesis that the BSP carries a generator, the BSP's reading falls by that generator's
output, and nothing in the Primaries' readings changes. Under the hypothesis that a Primary's raw
reading carries it, the remainder (the BSP minus the weighted Primaries) does not move with the
generator, and that Primary's reading falls by the output instead. Experiment
[206](../00206-dp-meter-regime-steps-e1-farm-outages-in-primary/result.md) found that D10's reading
did not change when two metered solar farms' meters read zero, which says those farms do not sit
behind D10; it could not test the other Primaries.

- **Regression.** The change from one half-hour to the next of the remainder on the change of each
  generator's output, with the remainder built as in experiment 204 (weights from the winter of
  2023-10 to 2024-03, all children with a long history). Taking changes removes the slow terms that
  both sides share. Wind, biomass, and the battery are regressed one by one. The six EHV solar
  farms share almost all of their variation (the same sun), so they enter as their sum, and as two
  sums by BSP.
- **Second check.** Each Primary's reading on each generator's output; the coefficient is -1 behind
  that Primary and near 0 elsewhere.
- **Interval.** Resampling whole days, 20,000 draws, the 97.5% level with a Bonferroni correction
  for the number of generators and groups tested.
- **Known-answer check.** Add a generator's observed output to the remainder (so the remainder
  carries it by construction) and to a randomly chosen Primary (so the Primary carries it by
  construction), and check that each regression classifies the construction correctly for at least
  four of five generators of each kind. A regression that cannot recover a planted generator cannot
  rule one out.

## Row B: does summing the generators help the BSP forecast?

Three arms, each experiment 204's arm (d) plus the EHV generators classified `carried by the BSP`
as extra children with weight 1 (a generator's output is metered, so no weight is estimated), summed
with the sign the BSP's reading implies. The remainder chain is refitted on the new remainder.

| Arm | Generator forecast |
|---|---|
| G-oracle | The observed output (a control: the most that any generator forecast could give) |
| G-clim | A climatology of the output by local month, half-hour of the day, and weekend flag, fitted before the scored year |
| arm (d) | Experiment 204's arm (d), the reference |

A generator forecast from the generator-forecasting theme would replace the climatology if one
existed for the scored year's 291 daily runs. None does: experiment 87's stored forecasts cover 53
runs and 9 generators. The experiment says so and recommends a generator forecast as the next step if
the oracle gains.

## Controls, run before any contrast is read

- Reproduction of experiment 204's arm (d) and top-down scores to three decimals.
- The known-answer check of Row A.
- Leakage: no weight, climatology, or regression coefficient used in Row B reads a half-hour on or
  after the first scored run (the Row A regression is a diagnostic of the network and is fitted on
  half-hours before 2025-07-01 only).

## Triggers and size

- **Stored:** private files only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, two, each pre-registered above.
- **Callers not nameable without searching:** no, the scripts read experiment 204's stored files.

## Resources

CPU only, at most 4 cores with every thread cap at 1, each job under `systemd-run --user --scope -p
MemoryMax=16G -p OOMPolicy=continue`. No GPU. The remainder chains reuse experiment 204's copy of
the chain.

## Advice followed and departures

- **Followed:** experiment 204's advice to aim the next hierarchy experiment at the remainder (the
  generators and the unmetered load beneath each BSP); the data-processing page's entry on MVA
  series.
- **Departure:** the generators are summed with weight 1 and no estimated weight, because their
  output is metered.
- **Departure from a forecast arm:** no generator forecast exists for the scored year, so the oracle
  and a climatology stand in.
