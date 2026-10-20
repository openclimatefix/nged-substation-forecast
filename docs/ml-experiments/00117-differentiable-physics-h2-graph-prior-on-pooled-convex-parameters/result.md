---
id: "00117"
slug: differentiable-physics-h2-graph-prior-on-pooled-convex-parameters
part: H2
theme: differentiable-physics-pooling-across-substations
also_relevant_to:
  - hierarchy-and-pooling
  - calibration-and-uncertainty
  - differentiable-physics-demand-calendar-and-weather
status: done
size: >-
  complex for the new module studies.dp_gam_graph (own pull request, reviewed by a Sonnet reviewer
  twice); medium for the scripts. Stored: private files only (cells, fits, cards data), no contract
  or production table. Serving path untouched. Degradation rule untouched. More than one design is
  defensible (what the penalty acts on, how the scale is smoothed, how the penalty is tuned), and
  the plan pre-registered one of each. Callers are this experiment's scripts only.
hypothesis: >-
  A penalty on the differences between interconnected Primaries' parameters (heating slopes, PV
  capacity as a share of effective capacity, the profile's scale, and the holiday coefficients)
  narrows those parameters' month-block bootstrap intervals without breaking their coverage when
  neighbouring Primaries are alike, and costs the full-history forecast at most 0.03 points.
prediction: >-
  R1 and R2 inside ±0.02 points with a half-width of 0.02 to 0.03; R3 (event days) −0.1 to −0.5
  points, R3b inside ±0.01. Smooth-cell coverage of H and NH 72 to 88% and within 5 points of G;
  width ratio 0.6 to 0.9 for slopes and scale, 0.8 to 1.0 for the capacity share; unlike-neighbours
  coverage 65 to 80%; the planted failure covers 40 to 60%. Real data: edge-similarity ratio 0.5 to
  0.9 for heating slopes; H's width ratio against G 0.8 to 0.95 and against H-perm 0.85 to 1.0. See
  "Predictions against outcome".
decision_rule: >-
  Supported needs the plumbing controls and the planted failure to pass, coverage within 5 points
  of G's in the smooth cell and within 10 points in the unlike cell, a width ratio of at most 0.85
  in two of three parameter families, R1 and R2 non-inferior at +0.03 points in the scored and
  replicate years with a scored half-width of at most 0.025, no winter loss (the upper end of the
  winter contrasts at most +0.03), and a real width ratio of H against H-perm below 1 for heating
  slopes. Refuted if the controls pass and H's or NH's smooth-cell coverage is under 60% and more
  than 10 points below G's at a weight that passed the harm bound, or R1 or R2 lies wholly above
  +0.03. The smallest worthwhile gain is a width ratio of at most 0.95, because a convex penalty
  adds no serving cost and only decides whether the hierarchy work builds on it.
benchmark: >-
  G, N (experiment 109's fleet mean, penalty 3,000), climatology, manual_heuristic; experiment
  109's stored N score as a reproduction check
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); tuning 2023-24, replicate 2024-25, scored 2025-26,
  each year frozen at its own 1 July cut
metric: >-
  Coverage and relative width of 80% month-block intervals per parameter family on synthetic
  fleets; R1 to R3b quantile score in points of effective capacity outside hand-filtered switching
  events and clock-change days, full history, as experiment 109 scores it; winter, evening-peak,
  and cold-snap contrasts
verdict: refuted
production_status: not-for-production
take_home_message: >-
  Tying the parameters of interconnected substations together does not improve the forecast: it
  costs about 0.02 points on average and 0.17 points in winter evening peaks, and its parameter
  intervals narrow only for heating slopes while becoming overconfident for PV capacity (17% of
  intervals hold the truth, against 78% without the tie) and for the profile's scale. A tie on
  heating slopes alone costs nothing measurable on real data (+0.006 and +0.001 points), which
  is an exploratory lead for the next experiment and not a result.
advice:
  - >-
    Put a graph penalty on heating slopes only, never on PV capacity or the profile's scale, and
    tune its weight on interval coverage in the smooth synthetic cell, not on forecast skill
    (figures 2 and 3).
  - >-
    Scale a penalty on a scalar such as the capacity sum by that scalar's own information (about
    1,200 rows for the latest capacity), not by the mean design column per row, because a weight
    of 10,000 then ties neighbouring capacities almost completely (figure 2).
  - >-
    Run the rewired-graph placebo before reading any width gain from a graph prior: H on a
    rewired graph narrowed real slope intervals as much as H on the real graph (figure 4).
  - >-
    Judge a penalised arm on winter evening peaks as well as the annual mean: H lost 0.17 points
    there while losing 0.02 overall (figure 1).
  - >-
    Score a calendar-event arm on its event days and on non-inferiority elsewhere: P-hol gained 0.099
    points on event days in the scored year and 0.018 in the replicate year, and left other days
    unchanged (figure 1).
  - >-
    Plan a gate's compute from the wall time of one whole fit on the slot's threads, not by dividing
    a per-fit time by the core count, and put coupled solves on the GPU: a 15-node fit takes 12 s
    against 118 s on the CPU and matches it to 4e-10 (disclosures).
date: 2026-10-21
pr: 1403
adopted_in: none
code: ml-experiments/00117-differentiable-physics-h2-graph-prior-on-pooled-convex-parameters/
---

# Differentiable physics, H2: a graph prior on the pooled convex model's parameters

**Tying the parameters of interconnected substations together does not help: it costs the forecast
about 0.02 points on average and about 0.17 points in winter evening peaks, and it narrows the
parameter intervals only for heating slopes.** The same tie makes the intervals of PV capacity and
of the profile's scale badly overconfident (capacity: 17% of the 80% intervals hold the truth,
against 78% without the tie). A tie on heating slopes alone costs nothing measurable on real data
(+0.006 and +0.001 points against fitting each substation alone), which makes it an exploratory lead
and not a result. The pre-registered verdict is **`refuted`**.

**The question.** The pooled convex demand model of [experiment
109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md) predicts a Primary
substation's demand from the calendar, the weather, and a PV term. A parameter such as a heating
slope (how much demand rises per 10 °C of cold, in each part of the day) comes from two years of
half-hourly readings, and its 80% interval says how well those readings fix it. [Experiment
113](../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md) found
heating slopes known only to within about 38% of their size. This experiment asked whether a penalty
on the differences between the parameters of interconnected substations narrows those intervals
without breaking their coverage (the share of intervals that hold the truth), and without hurting
the forecast by more than 0.03 points.

**How we tested it.** We tested the penalty twice. First on synthetic fleets of 15 substations,
built from the real calendar, weather, and PV shape of five Primary substations, with a known truth
and noise taken from real residuals (120 fleets per kind of truth, 40 in each of three sets), asking
how often each 80% interval holds the truth and how wide it is. Second on the real data: the five
decision substations (D5, D10, D11, D14, and D16) fitted jointly with D9 and eight substations that
train on night-time readings only, on a graph of 14 substations and 21 interconnections. Each fit is
frozen at 1 July and scored over the next 12 months (2023-24 tunes the weight, 2024-25 is the
replicate year, and 2025-26 is the scored year).

**What we found.** At the tuned weight the penalty narrows heating-slope intervals to 0.59 of the
width of fitting each substation alone, but they cover the truth 70% of the time against 76% without
the penalty, and the intervals of profile scale (56% against 75%) and PV capacity (17% against 78%)
fail worse. On real data the interconnection graph is informative (neighbours' heating slopes are
about half as far apart as random pairs'), yet a randomly rewired graph with the same degrees
narrows the intervals just as much, so the narrowing is shrinkage and not neighbours being alike.

## Terms

- **Points:** percentage points of a substation's effective capacity, so 1 point is 1% of the
  capacity. A heating slope is in points per 10 °C of cold.
- **Quantile score:** twice the mean pinball loss over 11 quantiles of the forecast, in points;
  smaller is better. Contrasts are the first arm minus the second, so a negative number favours the
  first.
- **Arms:** **G** fits each substation alone. **N** pools every substation's parameters towards a
  fleet mean with a penalty of 3,000 rows of evidence (the choice of experiment 109). **H** ties
  interconnected substations' heating slopes, profile, holiday coefficients, and PV capacity with a
  penalty of 10,000 rows of evidence per interconnection. **NH** has both penalties. **P-hol** pools
  only the holiday coefficients of the five decision substations to a fleet mean (10,000 rows).
  **H-perm** is H on a graph rewired at random with every substation's number of interconnections
  kept.
- **Event days:** the days of the holiday periods the model carries (each bank holiday, Easter, the
  bridge days, and the Christmas to New Year run), without the clock-change days. **Winter:**
  November to March. **Evening peak:** half-hours that start from 16:00 to 19:30 UTC. **Cold-snap
  day:** a winter day whose mean temperature is at or below the substation's 10th percentile of
  winter days in the two years before the cut.
- **Smooth cell** and **unlike cell:** synthetic fleets whose true parameters are alike across
  interconnections (correlation 0.8) or drawn independently per substation.

## Figures

**Figure 1 shows the graph prior costing a little on average and more in winter evenings.** In the
scored year H minus G is +0.020 points over all days, +0.055 over November to March, and +0.173 over
the winter evening peak. The holiday fleet mean (P-hol) gains 0.099 points on event days in the
scored year and leaves other days unchanged.

![Dot chart: eleven contrasts in the scored and replicate years; the graph prior is above zero for
winter and winter evening peaks, and the holiday fleet mean is below zero on event
days.](fig_contrasts.svg)

**Figure 2 shows the graph prior leaving scale and capacity intervals overconfident.** Over 120
smooth-cell fleets, H covers the true heating slope 70% of the time (G: 76%), the profile scale 56%
(G: 75%), and the PV capacity 17% (G: 78%). The fleet mean alone (N) also under-covers heating
slopes, at 59%.

![Dot chart: coverage of the 80% intervals for G, N, H, and NH, by parameter family, in the smooth
and the unlike cell; H and NH fall far below G for capacity and
scale.](fig_gate_coverage.svg)

**Figure 3 shows the narrowing being confined to heating slopes.** H's median slope interval is 0.59
of G's width, and NH's is 0.81 of N's. Scale and capacity intervals narrow by 5 to 7% at most, and
the holiday interval widens to 1.14 times G's.

![Dot chart: interval width as a share of the reference arm's, by parameter family; only the heating
slopes fall clearly below 1.](fig_gate_width.svg)

**Figure 4 shows a rewired graph narrowing real slope intervals as much as the real graph does.** In
the scored year H is 0.89 of G's width for heating slopes and the three rewired graphs give 0.83 to
0.90; H is 1.03 times the rewired graphs' width (1.04 in the replicate year).

![Dot chart: real interval widths against G for N, H, NH, and H on three rewired graphs, in the
scored and the replicate year.](fig_real_width.svg)

**Figures 5 and 6 give every arm's absolute score.** G is best or level in both years, and every
arm is far ahead of climatology and the manual heuristic. In the scored year N, H, and NH are ahead
of G on cold-snap days (4.04 to 4.06 against 4.20) and on event days (2.87 to 2.93 against 3.05),
and behind it in winter and in the winter evening peak.

![Dot chart: absolute quantile scores by arm and kind of day, scored year
2025-26.](fig_absolute_scored.svg)

![Dot chart: absolute quantile scores by arm and kind of day, replicate year
2024-25.](fig_absolute_replicate.svg)

## Results

- **Verdict `refuted`, by the pre-registered coverage rule.** The controls pass and the planted
  failure is detected, and at a weight that passed the harm bound H's smooth-cell coverage is
  under 60% and more than 10 points below G's for profile scale (56.3%, G 75.3%) and PV capacity
  (17.1%, G 78.0%), and so is NH's (54.5% and 16.0%). Readings (a) (coverage within 5 points of G's)
  and (c) (unlike-cell coverage within 10 points of G's) fail, and reading (b) fails because only
  heating slopes narrow to 0.85 or less (H 0.59, NH 0.81; scale 0.94 and 0.98, capacity 0.95 and
  0.97). **Winter would have capped the verdict anyway:** R1w's upper end is +0.093 and R2w's is
  +0.055 in the scored year, both above +0.03, so the hold-back rule applies.
- **Real-data contrasts (98.75% interval for R1 to R3b, 95% otherwise; scored year, then replicate
  year).**
    - **R1, H minus G:** +0.020 (−0.014 to +0.059) and +0.014 (−0.009 to +0.039). Non-inferiority at
      +0.03 is not shown, and the scored half-width (0.036) is over the 0.025 limit, so it is
      undecidable. H is better than G at 2 of 5 substations in both years.
    - **R2, NH minus N:** +0.013 (−0.012 to +0.041) and +0.019 (+0.008 to +0.032).
    - **R3, P-hol minus G on event days:** −0.099 (−0.282 to +0.002; 3,361 rows on 22 event days)
      and −0.018 (−0.071 to +0.024). **R3b, other days:** +0.000 (−0.002 to +0.002) and −0.001
      (−0.005 to +0.002). The evidence is thin: each substation holds 1 or 2 days per holiday period
      in a year. The scored-year gain on event days is at the edge of the −0.1 to −0.5 predicted,
      and the replicate year is outside it.
    - **Winter (R1w, R2w) and evening peak:** H minus G is +0.055 (+0.010 to +0.093) and +0.020
      (−0.006 to +0.046) over November to March, and +0.173 (+0.090 to +0.253) and −0.021 (−0.085 to
      +0.045) over the evening peak; NH minus N is +0.031 (+0.009 to +0.055) and +0.031 (+0.014 to
      +0.049) over November to March. **Cold-snap days:** H minus G is −0.165 (−0.414 to +0.043) and
      −0.005 (−0.137 to +0.129), NH minus N is +0.016 and +0.153 (+0.105 to +0.214); the cold-snap
      evidence is 11 to 13 days a site-year and the intervals are wide.
    - **Context:** N minus G is +0.058 (+0.022 to +0.102) in the scored year and −0.004 in the
      replicate year (N pools 14 nodes here, against 7 in experiment 109, whose scored-year C1 was
      +0.031), and H minus N is −0.037 (−0.079 to −0.003) in the scored year.

- **Absolute scores (quantile score in points; five decision sites, outside switching events and
  clock-change days).**

    | Arm | Scored, all days | Scored, winter | Scored, winter evening peak | Replicate, all days |
    |---|---|---|---|---|
    | G | 2.883 | 3.36 | 3.45 | 3.166 |
    | N | 2.941 | 3.49 | 3.63 | 3.163 |
    | H | 2.903 | 3.41 | 3.62 | 3.180 |
    | NH | 2.954 | 3.52 | 3.78 | 3.181 |
    | P-hol | 2.879 | 3.35 | 3.45 | 3.165 |
    | Manual heuristic | 3.484 | 4.10 | 3.91 | 3.843 |
    | Climatology | 4.065 | 4.65 | 4.49 | 3.904 |

- **The synthetic gate (smooth cell; 120 fleets; coverage of 80% intervals).**

    | Family | G | N | H | NH | H width / G | NH width / N |
    |---|---|---|---|---|---|---|
    | Heating slopes | 76.1% | 58.7% | 70.1% | 60.9% | 0.59 | 0.81 |
    | Profile scale | 75.3% | 69.5% | 56.3% | 54.5% | 0.94 | 0.98 |
    | PV capacity | 78.0% | 72.9% | 17.1% | 16.0% | 0.95 | 0.97 |
    | Holiday level (not gated) | 49.2% | 49.3% | 53.6% | 47.3% | 1.14 | 0.72 |

    In the unlike cell H covers 56.1% (slopes), 32.8% (scale), and 14.0% (capacity) against G's
    74.9%, 72.2%, and 77.3%. The wrong-graph cell (truth smooth on a different graph) gives H the
    same 69.0%, 49.8%, and 19.2% against G's 76.3%, 76.1%, and 77.7%. **The planted failure** (gamma
    100,000) covers 44.7% of slopes, 23.0% of scale, and 13.4% of capacity in the unlike cell, 30.1,
    49.2, and 63.9 points below G (20.8, 15.6, and 32.1 standard errors), so the gate can see
    over-shrinkage. Correlated noise (report only): 40 fleets per cell with a smooth truth and
    neighbour noise correlation 0.3 and 0.6. H covers heating slopes 75.3% and 70.3% against G's
    76.2% and 75.4%, profile scale 53.6% and 48.0% against 81.4% and 73.0%, and capacity 18.2% and
    19.5% against 77.7% and 77.5%, with slope width 0.58 of G's in both. Correlated noise does not
    rescue the scale or capacity intervals.

- **The next weight up (gamma 30,000, exploratory; the plan's next grid value).** In the smooth cell
  H covers slopes 66.0%, scale 46.5%, and capacity 16.6% (slope width 0.53 of G's), and on real data
  H minus G is +0.048 (+0.013 to +0.086) in the scored year and +0.031 (+0.007 to +0.056) in the
  replicate year (95% intervals); NH minus N is +0.038 and +0.041. A larger weight narrows slopes
  more, lowers every coverage, and costs more forecast skill.
- **Why capacity fails.** The capacity term acts on the sum of a substation's capacity increments,
  a single number carried by about 1,200 rows of information, while the penalty scales with the mean
  design column. At 10,000 rows per interconnection it ties neighbouring capacities almost
  completely; a graph penalty on capacity alone gave a bias of 4.3 points against 0.85 (four
  fleets, exploratory).
- **Tuning (before the gate).** Gamma was chosen by the plan's rule: every gamma from 0 to 30,000
  scored within one standard error of the best on the synthetic interval score, and the real
  tuning-year harm bound (+0.02) kept every gamma up to 10,000, so gamma = 10,000. Lambda_hol =
  10,000 is the top of its grid.
- **Real parameter intervals.** Neighbours are alike in heating slopes (edge-similarity ratio 0.46
  to 0.57 across the four bands in the scored year, 0.32 to 0.79 in the replicate year), less in the
  profile scale (0.88, 1.17) and alike in holiday level (0.35, 0.33). H's heating-slope width is
  0.89 (scored) and 0.70 (replicate) of G's, the rewired graphs' 0.80 to 0.90 and 0.60 to 0.71, and
  H over the rewired graphs is 1.03 and 1.04, so the real graph adds nothing over any other graph
  with the same degrees. G's residuals correlate 0.24 and 0.29 along interconnections.
- **Tails (all rows, scored year; coverage of the 0.01, 0.05, 0.95, and 0.99 quantiles).** G: 0.7%,
  2.7%, 84.1%, 93.9%; H: 0.7%, 2.7%, 83.9%, 93.7%; the other arms are within 0.3 points. The upper
  tail is under-covered by about 11 points at q95 and 5 at q99, as in experiment 109, whose q95
  shortfall the plan attributed to the spread coming from one replica year. Pinball loss at q95 is
  0.852 (G) and 0.864 (H).
- **Exploratory lead (not a result; the plan listed these arms as exploratory).** A graph penalty on
  heating slopes alone (H-heat, gamma 10,000) costs +0.006 (−0.011 to +0.026) in the scored year and
  +0.001 (−0.017 to +0.018) in the replicate year against G (95% intervals), and +0.021 (−0.011 to
  +0.048) and +0.012 in winter. On the synthetic fleets it leaves scale (75.5%) and capacity (81.6%)
  coverage at G's level and narrows slope intervals to 0.59 of G's, but still covers slopes only
  68.0% of the time (smooth cell; G 75.7%) and 55.0% (unlike cell; G 72.9%). A graph penalty on the
  holiday coefficients alone is level with G (−0.001 and +0.000).

## Predictions against outcome

- **R1 and R2 inside ±0.02 with a half-width of 0.02 to 0.03:** R1 +0.020 and +0.014 (half-width
  0.036 and 0.024); R2 +0.013 and +0.019 (0.026 and 0.012). Held except R1's scored half-width.
- **R3 −0.1 to −0.5 and R3b inside ±0.01:** R3 −0.099 and −0.018; R3b +0.000 and −0.001. R3b held;
  R3 held at the edge in the scored year only.
- **Smooth-cell coverage of H and NH 72 to 88% and within 5 points of G:** failed (H 70%, 56%, 17%;
  NH 61%, 55%, 16%). **Width ratio 0.6 to 0.9 for slopes and scale, 0.8 to 1.0 for capacity:**
  slopes held (0.59, 0.81), scale did not (0.94, 0.98), capacity held (0.95, 0.97).
  **Unlike-neighbours coverage 65 to 80%:** failed (56%, 33%, 14% for H). **Planted failure covers
  40 to 60%:** slopes 44.7% (held), scale 23.0% and capacity 13.4% (lower).
- **Real data, edge-similarity ratio 0.5 to 0.9 for slopes:** held in the scored year (0.46 to
  0.57), mixed in the replicate year (0.32 to 0.79). **H's width ratio against G 0.8 to 0.95 and
  against H-perm 0.85 to 1.0:** 0.89 held and 0.70 did not; 1.03 and 1.04 against H-perm did not.

## Checks

- **Median forecast minus mean forecast, per site and arm:** 0.000 at every decision site for G, N,
  H, NH, and P-hol, because the spread is centred on the median. This check cannot show a shifted
  spread here; the mean-forecast error beside each contrast is the informative check.
- **Error of the mean forecast beside each contrast (scored year):** R1 +0.036, R2 +0.030, R3
  −0.166, R3b −0.002, R1w +0.090, R2w +0.073, evening peak +0.312 (H minus G). Every sign agrees
  with the quantile-score contrast except R3b's (+0.000 against −0.002, both inside ±0.002).
- **Audit triggers (ml-experiment skill).** Fired: **2** (H-perm matches H on real slope widths,
  1.03 and 1.04 times, while the smooth cell shows a narrowing; explained by the placebo itself),
  **3** (R3b's two signs differ inside ±0.002), **4** (dropping the D10–D11 interconnection turns
  R2's reading from undecided to non-inferior; dropping D10 puts R1's whole interval above zero in
  the scored year; no other site, interconnection, or night-time substation changes a reading), and
  **5** (NH is worse than both N and H in both years). Not fired: **1** (all five controls pass and
  the planted failure sees over-shrinkage), **6** (R1 and R2 half-widths are 0.5 to 1.6 times
  experiment 109's C1 half-width of 0.022), **7** (a negative result: no later design builds on it).
  No Opus audit was run: a negative result merges on the Sonnet reviews alone, by the maintainer's
  ruling, and the keep-first-run rule applies.

## Disclosures

- **Controls at the scored cut.** The plumbing controls P1 to P4 (parameter comparisons between arms
  only) and P5 (re-scoring experiment 109's published scored-year N) ran fits at the 2025-07-01 cut
  before the module review. The manager accepted them as plumbing controls. No H, NH, or P-hol
  score at the replicate or scored cut was computed or seen until the real-data stage, which ran
  after tuning and beside the gate, with the manager's approval.
  P5 reproduces experiment 109's N score exactly (2.929030 against 2.929030 on 104,033 rows), using
  109's pool of seven series.
- **Code under test.** The tuning set, the main gate (sets 1 to 3 and the correlated-noise cells),
  and the real-data stage ran on module commit `c008d9f4b` (`studies.dp_gam_graph`, module pull
  request 1411); the tail passes that started after the merge ran on the merged module (merge commit
  `dbbfd2e84`, whose later changes count LU fallbacks, guard the joint size, and add tests, and
  alter no numeric path). The first fleets of set 1, the 10-fleet power check, and some tail passes
  ran on the CPU path (`67deaa053` and `c008d9f4b`); the CPU and CUDA solves agree to 4e-10, and the
  Cholesky solver equals the LU solver to 1e-7 on a 15-node fleet (a module test).
- **Clarifications made before any fit, after seeing the module or the power check** (all in the
  planning commits of this branch): a relative identification rule for profile and holiday columns;
  centring over the coordinates both ends identify; the per-edge meaning of the weight; dark rows as
  CAMS irradiance below 1 W/m²; the node set of N, NH, and P-hol; the winter breakdown; and the
  reading of the power-check sizing rule at the pooled size.
- **Compute.** The plan estimated 9 to 10 hours of CPU time; each 15-node fit takes about 3 seconds
  on 6 threads, so the gate would have taken about 33 hours on the CPU. A Cholesky solver and a GPU
  path brought it to about 9 hours. The next-gamma arms were restored in sets 2 and 3 after the GPU
  made the labelled shortcut unnecessary.
- **Not done.** No onboarding or short-history arm; no 0.97 power-factor sensitivity; the
  clock-change days are excluded from every score and the fits train on them as before.

## Substation cards (provisional)

**Provisional: fitted to past weather on a screening run.** Heating slopes are known to about 38%
of their size at real noise (experiment 113); read them as weakly identified. The panels show each
decision substation's parameters with 80% month-block intervals for G, N, H, and NH, and the
example weeks of G and NH. PV capacity is pinned to the real-time vintage at these substations, so
the graph prior's capacity term does not act on them and no capacity is printed. The four example
weeks were fixed before any score: Summer 14 July 2025, Autumn 13 October 2025, Winter 12 January
2026, and Spring 13 April 2026; by the rule fixed in advance, a week with a bank holiday or a
hand-filtered event at any decision substation moves to the next Monday, which moved Summer to 21
July 2025 and Winter to 19 January 2026.

![Dot chart: the five decision substations' heating slopes, profile scale, and holiday level, each
fitted alone, with 80% intervals.](card_cross_site.svg)

**D5.**

![Dot chart: D5's heating slopes, profile scale, and holiday level for G, N, H, and NH, with 80%
intervals.](card_D5_parameters.svg)

![Line chart: D5's observed power and the forecast median with 80% and 90% bands of G and NH in four
example weeks.](card_D5_weeks.svg)

**D10.**

![Dot chart: D10's heating slopes, profile scale, and holiday level for G, N, H, and NH, with 80%
intervals.](card_D10_parameters.svg)

![Line chart: D10's observed power and the forecast median with 80% and 90% bands of G and NH in
four example weeks.](card_D10_weeks.svg)

**D11.**

![Dot chart: D11's heating slopes, profile scale, and holiday level for G, N, H, and NH, with 80%
intervals.](card_D11_parameters.svg)

![Line chart: D11's observed power and the forecast median with 80% and 90% bands of G and NH in
four example weeks.](card_D11_weeks.svg)

**D14.**

![Dot chart: D14's heating slopes, profile scale, and holiday level for G, N, H, and NH, with 80%
intervals.](card_D14_parameters.svg)

![Line chart: D14's observed power and the forecast median with 80% and 90% bands of G and NH in
four example weeks.](card_D14_weeks.svg)

**D16.**

![Dot chart: D16's heating slopes, profile scale, and holiday level for G, N, H, and NH, with 80%
intervals.](card_D16_parameters.svg)

![Line chart: D16's observed power and the forecast median with 80% and 90% bands of G and NH in
four example weeks.](card_D16_weeks.svg)

**How each arm does in a cold snap (scored year; days are cold-snap days outside hand-filtered
events).** Quantile score in points and the share of observed values inside the 80% interval:

| Substation | Days | G | N | H | NH |
|---|---|---|---|---|---|
| D5 | 3 | too few days | too few days | too few days | too few days |
| D10 | 13 | 5.02 (54%) | 5.14 (52%) | 5.11 (56%) | 5.44 (53%) |
| D11 | 13 | 4.16 (51%) | 3.90 (51%) | 3.79 (54%) | 3.76 (52%) |
| D14 | 11 | 3.36 (48%) | 2.96 (52%) | 2.98 (55%) | 2.80 (58%) |
| D16 | 4 | too few days | too few days | too few days | too few days |

The 80% intervals cover only about half of the values on cold-snap days in every arm. The evidence
holds two independent winters (2024-25 and 2025-26); the 2023-24 winter tuned the weights.

## What the next experiment should answer

- **Does a graph penalty on heating slopes alone, with a weight tuned for coverage and not for
  forecast skill, keep slope intervals calibrated?** The heating-only arm cost +0.006 and +0.001
  points on real data, and spared scale and capacity, but still covered slopes 68% of the time. A
  weight in the range of 300 to 3,000 rows is untested for it on real data.
- **A penalty on a scalar such as PV capacity has to be scaled by the scalar's own information.**
  The mean design column per row overstates a scalar's evidence by a factor of about 40 for the
  capacity sum, so the weight read as rows of evidence is not rows of evidence for it.
- **The hierarchy work should not wait on a parameter graph.** The real graph adds nothing over a
  rewired graph with the same degrees, and the penalty hurts the winter evening peak; a
  forecast-level hierarchy (a bulk supply point informed by its Primaries) is untouched by this
  result.
