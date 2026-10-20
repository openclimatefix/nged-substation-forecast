---
id: "00117"
slug: differentiable-physics-h2-graph-prior-on-pooled-convex-parameters
part: H2
theme: differentiable-physics-pooling-across-substations
also_relevant_to:
  - hierarchy-and-pooling
  - calibration-and-uncertainty
  - differentiable-physics-demand-calendar-and-weather
status: planned
size: >-
  complex for the new module studies.dp_gam_graph (own PR, two code reviews); medium for the
  scripts. Stored: private files only (cells, fits, cards data), no contract or production table.
  Serving path untouched. Degradation rule untouched. More than one design is defensible (what the
  penalty acts on, how the scale is smoothed, how the penalty is tuned), and this plan pre-registers
  one of each. Callers are this experiment's scripts only.
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
  "Predictions".
decision_rule: >-
  See "Decision rule": supported needs the plumbing controls and the planted failure to pass,
  readings (a) to (c), R1 and R2 non-inferior at +0.03 points in the scored and replicate years with
  a scored half-width of at most 0.025, and a real width ratio of H against H-perm below 1 for
  heating slopes. The smallest worthwhile gain is a width ratio of at most 0.95, because the
  penalty adds no serving cost and only decides whether the hierarchy work builds on it.
benchmark: >-
  G, N (experiment 109's fleet mean, penalty 3,000), climatology, manual_heuristic; experiment
  109's stored N and G scores as a reproduction check
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); tuning 2023-24, replicate 2024-25, scored 2025-26,
  each year frozen at its own 1 July cut
metric: >-
  S1 relative 80% interval width and coverage per parameter family on synthetic fleets; R1 to R3
  quantile score in points of effective capacity outside hand-filtered switching events, full
  history, as experiment 109 scores it
production_status: not-for-production
code: ml-experiments/00117-differentiable-physics-h2-graph-prior-on-pooled-convex-parameters/
---

# Differentiable physics, H2: a graph prior on the pooled convex model's parameters

**Status: planned. No fit has run.** This page is the plan, committed before any code runs; the
result and write-up replace it later.

## The question, and what the data allow

**Experiment 109's fleet mean says every Primary resembles the average Primary; the graph prior
says each Primary resembles the Primaries wired to it.** Experiment 109 pooled five Primaries
towards one fleet mean and lost 0.03 points on full histories. It never asked whether the pooled
parameters are better known, which matters for the substation cards (an interval on each fitted
slope and capacity) and for the hierarchy work after this part. Experiment 113 found that at the
real noise (standard deviation 4.97 points, autocorrelation 0.83) heating slopes per hour band have
an 80% interval about 38% of the truth and capacity 12%. A graph prior is useful only if it narrows
those intervals while coverage stays near G's.

**The real graph carries little capacity signal, so the capacity family is a synthetic-stage reading
and descriptive on real data.** Only D5–D16 and D10–D11 have both ends identifying capacity in the
tuning and replicate years; every other edge of the 21-edge decision graph has an end that trains on
dark rows only, whose capacity is not identified. Heating slopes and the profile's scale have all 21
edges, restricted to the hour bands each end identifies. **Holiday coefficients** rest on 1 or 2
days per site and their error is mostly a level offset shared across sites; they enter the graph
term and also get their own arm (below).

**Arms, the same rows and cache, fits frozen at the cut with no refit in the year.**

| Arm | Fleet-mean penalty (rows) | Graph penalty (rows) | Other |
|---|---|---|---|
| G | 0 | 0 | per-site, `dp_gam` |
| N | 3,000 (109's choice) | 0 | |
| H | 0 | gamma | |
| NH | 3,000 | gamma | |
| P-hol | 0 (holiday family only: lambda_hol) | 0 | the holiday family pools to a fleet mean over the five decision sites only (no MVA node, no D9) |

H and NH include the holiday family in the graph term. P-hol is the labelled holiday arm.
**lambda_hol is tuned on the tuning year over {0, 300, 1,000, 3,000, 10,000} by the quantile score
of P-hol on the tuning year's event days of the five decision sites (event days are defined under
R3; clock-change days excluded); it is the largest value whose score is within one event-day
standard error of the lowest.** The harm bound of the gamma rule does not apply, because P-hol is a
gain arm and R3 and R3b never change the verdict.

**Clock-change days are excluded from the scored rows of every arm and every contrast** (G, N, H,
NH, P-hol, climatology, `manual_heuristic`; R1, R2, R3, R3b, and every descriptive score), so all
pairs stay on the same rows. In the scored year the excluded days are 26 October 2025, 28 March
2026, and 29 March 2026. In the replicate year they are the matching autumn change day, the spring
change day, and the day before it (27 October 2024, 29 and 30 March 2025), and in the tuning year
likewise (29 October 2023, 30 and 31 March 2024). The fits themselves train on those days as before.

## Builds on

- [Experiment 109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md) (the
  fleet-mean penalty N, its gate, and its advice), [experiment
  113](../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md)
  (bootstrap intervals at realistic noise), and [experiment
  105](../00105-differentiable-physics-3-pooled-and-per-site-demand-model/result.md) (pooling at
  full history).
- [Experiment 103](../00103-differentiable-physics-1-data-look/result.md) for the rows, the start
  cuts, and the hand-filtered switching events.
- The technique pages on [differentiable physics](../../techniques/differentiable-physics.md) and
  [convex optimisation](../../techniques/convex-optimisation.md).
- [Hallac, Leskovec, and Boyd (2015)](https://doi.org/10.1145/2783258.2783313), who penalise
  the differences between the parameters of neighbouring nodes of a graph (the network lasso). Here
  the penalty is a squared difference, so the fit stays a convex quadratic program.

## Graph

**The decision graph is the 21 metered edges of the Network line's interconnection file, without the
two excluded series (G-c).** The script counts nodes and edges from the interconnection file and
commits the counts, and the number of MVA nodes with 12 months of dark rows before each cut, before
reading any residual. An MVA node joins a cut only if it holds those 12 months. MVA nodes are
counted, never named, on the page.

**An edge penalises a coordinate only if both ends identify it:** a node identifies a coordinate if
the rows its fit trains on make the column non-zero in at least 1,000 rows (heating and capacity
columns), or in at least 5% of the median non-zero count of its own family's non-zero columns
(profile and holiday columns). **This is a clarification made after seeing the module and before any
fit; the Opus review of the module will rule on it.** The plan's single rule of 1,000 rows would
leave profile columns (one half-hour of one day type, non-zero in about 100 to 500 rows over two
years) and holiday columns (48 to 96 rows) with no identified coordinate, and a one-row test would
count a column that is non-zero in a handful of twilight rows. A share of the family's own median
count scales with the window length. **The profile's centring runs over the coordinates both ends
identify,** so a level shift of an hour band is never penalised, as in the fleet-mean penalty. **The
graph weight is per edge:** a substation with `d` edges is pulled `d` times as hard as one with one
edge, and a complete graph on `S` substations with weight gamma equals the fleet-mean penalty with
weight `S` times gamma, so the grid of gamma values is read against N's 3,000 with the graph's mean
degree of about 2.8 in mind. An MVA node trains on dark rows (CAMS irradiance below 1 W/m², the
series rule for dark rows; this replaces the plan's elevation-plus-margin wording, made before any
fit), so its daytime-band heating columns and its capacity are not identified. D5–D16 and D10–D11
carry capacity and daytime bands; the other edges carry night-band heating, profile, and holiday
terms. Parameters are in points of effective capacity. An MVA node's effective capacity is built
from pre-cut rows only (experiment 109's advice); the 0.97 power-factor rescale is a sensitivity.

**The node set of every graph arm, and of N and NH, is the cut's graph:** the decision sites, D9
where it holds rows, and the MVA nodes that join (clarification before any fit). N pools its fleet
mean over the same nodes as NH, so R2 differs from N by the graph term alone. Experiment 109's own
pool of seven series is used only for the reproduction control P5, and P-hol pools over the five
decision sites alone, as the arm table says.

**The penalty is convex.** For the heating, profile, and holiday families the graph term is

    gamma * g_f * sum over edges e=(i,j) of ||C_f (theta[i, f] - theta[j, f])||^2

with `g_f` and `C_f` those of `studies.dp_gam_pooled`, so gamma reads as rows of evidence as lambda
does. Capacity is a share (fitted capacity over effective capacity) on edges whose two ends both
identify it. Non-negativity bounds stay; the fit is a box-constrained quadratic program with a
unique optimum. **The profile's scale is smoothed through the centred profile vector, so the
profile's shape is smoothed too:** a scalar scale times a shared shape would be non-convex. The
scale is read out as the root-mean-square of the centred profile. Drift is none and day-of-year
harmonics are in, because the pooled fit requires it (`PooledSite`); this departs from experiment
113's G-full, so S1 reads widths against G-H, whose slope widths 113 found to be about 38%.

## Module

**`studies.dp_gam_graph` (own PR, two code reviews, tests under 5 minutes):** a graph builder from
the interconnection file (edges, identified-coordinate masks, counts), a Laplacian block added to
`dp_gam_pooled`'s normal equations, the box-constrained solve, a joint month-block bootstrap (the
same calendar months drawn at every node, quarters balanced as experiment 113's I1 does), and the
scale readout. G runs per node (separable). **Plumbing controls run first; an unfixable failure
makes the verdict `inconclusive`:**

- P1: with gamma = 0 and lambda = 0, H equals `dp_gam.fit_gam` per node to 1e-8; with gamma = 0, NH
  equals N; with no edges, H equals G and NH equals N;
- P2: on a 3-node path graph the solver matches a dense box-constrained program to 1e-6, and a
  permutation of node order changes no fit to 1e-8;
- P3: as gamma grows, two connected nodes' identified parameters converge to a common value;
- P4: changing any reading, residual, or weather value after a cut leaves that cut's fits unchanged
  at every node, so a neighbour's later data never reaches a scored site; spreads and `sigma` use
  pre-cut rows;
- P5: N reproduces experiment 109's C1 score on the five decision sites to 0.005 points. Experiment
  109's own verdict was `inconclusive`, so P5 reproduces a number that is itself unreplicated.

## The known-answer stage (S1) and its gate

**The generator is experiment 113's, extended to a graph.** Synthetic fleets reuse the real
calendar, weather, and PV shape of the five decision substations, three copies each, so 15 nodes, on
a synthetic graph of 21 edges (a ring plus 6 random chords, a fixed seed per fleet) with 4 random
nodes labelled MVA (dark rows only). The synthetic graph is not the real graph. **No row after the
2024-07-01 cut enters the generator:** calendars, weather, donor residuals, and the real fits behind
the truths all come from rows before it. Truths follow 113 (heating totals U(0.7, 1.3) times the
real fit's, floor 2 points per 10 °C; capacity U(5, 25) points then U(1, 5) increments; a profile
scale U(0.7, 1.3) on the mean of the five real centred shapes; holiday coefficients from the real
fits with a node-level offset). Noise is 113's decision noise, whole donor months with a random
sign, **drawn jointly across nodes** (one draw of donor months per fleet, shared by every node, each
node taking a different donor site).

**Truth cells, 40 fleets per set:**

- **Smooth:** each family's per-node value is a fleet mean plus a Gaussian Markov random field on
  the synthetic graph, neighbour correlation about 0.8 (computed and committed before any fit).
- **Unlike-neighbours:** values drawn independently per node (renamed to avoid clashing with 113's
  independent-noise cell).
- **Wrong graph:** values smooth on a different random graph; reported, for H and NH only.
- **Correlated noise (report-only, one set):** the smooth cell with neighbour noise correlation
  0.3 and 0.6 (a shared component of weight sqrt(rho) plus an own component), for G and H.

**Every cell runs the no-graph control G, and N, with the same bootstrap.** G's own coverage is the
reference: 113's G-full covered capacity only 52% at independent noise because of its drift penalty,
and this experiment has no drift, so G's coverage is measured, not assumed. Intervals are 80%
month-block bootstrap, B = 40 resamples; a posterior interval covered 20% in 113 and is not used.
Coverage is clustered by fleet.

**Only the plumbing controls and the planted failure cap the verdict at `inconclusive`.** Coverage
and width readings on H and NH are results that choose a verdict row. The gate:

1. P1 to P5 pass.
2. **Planted failure (the line can fail):** H at gamma = 100,000 (off the grid, so a tuned gamma
   cannot hide it) on the unlike-neighbours cell covers at least 10 points below G in the same cell
   for at least one family. Otherwise the gate cannot see over-shrinkage (109's line (b) could not).
3. **Power check:** the first step runs 10 fleets and prints the fleet-clustered standard error of
   each coverage and the planted failure's coverage. The gate is called sized if the standard error
   is under 4 points and the planted failure sits at least 3 standard errors below G; otherwise
   the cells grow to 60 fleets.

**Readings that decide the row (reported per set; the verdict uses the pooled three sets):**
(a) smooth-cell coverage of H and NH is within 5 points of G's in the same cell, and within 70 to
90% when G's is; (b) smooth-cell median width ratio against G (for H) and N (for NH) is at most
0.85 for at least two of the three **width families: heating slopes, the profile's scale, and the
capacity share** (the holiday family is reported, not gated, because its coefficients rest on 1 or 2
days per site), with a fleet-clustered 98.3% interval wholly below 1 (98.3% matches three width
families after Bonferroni); (c) unlike-neighbours coverage within 10
points of G's; (d) the bias under a quarter of the interval width.

## Tuning, years, and the real-data stage

| Period | Cut | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: gamma, lambda_hol; harm bound |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

**Gamma is chosen in this order, before the gate runs.** (1) On a separate tuning set of 20
synthetic smooth-cell fleets (its own seeds), score H at each gamma in {0, 300, 1,000, 3,000,
10,000, 30,000} by the 80% interval score (width plus a penalty for missing the truth). (2) Bound
harm on the real tuning year: keep only gammas whose H minus G and NH minus N quantile score on the
five decision sites is at most +0.02 points. (3) Gamma is the largest gamma that passes (2) and
whose synthetic interval score is within one standard error of the best from (1). A real-score-only
rule would favour 0 or 300 (experiment 109's penalty was chosen on short histories and cost +0.031
on long ones) and narrow nothing. If only gamma = 0 passes, the verdict follows `not supported`. The
gate and the scored years then run at that gamma and at the next grid value above it; if the chosen
gamma is 30,000, there is no next value, only that gamma runs, and the top of the grid is a
trigger-5 audit item. The harm bound compares point estimates; it is a screen, not a test.

**Scored sites.** Decision sites D5, D10, D11, D14, D16 with full history (24 months); D9 beside
them. MVA nodes are neighbours; their dark-row scores are descriptive and unlabelled by D-label. The
spread is the replica fit's residual spread, one fit 12 months before each cut. No onboarding or
short-history arm (maintainer scope).

## Planned contrasts

**Four real-data contrasts, Bonferroni-corrected, 98.75% intervals from `studies.bootstrap`
(3 seeds, paired across arms).** R1, R2, and R3b resample whole calendar months. **They use the
joint draw: each resample picks the same calendar months at all five decision sites,** so the
correlation between neighbouring sites is kept, as in the parameter bootstrap; the contrast is the
mean over sites. If `studies.bootstrap` draws months per site, the script supplies the month
indices. R3 resamples whole event days (below). Each contrast reports sites moving each way, every
arm's absolute score, the mean-forecast error beside it, and the replicate year.

- **R1, the graph prior alone does no harm:** H minus G, non-inferior if the upper end is at most
  +0.03 points in the scored year and in the replicate year.
- **R2, the graph prior on top of the fleet mean does no harm:** NH minus N, same rule.
- **R3, holiday pooling on the event days (primary):** P-hol minus G on the **event days**, the days
  of the holiday periods that the holiday coefficients cover (each bank holiday, Easter, the bridge
  days, and the Christmas to New Year run, as `studies.dp_calendar.HOLIDAY_PERIODS` lists them),
  without the clock-change days. P-hol gains if the interval lies wholly below zero. The interval
  resamples whole event days (all 48 half-hours of a day together), with the same days drawn at
  every decision site, within each holiday period so that every period keeps its size. **The
  evidence is thin:** each site holds 1 or 2 days per holiday period in a year, and the script
  prints the number of event days per site and year before any score is read. The annual average
  is reported and is not decisive.
- **R3b, holiday pooling leaves the other days alone:** P-hol minus G on all days that are not
  event days and not clock-change days, non-inferior if the upper end is at most +0.02 points in the
  scored year. P-hol counts as a gain only if R3 gains and R3b is non-inferior. R3 and R3b never
  change the verdict.

**What the margin buys.** Both R1 and R2 are nested pairs, so their half-width is likely near
experiment 109's C1: about ±0.025 at 99% and ±0.022 at 97.5%, so about ±0.025 at the 98.75% level
these contrasts use (scaled by the normal quantiles 2.50 and 2.24), below Part 4's C4 of ±0.023 for
two filtered arms at its level. The 0.025 limit for `supported` is therefore nearly binding, and a
scored half-width above it is reported as undecidable, not as a failure. Porting a convex penalty
adds no serving cost, so the 0.03 margin only decides whether the next experiment builds on H.
Experiment 109's N minus G was +0.031, so a penalty costing as much as the fleet mean fails R1. If
the scored year's half-width exceeds 0.025, non-inferiority is stated as undecidable at this noise
level.

**Real-data width role (descriptive, with a placebo).** From the unpenalised G fits the script
prints the **edge-similarity ratio**: the mean squared parameter difference along true edges over
the mean over random pairs, per family (below 1 means neighbours are alike). It reruns H on 3
degree-preserving **permuted graphs** (H-perm) and prints H's real interval width ratio against
H-perm and against G; a width gain that H-perm shares is the prior shrinking variance, not
neighbours being alike. It also prints G's residual correlation along true edges (the real
counterpart of the correlated-noise cell). A real width ratio of H against H-perm below 1 for
heating slopes is required by `supported`.

**Also descriptive:** N minus G (reproduces 109's +0.03), NH minus G, H minus N; D9; the 0.97
sensitivity; real interval widths per arm (the cards' numbers). **Exploratory at 95%:** the next
gamma in each direction; heating-only and holiday-only graph terms.

**Tails, every arm:** coverage and pinball at q01, q05, q95, q99 on all rows and per site, from
`studies.tail_calibration`; predictions carry `time_series_id`, `q01`, `q05` to `q95`, `q99`,
`actual`, `lead_hours`. No curtailment cost. **Comparators:** every arm's absolute score,
climatology, `manual_heuristic`, 109's stored N and G.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **R1 and R2:** inside ±0.02 points, with a half-width of 0.02 to 0.03. **R3 (event days):** −0.1
  to −0.5 points, because the event days are about 4% of a year and an annual gain of 0.01 to 0.03
  would then sit on them; **R3b:** inside ±0.01.
- **S1:** smooth-cell coverage of H and NH 72 to 88% and within 5 points of G; width ratio 0.6 to
  0.9 for slopes and scale, 0.8 to 1.0 for the capacity share; unlike-neighbours coverage 65 to 80%;
  the planted failure covers 40 to 60%.
- **Real data:** edge-similarity ratio 0.5 to 0.9 for heating slopes; H's real width ratio against G
  0.8 to 0.95 and against H-perm 0.85 to 1.0; capacity changes by under 5% (2 edges).

## Decision rule

**Only a failed plumbing control or planted failure caps the verdict at `inconclusive`.**

- **`supported`:** the controls pass; readings (a), (b), (c) pass; R1 and R2 are non-inferior in
  both years with the scored half-width at most 0.025; R1w and R2w pass the hold-back rule under
  "Winter breakdown"; and the real width ratio of H against H-perm is below 1 for heating slopes.
  Without the real-width condition the take-home is worded as synthetic-only and the verdict is
  `partly supported`.
- **`partly supported`:** the controls pass and (a) and (b) pass, but R1 or R2 fails in the
  replicate year only, is undecidable, or the real placebo does not separate H from H-perm.
- **`small gain`:** the controls pass, R1 and R2 are non-inferior in both years, and the smooth-cell
  median width ratio is 0.85 to 0.95 in at least two of the three width families with reading (a)
  holding.
- **`not supported`:** the controls pass, R1 and R2 are non-inferior in both years, and either the
  smooth-cell median width ratio is above 0.95 in at least two of the three width families (any
  interval, so [0.93, 1.01] at a median of 0.97 lands here), or the ratio is at most 0.95 but
  reading (a) fails with coverage still at least 60% or within 10 points of G's (narrowing bought by
  losing coverage is not support), or only gamma = 0 passed the harm bound.
- **`refuted`:** the controls pass and either R1 or R2 lies wholly above +0.03 in the scored year,
  or H's or NH's smooth-cell coverage is under 60% and more than 10 points below G's (at a gamma
  that passed the harm bound).
- **`inconclusive`:** a failed control, or anything else (for example R1 or R2 straddling +0.03 in
  the scored year).

**Verdict wording** states the capacity family as synthetic-only. R3 and R3b are reported beside the
verdict and never change it.

## Winter breakdown (pre-fit clarification, 2026-10-20)

**The forecast goes to production in about a month and NGED first sees it in winter, so every
contrast is also read on the winter rows, and a winter loss holds an arm back.** This section was
added before any fit, at the maintainer's request. It adds three things to the plan above and
changes none of the other rules.

- **Winter rows:** valid times from 1 November to 31 March (UTC) in the scored and replicate years,
  without the clock-change days. Two row sets sit inside them: the **evening peak**, the half-hours
  from 16:00 to 19:30 UTC, and the **cold-snap days**.
- **Cold-snap rule (fixed before any score is read):** a winter day is a cold-snap day at a site if
  its mean ERA5 temperature is at or below that site's 10th percentile of winter daily means over the
  24 months before the cut. The script prints the number of cold-snap days per site and year before
  any score is read. A site-year with fewer than 5 cold-snap days is reported as "too few days" and
  gets no interval.
- **Winter contrasts:** R1w (H minus G) and R2w (NH minus N) on all winter rows, with a 98.75%
  interval from resampling whole weeks inside the winter (a winter holds only 5 calendar months, too
  few blocks for a month resample), the same weeks drawn at every decision site. The evening-peak and
  cold-snap rows are reported with the same intervals where there are enough days, and descriptively
  otherwise. R3w reports P-hol minus G on the winter event days.
- **Hold-back rule:** if R1w or R2w has its upper end above +0.03 points in the scored year, the
  verdict is capped at `partly supported` and says that the arm is held back for winter, even if
  the annual mean improves. A winter loss in R3w holds P-hol back in the same way.
- **Independent winters:** the evidence holds two independent winters, 2024-25 (replicate) and
  2025-26 (scored). The 2023-24 winter tunes gamma and lambda_hol, so it is not evidence. Every
  winter reading is therefore descriptive of two winters, and the page says so.

## Substation cards (provisional)

**Each decision site's card gets a "graph prior" panel**, labelled "Provisional: fitted to past
weather on a screening run", with this line: "Heating slopes are known to about 38% of their size
at real noise (experiment 113); read them as weakly identified." It shows heating slopes, scale,
holiday level, and for G, N, H, and NH their 80% month-block intervals. **Capacity is printed only
if its own coverage reading passes in the smooth cell (G's coverage within 70 to 90%)**; otherwise
the panel says "not printed: coverage not established". Example forecasts for one fixed week per
season of the scored year, chosen before scoring:

| Season | Week (Monday to Sunday) |
|---|---|
| Summer | 2025-07-14 to 2025-07-20 |
| Autumn | 2025-10-13 to 2025-10-19 |
| Winter | 2026-01-12 to 2026-01-18 |
| Spring | 2026-04-13 to 2026-04-19 |

Each week shows the median and 80% and 90% bands of NH and G against observed power, 09:00 UTC
day-ahead origins, in points of effective capacity. **Fallback rule, fixed before scoring:** if a
week holds a bank holiday or a hand-filtered event at any decision site, take the next Monday and
record it. Figures follow the `dataviz` skill and the card style of experiments 109 and 111.

**Privacy: D-labels only.** The page names no substation, no topology beyond interconnected
D-label pairs such as D5–D16 and D10–D11, and does not say which D-labels are MVA. The two excluded
series (G-c) and their sum are generator data: never beside a D-label, name, or identifier, no
dates, own-capacity units, `aria=False`; this plan puts no G-c series on a card and has no
arm that uses them. No generator public name appears with a series. The synthetic graph may be
shown; the real graph is shown only as the allowed D-label pairs.

**Cold-snap panel (added with the winter breakdown).** Each decision site's card gets a "how it does
in a cold snap" panel: the quantile score and the 80% and 90% interval coverage of G, N, H, and NH
on the site's cold-snap days of the scored year, with the day count printed, or "too few days" where
the rule above leaves fewer than 5.

## Audit before merge: the seven triggers

1. **A control fails:** any plumbing control, the planted failure not failing, or the power check.
2. **A contradiction:** width falling while coverage rises; NH better than both H and N with gamma
   on a grid edge; H-perm matching H on real widths while the smooth cell shows a gain.
3. **Quantile score against mean error:** R1, R2, R3, or R3b with opposite signs in the two.
4. **Leave one out:** dropping D5–D16, D10–D11, any decision site, or any MVA node flips R1, R2, or
   a reading.
5. **Nested arms:** NH worse than both N and H; H at gamma 0 not equal to G; gamma at the grid top.
6. **Interval width:** R1 or R2 half-width over twice or under half experiment 109's C1 width.
7. **Dependence:** the hierarchy work builds on a positive verdict, so any `supported` or
   `partly supported` triggers an audit.

"Checks" bullets in the write-up: median minus mean forecast per site and arm, the mean-forecast
error beside each contrast, and which triggers fired.

## Advice followed, caveats, and compute

**Experiment 109's advice followed:** penalty of 3,000 rows for the fleet mean in N and NH; capacity
of a part-window node rebuilt from the rows it holds; the gate repeated with a line that can detect
over-shrinkage and a multi-seed rule (three sets); the upper tail distrusted and reported. **Not
followed:** none. **Caveats carried over:** the tuning year's capacity vintage is known 5 weeks
after its cut (a small leak, as in 109); the replica spread uses one stale replica year, which gave
109 an 83% q95 coverage, so tail results are reported with that caveat.

**Order:** (1) module PR; (2) this plan commit; (3) P1 to P5 and the timing and standard-error
measurement on 10 fleets; (4) the synthetic tuning set and the real tuning-year harm bound, choosing
gamma and lambda_hol; (5) the gate on 3 sets of 40 fleets; (6) scored and replicate years; (7)
cards, figures, audit, write-up in the short format, leading with the take-home message. Steps 6 and
7's scored fits are skipped if the controls fail.

**Compute, CPU only: 12 cores across slots B and C (two workers of 6 threads), about 9 to 10 hours
of wall clock.** Per gate set: smooth 6 arm-cells (G, N, H and NH at two gammas), unlike-neighbours
6, planted 1, wrong graph 2, so 15 cells times 40 fleets times 41 joint fits (one fit plus 40
resamples), about 25,000 fits, and 74,000 over three sets. At about 4 seconds a fit, 82 hours of
fits, 6.9 hours on 12 cores; G is separable and cheaper. The correlated-noise cells (one set), the
tuning set, the lambda_hol grid fits (5 values, under half an hour), and the real stage (about 100
joint fits at 2 to 5 minutes) add 2 to 3 hours. **If the power check grows the cells to 60 fleets,
the fit count and the wall clock rise by half, to about 13 hours, and the session tells the manager
before starting.** **Pre-registered thinning if the timed fit exceeds 4 seconds:** B drops to 30 and
the correlated-noise cells drop to rho = 0.6. The gate keeps its three sets. Agent time about 2 days
(module and two reviews about 6 hours).
