# Estimating the money a better forecast saves

> **Status: 🚧 Planned (v0.3).** Epic:
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6); issue:
> [#606](https://github.com/openclimatefix/nged-substation-forecast/issues/606). This page is the
> plan for two leaderboard metrics that express forecast skill in pounds. It is written to be read
> by anyone numerate. Assumptions marked as open are awaiting confirmation from NGED. See the
> [roadmap index](index.md) for status conventions.

## Read this first: these metrics only exist to rank ML models; they are too simplistic to be used for anything more

These metrics put a **£** figure against every model we train. That figure rests on an invented
network limit and a couple of average prices, so it is a **rough proxy**: not a cost analysis, not a
business case, and not quotable as either. Its one job is to rank forecasts on the axis NGED care
about, turning "this model has a lower threshold-weighted continuous ranked probability score" into
"this model would have reduced costs". We use pounds rather than a unitless score because the
parameters genuinely are prices, and because pounds are what the decisions these forecasts feed are
actually made in. Forecast developers do not usually tune on money: [Gürses-Tran and Monti
(2022)](https://doi.org/10.3390/forecast4020028) observe that forecast developers "predominantly
assess residuals and error statistics when tuning the targeted model's quality". As a result,
"eventual cost or rewards of the underlying business application are typically not considered in the
model development phase".

## Two savings, measured separately

A better forecast creates value through two different mechanisms, with different prices and
different beneficiaries. We compute them as **two metrics, reported as two numbers**, and never add
them up:

1. **Flexibility procurement.** NGED pay flexible customers to reduce demand when a site risks
   running beyond its limit. NGED procures flexibility conservatively as a safeguard against
   forecast uncertainty, so a sharper forecast needs less flexibility for the same security. Metric
   1 measures NGED's own spend on flexibility.
2. **Curtailment of generation.** Generators are curtailed to keep a substation's export within
   its limit.
   Curtailment avoided is generation sold, priced as a whole-system cost rather than a saving to
   NGED or the connected generator specifically — see [curtailment price
   basis](#curtailment-price-basis).

A third saving — the engineer-hours freed by replacing the manual heuristic with an automated
forecast — is real, but it is **not a leaderboard metric**: it is identical for every model we
train, so it cannot rank them. This third saving belongs in the project's final report, priced in
engineer-hours.

## The shared idea: same risk, then compare the spend

The textbook way to price a forecast charges it for what goes wrong: £X per action taken, £Y per
limit breach nobody saw coming. We cannot follow that route, because £Y — the cost of a breach — has
no single figure a leaderboard can use. Pricing a breach is a business judgement rather than a
property of a forecast. And in the literature that does price a forecast this way, we found no case
of it being done on a real distribution network at a money-denominated cost. The [energy-forecasting
review](../background/energy-forecasting-review.md#evaluating-the-performance-of-power-forecasts)
reports [Richardson (2000)](https://doi.org/10.1002/qj.49712656313)'s cost-loss framing, [Bernecker
et al. (2025)](https://doi.org/10.1016/j.ijepes.2025.110713)'s 97% synthetic-network saving, and
[Angus et al. (2027)](https://doi.org/10.1016/j.epsr.2026.113545)'s and [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335)'s capacity- and volume-denominated results.

So we invert the question. **Models are aimed at equal safety, and we compare what each one spends
to get there.** Each model may be as conservative as it likes, and we tune that conservatism until
it leaves the same small amount of risk unaddressed. Then the only thing left to compare is cost.
This matches the framing the design is built on: the risk to manage is unnecessary spend, not an
unaddressed breach.

The knob is the **procurement quantile** $\tau$ — how far up its own forecast distribution a model
looks when deciding to act. A timid model uses a high $\tau$, buys a lot, and is rarely caught out.
Calibration picks each model's $\tau$ so that its **unmet fraction** — the share of genuinely-needed
megawatt-hours it failed to cover — hits a common target (5% to begin with):

$$ \text{unmet fraction} = \frac{\sum_{i,t} \max(0,\; N_{i,t} - V_{i,t})}{\sum_{i,t} N_{i,t}} $$

where $V_{i,t}$ is the volume the model would have bought (or curtailed) for time series $i$ in
half-hour $t$, and $N_{i,t}$ is the volume that turned out to be needed. Measuring unmet *energy*
rather than counting missed events matters: exceedances are rare by construction, and a count of
them is too noisy to rank models by.

**$\tau$ is calibrated on the training window of the leaderboard fold, never on the validation
window it is scored on** — otherwise a model sees its own future and every pound of the "saving" is
lookahead. This has a methodological drawback: the training window is data the model was fitted to,
so its residuals are smaller than they will be out of sample. As a result, $\tau$ comes out too low,
and every model under-procures on the scored window. The model that overfits hardest gains most from
this.

**Equal risk is therefore a target, not a guarantee, and this is the design's main weakness.** What
a model *realises* on the scored window is whatever its tail calibration delivers there. A model
that overshoots the target spends less and can top the leaderboard while being materially less safe.
The **realised out-of-sample unmet fraction is reported beside every cost, and a cost read without
it is meaningless.** Two models are only comparable on cost when their realised unmet fractions are
close.

## What the volumes cost

Flexibility is bought in two parts, and the distinction is the whole point of these metrics.
**Availability** is paid on every megawatt-hour held ready, whether or not it is called;
**utilisation** is paid only on what is actually dispatched. Over-procurement therefore costs the
*availability* price on the excess, not the far larger utilisation price:

$$ C = p_{\text{avail}} \sum_{i,t} V_{i,t} \;+\; p_{\text{util}} \sum_{i,t} \min(V_{i,t},\, N_{i,t})
$$

The second term is nearly identical for every model — it is set by what the network actually needed.
So the ranking is carried by the first. Charging one blended price against all procured volume would
overstate the cost of over-procurement several times over.

This formula only holds for **short-term contracts**, tendered day-ahead against a forecast. NGED
also buy **long-term contracts**, whose availability is tendered roughly a year ahead of delivery,
independent of any forecast we produce now. A better forecast cannot reduce that volume, because it
is already committed by the time our forecast exists. Only the **utilisation** decision on
long-term-covered volume — whether to call on capacity already secured, decided day-ahead (up to 5
days ahead around a weekend or bank holiday) — is forecast-sensitive. Metric 1 therefore has two
components, scored separately per `(time_series_id, direction)`:

- **Long-term-covered volume**: only the utilisation term is scored against $N$. Availability spend
  is fixed regardless of forecast and is excluded from the metric.
- **Short-term volume**: the full formula above applies unchanged.

Both prices, for both contract types, are configuration: we compute a volume-weighted average price
per contract type from National Grid's published flexibility-trades data (see [implementation
details](#implementation-details-deleted-when-this-ships)) rather than hardcoding a placeholder, and
NGED's own rates can replace that computed price without a retrain once confirmed — see [questions
for NGED](#questions-for-nged).

## Metric 1 — flexibility procurement cost

For time series $i$ and half-hour $t$, with demand-side limit $L_i$:

| Quantity | Definition |
|---|---|
| Volume procured | $V_{i,t} = \max(0,\; \hat q_{i,t}(\tau) - L_i) \times 0.5$ MWh |
| Volume needed | $N_{i,t} = \max(0,\; y_{i,t} - L_i) \times 0.5$ MWh |

$\hat q_{i,t}(\tau)$ is the model's $\tau$-quantile forecast, $y_{i,t}$ the observed power, and
$\times 0.5$ converts MW held for a half-hour into MWh. For a half-hour covered by a long-term
contract, $V_{i,t}$ and $N_{i,t}$ are unchanged but only the utilisation term $p_{\text{util}}
\min(V_{i,t}, N_{i,t})$ is charged.

**Worked example.** A substation whose limit sits at 30 MW, on one winter evening half-hour, with
short-term availability at £75/MWh and utilisation at £750/MWh (both placeholders, pending real
volume-weighted prices). Forecast A (more conservative) says 33 MW, so it procures $(33 - 30) \times
0.5 = 1.5$ MWh. Demand turns out to be 30.6 MW, so 0.3 MWh was needed. Forecast A pays £112.50
availability and £225 utilisation. Forecast B says 31.0 MW, so it procures 0.5 MWh and pays £37.50
availability and the same £225 utilisation — £75 less than Forecast A. If the same half-hour were
instead covered by a long-term contract, both forecasts pay only the utilisation term (£225 each)
and Forecast B saves nothing on that half-hour. The metric sums this over every half-hour and every
series, split by which contract type covered each half-hour.

## Metric 2 — curtailment cost

**Curtailment is not decided per meter: a generation constraint binds at a substation, driven by
the aggregated flow through a bulk supply point (BSP) or grid supply point (GSP).** Scoring each
`time_series_id` independently, as Metric 1 does for flexibility, would misrepresent how
curtailment gets triggered. Metric 2 therefore scores forecasts against the limit of each BSP and
GSP, in two tiers of increasing realism. Both keep the "these numbers rank models, they do not cost
anything" disclaimer above. Neither is a validated £-saved figure without a real case study behind
it.

### Tier 1 — substation limits, no power-flow

**The node metric asks one question per half-hour: will a BSP or GSP export more than its firm
limit, and if so, which generators beneath it must be capped, and what does that cost in pounds?**
A generator is capped only when the forecast says the substation's export would exceed the
substation's limit. Two tables hold every fact about the electricity network. The node table has
one row per BSP or GSP, with its parent, its export sign, and its limits in each season. The
generator table has one row per generator, with its capacity, connection date, and whether a
forecast exists. Scoring the ~2,500 series due in January 2027 therefore needs longer tables and no
new code. Primaries are out of scope until NGED supplies the topology.

**The limit is the firm (N-1) limit: the smaller of the per-transformer pre-event limits of the
substation's redundant transformer pair.** One transformer carries the full load when the other is
out. The limit is set per NGED season, and each season is a set of whole calendar months: Winter is
December to February, Intermediate Cool is March, April, and November, Intermediate Warm is May,
September, and October, and Summer is June to August. The import limit follows the same rule and is
used only to count import exceedances.

**Net export is an explicit sign times our own metered flow.** The sign `export_sign` is +1 where a
positive flow is export, -1 where a positive flow is import, and null where the sign is unconfirmed.
A node with a null sign is excluded and counted, never guessed. Our own cleaned power data is
authoritative for flows and signs. NGED's public portal supplies only the published limits and the
season boundaries, because the portal's flows are sums of several meters and can carry the wrong
sign.

**The rule plans a cut from the forecast, shares the cut among generators, and charges what the
forecast got wrong.** For each half-hour with a forecast lead of at least 12 hours and under 36
hours:

1. **Planned node curtailment** is
   $v_{\text{node}} = \max(0, q_\tau(\text{export}) - L) \times 0.5$ h, in MWh, where $L$ is the
   limit in MW and $q_\tau(\text{export})$ is the $\tau$ quantile of net export in MW. For a node
   with sign -1 that is minus the $1 - \tau$ quantile of the metered flow. The breach ratio $r$ is
   fixed at 20, so $\tau = 1 - 1/r = 0.95$, and the manual heuristic is scored at the same $\tau$.
2. **The generators with a forecast share $v_{\text{node}}$ pro rata by capacity.** A share larger
   than a generator's forecast energy is clamped to that energy, and the excess goes to the
   unclamped generators by capacity, repeating until nothing is clamped. Any remainder is
   unallocated and shows up as residual breach. A generator with no forecast is never capped,
   because capping it from its actual output would leak the outcome. This holds half-hour by
   half-hour, so a node row is still scored when a generator's forecast is missing at that
   half-hour, with the generator giving no relief. A generator with a forecast but no metered
   output is capped as normal and its energy loss is counted as 0 MWh, so a missing output can
   understate the curtailed energy. The allocation order is a
   stated assumption. Last-in-first-out by connection date is a sensitivity only where connection
   dates exist.
3. **A generator with a share is capped at its forecast median minus its share**, $\text{cap}_g =
   \max(0, f_g - v_g / 0.5\text{ h})$ in MW, and loses $c_g = \max(0, a_g - \text{cap}_g) \times
   0.5$ h in MWh against its metered output $a_g$. The generator forecasts enter the score here.
4. **The residual breach** is $\max(0, \text{actual export} - \sum_g c_g / 0.5\text{ h} - L)
   \times 0.5$ h, in MWh.

**Import-side exceedances are counted, not priced.** Capping generators cannot relieve an import
constraint. [Metric 1](#metric-1-flexibility-procurement-cost) prices that side. The output reports
import-exceedance half-hours per node and season.

**Two limits are reported, as co-equal headlines.** The real firm limit is what NGED enforces, and
it binds on very few half-hours, so a score at the real limit demonstrates the arithmetic more than
it ranks forecasts. The scaled limit is defined for every node, including those with no known real
limit: $L = f \times$ the node's maximum observed export (the negative tail for a node whose
positive flow is import), computed only from history before the scored period. The fraction $f$ is
0.9 by default, with 0.75 and 0.5 swept. "p90" here means 0.9 times the maximum, not the 90th
percentile of the node's flow, because a percentile of a node's own flow is exceeded on 10% of
half-hours by construction. A scaled-limit figure is a calibration to history, not a saving. An
event census of export and import exceedances at the real limit and at each $f$ comes first, and it
decides whether a ranking of forecasts is possible.

**Caveats.**

- **No reactive power, voltage drop, or N-1 contingency modelling beyond the firm limit.** A sum of
  active power is not a load-flow study; Tier 2 is where that modelling belongs.
- **Nesting is not modelled.** A GSP whose BSPs also have limits is scored node by node. Passing a
  cut from a parent to its children waits for confirmed signs at both levels.
- **Coverage gap.** Only some of the generators beneath a node have a forecast in the trial area.
  This gap shrinks as trial coverage scales towards the ~2,500-series v2 rollout.
- **The capacity of each generator is an estimate.** The forecasts are fractions of capacity,
  converted to MW with each series' effective capacity, so the error in that estimate carries into
  every cap.

### Tier 2 — full power-flow modelling

Out of scope for Flexpectation, including v2. Tier 2 is documented as the eventual correct approach
— modelling reactive power, voltage drop, and N-1 contingencies explicitly — but it is gated on
power-flow integration work that sits outside this project.

Tier 2 is also where a **real curtailment case study** becomes possible. Tier 1 can be computed and
can rank models without Tier 2. What Tier 1 cannot yet do is be validated against a real curtailment
event, because no existing site maps a curtailment case to a specific series or a single hierarchy
node — see [case studies](#case-studies) below.

**NGED derives upstream limits by combining substation forecasts in a power-flow model, so the
forecasts must stay consistent across substations.** Percentiles cannot be added across substations:
the sum of each substation's 95th percentile is generally not the 95th percentile of the combined
flow. An ensemble member describes one weather outcome at every substation at once, so ensemble
members can be fed through a power-flow model member by member. The power-flow model is therefore a
reason to keep delivering the [ensemble of deterministic
forecasts](delivery-tables.md#representation-1-ensemble-of-deterministic-forecasts) alongside the
percentile representations.

**Tier 2 is not gated on the speed of the power-flow solve.** [Nguyen et al.
(2026)](https://doi.org/10.48550/arXiv.2608.25095) train a self-supervised surrogate for multiphase
alternating-current (AC) optimal power flow. That is the dispatch optimisation layered on top of a
power-flow solve, and Tier 2 needs only the solve. The surrogate still shows how fast the harder of
the two problems can run. On feasible, in-distribution test cases it stays within 0.2% of the cost
the interior-point solver IPOPT reaches. It runs one to three orders of magnitude faster than
IPOPT, reaching 780 and 1,121 times on the two largest IEEE test feeders it was tried on, which
range from 13 to 8,500 nodes. That comparison times batched inference on a graphics processing unit
against IPOPT on a central processing unit. Nguyen et al. do not time the power-flow solve
separately from the optimisation. Their results give no reason to think the solve is too slow to
push an ensemble through member by member.

**NGED already publishes an electricity-network model, so the gate is integration work rather than
a missing model.** Nguyen et al.'s method needs three inputs: per-branch impedances, per-branch
thermal ratings, and voltage limits. NGED publishes the impedances and the thermal ratings as open
data, in its Long Term Development Statement. See [data
sources](data-sources.md#ngeds-published-electricity-network-model-the-long-term-development-statement)
for what the two publications contain. Joining either publication to the `time_series_id`s
Flexpectation forecasts is unscoped work.

**[Fractal Flow](https://smarter.energynetworks.org/projects/npg_sif_026/), a Northern Powergrid
project building a shared cross-operator digital twin, is not a route to that model.** The twin is
wider in scope than Tier 2 needs, and arrives later. The project's Strategic Innovation Fund Beta
registration describes a shared, near-real-time digital twin across the transmission-distribution
boundary. Every GB distribution system operator (NGED included) and the National Energy System
Operator (NESO) are Beta partners. The Beta phase runs from September 2026 to September 2028. That
phase trials the twin on historic data in three control rooms, none of them NGED's. The
registration puts phased rollout at 2030, extending to all distribution system operators by 2035.

**Fractal Flow also sits on the far side of the forecasting boundary from Flexpectation.** The
registration describes the twin combining distribution network operator ("DNO") and NESO forecasts
"into a unified view up to 24-hours ahead of the settlement period". Separately, the registration
describes the twin predicting "possible power-flow changes through the network 24-hours ahead of
the settlement period".

### Curtailment price basis

**The headline prices the energy actually lost, in pounds per MWh, and sweeps three prices for what
NGED might pay because no published figure fixes that price.** A fourth, lower price of £100 is an
illustrative balancing-style constraint price and not a payment by NGED (see the prices below). The
headline is labelled "energy proxy" wherever it appears. It is `price × (Σ c_g + r × residual
breach)`, with the breach priced at $r$ times the price. Two reasons favour the energy proxy:
$\tau = 1 - 1/r$ minimises it, and the payment rule's own figure depends on how NGED's active
network management expresses a set-point, which is unknown.

**The payment rule is DCUSA Schedule 2D (version 15.3), and the DCUSA-form figure is reported beside
the headline.** Clause 3.2 measures Full Export Curtailment Hours as the sum of duration times
Curtailment Instruction Value, divided by the Curtailable Export Capacity. The Curtailment
Instruction Value is the instructed reduction of the generator's Maximum Export Capacity, not the
energy actually lost. Clause 3.4 pays (hours minus the Export Curtailment Limit) times capacity times
the Exceeded Curtailment Price. The DCUSA form takes the instruction value to be $\text{capacity}_g -
\text{cap}_g$ in MW on every half-hour where the generator's
share is above zero. Under that form an instruction to a generator running well below capacity is
charged for the whole headroom, so the two forms can rank forecasts differently. The DCUSA-form
figure adds the same residual-breach term as the energy proxy, $r \times$ breach $\times$ price. The
Curtailment Limit enters only the DCUSA-form figure, and the metric applies it to the scored period
as given, so a caller scoring part of a year pro-rates an annual limit to that part. Megavolt-amperes
are taken as MW at unity power factor, and all capacity as curtailable.

**The four prices are £100, £380, £2,000 (the headline), and £11,100 per MWh.** The £100 price is an
illustrative balancing-style constraint price. It is a whole-system cost: the cost to all electricity
users of re-dispatching generation up the merit order to relieve the constraint, not a cost borne
specifically by NGED as network operator or by the curtailed generator. The other three prices are
the ends and the geometric middle of the £380 to £11,100 per MVAh range reconstructed from April
2025 contract data, which is not a published figure. Results are reported as total pounds over the
scored period and as pounds per MW of curtailable generator capacity per year.

**Ranking uses the energy proxy, which does not depend on the Curtailment Limit.** At any limit
$\tau = 1 - 1/r$ is the cost-minimising quantile for the energy proxy, so the score is proper
(exactly so when the generator forecasts are perfect; otherwise the generator forecasts are an
operating rule that every forecast shares). The price scales every forecast alike, so the ranking
does not depend on the price, though the pound figure does. The DCUSA-form figure is reported at a
Curtailment Limit of 0 for ranking. A Curtailment Limit of 100 or 500 hours a year is shown, if at
all, as a restatement of the DCUSA-form figure for the same plans, never ranked. The avoidable cost
of a forecast is its cost minus the cost of a perfect forecast on the same rows, which uses actual
node flow and actual generator output. The metric does not subtract the perfect forecast; the
caller scores it as one more forecast and subtracts.

**A published annual curtailment-saving figure may use a capacity-based method, in which case the
published figure is not directly comparable with this metric.** This metric prices MWh curtailed,
whereas a capacity-based method starts from MW of capacity and scales it for export volume. A
comparison between the two figures needs either a separate replication of the capacity-based method
or a label saying the comparison is order-of-magnitude only.

### Which direction is the constraint on?

There is no single sign rule — see [sign convention](forecast-building-blocks.md#sign-convention).
The trial area contains both conventions, plus battery sites that both charge and discharge.
Metric 1's constraint-side direction is therefore resolved **per `time_series_type`**, reusing the mapping
the [tail and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
already need, with the ambiguous types to be confirmed with NGED.

A series constrained in both directions gets **a limit in each**, so the threshold is one scalar per
`(time_series_id, direction)` rather than per series alone. Each metric is computed only where its
direction is constrained: a demand primary with no connected generation gets no curtailment cost,
and a solar meter gets no flexibility procurement cost. Applying one global rule instead would
silently score £0 for every generator meter in the trial area. Metric 2 does not use this mapping:
each BSP and GSP carries an explicit export sign.

## What each number is compared against

Every model's cost is reported beside two reference points, computed on the same series and
half-hours:

- **Manual heuristic** — the 13-analogue ensemble, summarised at one percentile if a single number
  is needed ([the manual heuristic forecast](../background/manual-heuristic-forecast.md)). That
  percentile is an upper percentile chosen to match the company's risk appetite; our scoring uses
  the 95th percentile. [The residual-risk level the operating point is set to is still an open
  question](#questions-for-nged). The manual heuristic is scored at that **actual operating point,
  not calibrated to the common risk target**, because the point is to measure the manual heuristic
  as it is operated. Its realised unmet fraction is therefore an output — the number saying what
  risk level the manual heuristic works to — and the saving against it mixes a change in spend with
  a change in risk. Both are reported; neither means much alone.
- **Perfect forecast** — the least that can be spent while leaving no more than the target fraction
  $u$ unmet. Truth is not a distribution, so there is no quantile to calibrate: the floor is the
  cost of procuring exactly $(1-u)N_{i,t}$ in every half-hour, put straight through the same price
  formula. Setting it at zero unmet instead would put it *above* a model calibrated to 5%, and
  models would routinely score over 100% of the available saving.

The headline is *"£X less than the manual heuristic, which is Y% of the £Z a perfect forecast would
save"*, always alongside the realised unmet fractions. Note that a model whose realised unmet
fraction overshoots the target can still exceed 100%; that is a signal to read the risk column, not
a bug.

## Choosing the limit

Real network limits move with ambient temperature, with how long an overload lasts, with season and
with switching state, so no single number is correct — the fuller version of this caveat is in [the
threshold-choice
discussion](../techniques/evaluation-metrics.md#choosing-the-thresholds-static-per-series-quantile-derived).

**NGED's own limits are more complicated than a percentile of one series' history.** In NGED's
description, about half of generators have their limit at the next substation up (the bulk supply
point or grid supply point). In other cases the limit is on a circuit and only binds after the next
fault. It is set by power-flow and N-1 contingency analysis: simulate one fault, run a power flow,
simulate the next credible fault, and repeat over all credible faults.

**Metric 1 uses a synthetic limit: the 99th percentile of each series' own full observation
history,** in the constrained direction, labelled `historical_p99` to keep it distinct from the
forecast-quantile label `p99` — one is a fixed power level derived from history, the other a level
of the forecast distribution. This mirrors a percentile-of-history convention already used for
capacity setting, treating winter as close to the limit throughout, and it is the same single rung
the [tail and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
use, so the leaderboard carries one threshold concept for those metrics. The percentile sets the
absolute size of every £ figure of Metric 1 — a lower rung would multiply them — which is another
reason to read these numbers as a ranking instrument rather than a total. `historical_p99` is a
closer proxy to a real network rating than an arbitrary percentile would be, because comparable
network ratings are typically derived in a similar way.

**Metric 2 does not use `historical_p99`.** It uses the firm limit of each BSP and GSP, and a scaled
limit of 0.9 times the node's maximum observed export before the scored period, both described under
[Tier 1](#tier-1-substation-limits-no-power-flow). A percentile of a node's own flow is exceeded on
10% of half-hours by construction, so it cannot stand in for a limit that a node almost never
reaches. Metric 2 holds every forecast to the same $\tau = 1 - 1/r$, not to equal risk, so the
leaderboard's `historical_p99` rung is not shared with it.

**A rating never breached during the scored period is not useless.** Procurement and curtailment
volume are driven by the *forecast* crossing the limit, so forecasts still rank, but Metric 1's
unmet fraction goes undefined. Ratings are not available for every series and sit at different points
of each series' distribution, so they cannot carry the cross-series leaderboard.

## Case studies

- **Flexibility procurement (Metric 1) — Tavistock Primary, `CMZ_T9A_SWE_0050`, South West.**
  Buildable now. Tavistock has winter demand that reaches the level at which the zone procures
  flexibility, a seasonal transformer rating (13 MVA in winter, 10 MVA in the intermediate-cool,
  intermediate-warm, and summer periods — treated as MW, assuming negligible reactive power at this
  transformer), and real procurement history, all from public sources. "Winter" here is whatever
  calendar period NGED's own rating table uses, not a fixed date range; the exact month boundaries
  need confirming against the dataset rather than assumed. Procurement history for Tavistock
  specifically is in National Grid's [flexibility trades data and
  results](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/72b618d2-34c2-4347-8786-111d1cc93ce2)
  dataset; the same dataset's [long-term
  contracts](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/0e0b3921-c4d0-494c-a5ca-f529ad328ee3)
  resource (excluding any CMZ with "LV" in its name) and [short-term
  contracts](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/b04ce2c2-8798-486a-8591-48bfdd05d979)
  resource, across all zones, feed the volume-weighted average prices in [what the volumes
  cost](#what-the-volumes-cost).

- **The curtailment *case study* is blocked, not the curtailment metric.** Tier 1 needs no new data
  for the bulk supply points whose export sign our own data establishes. The grid supply points'
  signs are unconfirmed. Tier 1 can therefore rank models on those bulk supply points as soon as it
  is implemented, and does not wait on Tier 2. What is blocked is validating Tier 1 against a real
  curtailment event: mapping such an event to a specific series or hierarchy node needs Tier 2
  power-flow modelling — see [Tier 2](#tier-2-full-power-flow-modelling). The trial area has two
  nominal flexibility zones with an export-side constraint, but neither has enough dispatch history
  to stand in as a case study.

## What these numbers do not capture

- **The limit is invented, and fitted on the scored window.** A percentile of history is not a
  network rating, and it is computed over the full history including the months being scored. It is
  model-independent, so it cannot favour one entrant, but it is not an out-of-sample quantity.
- **The history is already post-intervention.** At a genuinely constrained site the metered power
  reflects flexibility that *was* dispatched and generation that *was* curtailed. So $N$ understates
  true need, and the percentile limit derived from that same history is itself shaped by the
  interventions we are pricing.
- **Ten trial-area sites cannot see direction at all.** They are metered in MVA, which reports the
  magnitude of flow, so reverse power flow appears as a *rise* rather than a sign change (see [data
  quality](../background/network.md#apparent-power-mva-metering)). At those sites an export event
  would be billed as demand-side procurement, and multiplying MVA by half an hour gives MVAh, which
  is not the megawatt-hour a flexibility price is quoted against.
- **Unmet energy is pooled across series and half-hours.** A model can hit the 5% target by covering
  the largest site well and abandoning many small ones, and 5% concentrated in one deep breach is
  far worse operationally than the same 5% spread thinly. Harm grows faster than depth; equalising
  energy does not equalise harm. The per-series distribution of unmet energy is reported for this
  reason.
- **The prices are single averages.** Two numbers stand in for a tendered market with zone-by-zone
  clearing prices and finite liquidity.
- **Procurement is not per-half-hour.** NGED tender flexibility ahead, in blocks and windows. Our
  arithmetic assumes perfectly granular buying, which flatters every model equally but overstates
  the achievable saving.
- **Ensemble size limits how finely $\tau$ can be tuned.** The manual heuristic has 13 analogues, so
  its quantiles come in coarse steps; a 51-member ensemble is far finer. Models of different
  ensemble size cannot be landed on exactly the same risk.
- **Costs are per fold, and folds are seasonal.** The limit concentrates exceedances at winter peak,
  so annualising a fold that does not span a whole year is meaningless, and a fold with no
  exceedance leaves the unmet fraction undefined.
- **Nothing is validated against real spend**, except the Tavistock flex-procurement [case
  study](#case-studies). The curtailment metric has no equivalent validation yet, and cannot until
  Tier 2 exists.
- **Asset failure and outage costs are excluded.** Outage quantification is valuable but harder, and
  is not in this design.
- **Some procurement does not depend on the forecast.** Procurement that serves wider aims, such as
  developing the flexibility market and supporting the capital programme, does not depend on the
  forecast. A better forecast should not be credited with removing that procurement.

## Questions for NGED

**Resolved:**

- **Flexibility prices, split by contract type** — rather than asking NGED for a single availability
  and utilisation rate, we compute volume-weighted average prices separately for long-term and
  short-term contracts from National Grid's published flexibility-trades data; see [what the volumes
  cost](#what-the-volumes-cost).
- **Curtailment price basis and who bears it** — Metric 2 sweeps four prices, £100 (a whole-system
  cost, not one borne specifically by NGED or by the curtailed generator), £380, £2,000, and £11,100
  per MWh; see [curtailment price basis](#curtailment-price-basis).
- **Case-study data availability** — Tavistock Primary (`CMZ_T9A_SWE_0050`) has real procurement
  history and a seasonal transformer rating, and anchors the flexibility-procurement metric against
  a real site; see [case studies](#case-studies).
- **Whether curtailment can be validated against a real event today** — no, not below [Tier
  2](#tier-2-full-power-flow-modelling).

**Still open:**

1. **Is the 95th percentile of the 13 analogues the operating point** the manual heuristic is
   operated at? What residual-risk level is the operating point set to?

## Cost-benefit analysis in the final work package

**NGED will run their own cost-benefit analysis in the project's final work package, and the choice
of method is theirs.** The final work package is tracked as
[WP7](https://github.com/openclimatefix/nged-substation-forecast/issues/684), due February 2028.

**One method worth recommending to NGED is a relative-economic-value curve in the shape of
[Richardson (2000)](https://doi.org/10.1002/qj.49712656313), computed per substation, across the
range of ratios between the cost of acting on a forecast and the loss avoided by acting.** [The
shared idea](#the-shared-idea-same-risk-then-compare-the-spend) above explains why this page took a
different route: £Y, the loss avoided by a breach, has no single figure the leaderboard metrics can
use. NGED's own final work package is better placed to supply that figure, because pricing a breach
is a business judgement rather than a property of a forecast.

## Implementation details (deleted when this ships)

- Two functions in `packages/ml_core/src/ml_core/metrics.py`, sharing a private helper taking the
  limit, the prices and the direction. They consume the same ensemble-member rows as the existing
  quantile metrics.
- **This needs a `Metrics` contract change, still to be reviewed and agreed when we implement**:
  `METRIC_NAMES` gains `flex_procurement_cost_gbp`, `curtailment_cost_gbp` and `unmet_fraction`, and
  `METRIC_PARAMS` gains a `historical_p99` label per constrained direction. Bare `"p99"` already
  exists in `QUANTILE_METRIC_PARAMS` meaning a *forecast* quantile, so a history-derived power level
  has to stay distinct from a level of the forecast distribution.
- **Two things the design needs to store have nowhere to live in `Metrics` today**: the number of
  half-hours a cost row covers (without it, totals cannot be compared across folds of unequal
  length) and the calibrated $\tau$ (which says how conservative a model had to be to reach the
  common risk target, and is interpretable on its own). Both need resolving with the contract
  change.
- Costs are stored **per `time_series_id`**, summed over time only, so they fit the existing primary
  key. Only $\tau$ and the unmet target are pooled across series. **Curtailment Tier 1 nets
  across series before pricing**, so its cost belongs to a hierarchy node, not a single
  `time_series_id` — this needs its own key, still to be resolved with the contract change.
- **Flexibility-price computation is a data pipeline task**, not hardcoded: pull National Grid's
  published flexibility-trades resources (see [case studies](#case-studies)), split by contract
  type, and compute a volume-weighted average price per type. This price-computation step has no
  home in the codebase yet.
- **`_log_metrics_to_mlflow` aggregates with `mean()`**, which is right for every metric that exists
  today and wrong for these: the portfolio headline is a *sum* over series, and pooled
  `unmet_fraction` is $\sum N$-weighted, not an unweighted mean that a tiny site can dominate. The
  MLflow aggregation needs a per-metric rule before these land.
- Prices, the risk target and the limit percentile are configuration, not constants in code, so
  NGED's answers can be applied without a retrain.
- Scored on a single lead-time slice (`day_ahead` until we know when the procurement decision is
  actually made) rather than all horizons pooled.
