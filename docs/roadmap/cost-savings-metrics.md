# Estimating the money a better forecast saves

> **Status: 🚧 Planned (v0.3).** Epic:
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6); issue:
> [#606](https://github.com/openclimatefix/nged-substation-forecast/issues/606). This page is the
> plan for two leaderboard metrics that express forecast skill in pounds. It is written to be read
> by anyone numerate. Everything attributed to NGED below comes from a meeting in July 2026 and is
> flagged for them to confirm. See the [roadmap index](index.md) for status conventions.

## Read this first: these pounds rank models, they do not cost anything

These metrics put a **£** figure against every model we train. That figure rests on an invented
network limit and a couple of average prices, so it is a **rough proxy**: not a cost analysis, not a
business case, and not quotable as either. Its one job is to rank forecasts on the axis NGED care
about, turning "this model has a lower threshold-weighted continuous ranked probability score" into
"this model would have spent less to keep the network within limits". We use pounds rather than a
unitless score because the parameters genuinely are prices, and because pounds are what the
decisions these forecasts feed are actually made in. Forecast developers do not usually tune on
money: [Gürses-Tran and Monti (2022)](https://doi.org/10.3390/forecast4020028) observe that
forecast developers "predominantly assess residuals and error statistics when tuning the targeted
model's quality". As a result, "eventual cost or rewards of the underlying business application are
typically not considered in the model development phase".

## Two savings, measured separately

A better forecast creates value through two different mechanisms, with different prices and
different beneficiaries. We compute them as **two metrics, reported as two numbers**, and never add
them up:

1. **Flexibility procurement.** NGED pay flexible customers to reduce demand when a site risks
   running beyond its limit. Procurement today is deliberately conservative, so a sharper forecast
   buys less flexibility for the same security. Flexibility procurement is money NGED spend.
2. **Curtailment of generation.** Generators are curtailed to keep exports within network limits.
   Curtailment avoided is generation sold, priced as a whole-system cost rather than a saving to
   NGED or the connected generator specifically — see [curtailment price
   basis](#curtailment-price-basis).

A third saving — the engineer-hours freed by replacing a manual review of time-series plots with an
automated forecast — is real, but it is **not a leaderboard metric**: it is identical for every
model we train, so it cannot rank them. This third saving belongs in the project's final report,
priced in engineer-hours.

## The shared idea: same risk, then compare the spend

The textbook way to price a forecast charges it for what goes wrong: £X per action taken, £Y per
limit breach nobody saw coming. We cannot follow that route, because £Y — the cost of a breach — is
not a figure NGED hold in a form we can use. And the literature that does price a forecast this way
has never done so on a real distribution network at a money-denominated cost — the
[energy-forecasting
review](../background/energy-forecasting-review.md#evaluating-the-performance-of-power-forecasts)
reports [Richardson (2000)](https://doi.org/10.1002/qj.49712656313)'s cost-loss framing, [Bernecker
et al. (2025)](https://doi.org/10.1016/j.ijepes.2025.110713)'s 97% synthetic-network saving, and
[Angus et al. (2027)](https://doi.org/10.1016/j.epsr.2026.113545)'s and [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335)'s capacity- and volume-denominated results.

So we invert the question. **Models are aimed at equal safety, and we compare what each one spends
to get there.**

The knob is the **procurement quantile** $\tau$ — how far up its own forecast distribution a model
looks when deciding to act. A timid model uses a high $\tau$, buys a lot, and is rarely caught out.
Calibration picks each model's $\tau$ so that its **unmet fraction** — the share of
genuinely-needed megawatt-hours it failed to cover — hits a common target (5% to begin with):

$$
\text{unmet fraction} = \frac{\sum_{i,t} \max(0,\; N_{i,t} - V_{i,t})}{\sum_{i,t} N_{i,t}}
$$

where $V_{i,t}$ is the volume the model would have bought (or curtailed) for time series $i$ in
half-hour $t$, and $N_{i,t}$ is the volume that turned out to be needed. Measuring unmet *energy*
rather than counting missed events matters: exceedances are rare by construction, and a count of
them is too noisy to rank models by.

**$\tau$ is calibrated on the training window of the leaderboard fold, never on the validation
window it is scored on** — otherwise a model sees its own future and every pound of the "saving" is
lookahead. This has a methodological drawback: the training window is data the model was fitted to,
so its residuals are smaller than they will be out of sample. As a result, $\tau$ comes out too
low, and every model under-procures on the scored window. The model that overfits hardest gains
most from this.

**Equal risk is therefore a target, not a guarantee, and this is the design's main weakness.** What
a model *realises* on the scored window is whatever its tail calibration delivers there. A model
that overshoots the target spends less and can top the leaderboard while being materially less
safe. The **realised out-of-sample unmet fraction is reported beside every cost, and a cost read
without it is meaningless.** Two models are only comparable on cost when their realised unmet
fractions are close.

## What the volumes cost

Flexibility is bought in two parts, and the distinction is the whole point of these metrics.
**Availability** is paid on every megawatt-hour held ready, whether or not it is called;
**utilisation** is paid only on what is actually dispatched. Over-procurement therefore costs the
*availability* price on the excess, not the far larger utilisation price:

$$
C = p_{\text{avail}} \sum_{i,t} V_{i,t} \;+\; p_{\text{util}} \sum_{i,t} \min(V_{i,t},\, N_{i,t})
$$

The second term is nearly identical for every model — it is set by what the network actually
needed. So the ranking is carried by the first. Charging one blended price against all procured
volume would overstate the cost of over-procurement several times over.

This formula only holds for **short-term contracts**, tendered day-ahead against a forecast. NGED
also buy **long-term contracts**, whose availability is tendered roughly a year ahead of delivery,
independent of any forecast we produce now. A better forecast cannot reduce that volume, because
it is already committed by the time our forecast exists. Only the **utilisation** decision on
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
contract, $V_{i,t}$ and $N_{i,t}$ are unchanged but only the utilisation term
$p_{\text{util}} \min(V_{i,t}, N_{i,t})$ is charged.

**Worked example.** A substation whose limit sits at 30 MW, on one winter evening half-hour, with
short-term availability at £75/MWh and utilisation at £750/MWh (both placeholders, pending real
volume-weighted prices). Manual review forecasts 33 MW, so it procures $(33 - 30) \times 0.5 = 1.5$
MWh. Demand turns out to be 30.6 MW, so 0.3 MWh was needed. It pays £112.50 availability and £225
utilisation. A model forecasting 31.0 MW procures 0.5 MWh, pays £37.50 and the same £225 — saving
£75. If the same half-hour were instead covered by a long-term contract, both models pay only the
utilisation term (£225 each) and the forecast saves nothing on that half-hour. The metric sums this
over every half-hour and every series, split by which contract type covered each half-hour.

## Metric 2 — curtailment cost

Curtailment is not decided per meter: a generation constraint typically binds above the primary,
driven by the aggregated flow across several substations rather than by any single meter in
isolation. Scoring each `time_series_id` independently, as Metric 1 does for flexibility, would
misrepresent how curtailment actually gets triggered. We instead define **three tiers of increasing
realism**, each documented separately with its own scope and caveats, all keeping the "these numbers
rank models, they do not cost anything" disclaimer above — none is a validated £-saved figure
without a real case study behind it.

### Tier 1 — per-primary netting

At one primary, net the local generation forecasts against the local demand forecasts before
comparing the result to the primary's export limit, instead of scoring each meter independently.
Otherwise identical to the Metric 1 arithmetic, with the export-side limit and the curtailment price
$p_{\text{curt}}$ (£/MWh of network access) charged against total volume curtailed, since curtailed
generation is lost whether or not the constraint was real. There is no hierarchy in this tier: each
primary is scored in isolation.

**Worked example.** A generation-dominated feeder with an export limit of 8.5 MW. The netted
forecast at its calibrated quantile says 9.6 MW, so 0.55 MWh is curtailed; actual net export is 8.7
MW, so 0.1 MWh needed curtailing. At £100/MWh that is £55 against £10 of real constraint — £45 of
generation curtailed for nothing.

**Caveat.** This tier is only valid where a primary's export is electrically isolated from its
neighbours. Where primaries share a constraint — operate "in parallel" — curtailment must be
assessed jointly across the group, not per primary. Which primaries in the trial area operate in
parallel is an **open per-site question**, not a question this tier can assume away. Treating
every primary as isolated will overstate how much curtailment Tier 1 avoids at any site that
actually shares a constraint.

### Tier 2 — substation hierarchy, no power-flow

Sum active-power forecasts (generation minus demand) up the known substation hierarchy, one node at
a time, and compare each node's summed forecast against a limit for that node. This is Tier 1 with
the netting extended from one primary to every node above it.

The limit at each node is a **synthetic toy limit**: the same `historical_p99`-style
percentile-of-history device already used per series (see [choosing the
limit](#choosing-the-limit)), applied at the node level instead — the percentile of that node's own
historical summed power. Where NGED supply a real rating for a node, we use that instead.

**Caveats.**

- **No reactive power, voltage drop, or N-1 contingency modelling.** A hierarchy sum of active power
  is not a load-flow study; Tier 3 is where that modelling belongs.
- **Coverage gap.** Only some of a node's children have a forecast in the trial area; the rest need
  a metered-actuals or historical-baseline stand-in, which adds a second error source on top of
  forecast error at every node above the trial-area leaves. This gap shrinks as trial coverage scales
  towards the ~2,500-series v2 rollout — it is a limitation of the trial's current coverage, not a
  fixed limitation of the tier.

### Tier 3 — full power-flow modelling

Out of scope for Flexpectation, including v2. Tier 3 is documented as the eventual correct approach
— modelling reactive power, voltage drop and N-1 contingencies explicitly — but it is gated on
power-flow integration work that sits outside this project.

Tier 3 is also where a **real curtailment case study** becomes possible. Tiers 1 and 2 can each be computed and can rank models without Tier 3. What they cannot yet do is be validated against a real curtailment event, because no existing site maps a curtailment case to a specific series or a single hierarchy node — see [case studies](#case-studies) below.

### Curtailment price basis

£100/MWh is a **whole-system cost**: the cost to all electricity users from re-dispatching
generation up the merit order to relieve the constraint, not a cost borne specifically by NGED as
network operator or by the curtailed generator. This is the basis Metric 2 is priced on throughout.
It may change: the network operator could in future need to pay a flexibility counterfactual cost
to curtail generators directly, at which point curtailment would carry a network-operator-borne
price alongside the whole-system one.

**A published annual curtailment-saving total is not automatically comparable to this metric.** A
headline £-per-year figure NGED report elsewhere may be computed on a different basis — capacity or
MW-based, with a scaling factor applied for export volume — rather than the MWh-curtailed
calculation this metric uses. Comparing the two numbers directly without either replicating that
other method separately, or explicitly labelling the comparison as order-of-magnitude only, would
overstate how precisely they agree.

### Which direction is the constraint on?

There is no single sign rule. This repo carries two conventions — at a substation, positive power
flows towards end-users; at a customer's meter, positive means the customer is *exporting* to
NGED's grid (see [sign convention](forecast-building-blocks.md#sign-convention)) — and the trial
area contains both,
plus battery sites that both charge and discharge. Constraint-side direction is therefore resolved
**per `time_series_type`**, reusing the mapping the [tail and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
already need, with the ambiguous types confirmed by NGED.

A series constrained in both directions gets **a limit in each**, so the threshold is one scalar
per `(time_series_id, direction)` rather than per series alone. Each metric is computed only where
its direction is constrained: a demand primary with no connected generation gets no curtailment
cost, and a solar meter gets no flexibility procurement cost. Applying one global rule instead
would silently score £0 for every generator meter in the trial area.

## What each number is compared against

Every model's cost is reported beside two reference points, computed on the same series and
half-hours:

- **Manual review** — the incumbent method: the 13-analogue ensemble, summarised at the 95th
  percentile if a single number is needed ([the incumbent
  forecast](../background/nged-incumbent-forecast.md); we have not confirmed that the 95th
  percentile is what they use, and [this is still an open question](#questions-for-nged)). It is
  scored at that
  **actual operating point, not calibrated to the common risk target**, because the point is to
  measure what NGED do today. Its realised unmet fraction is therefore an output — the number
  saying what risk level they currently work to — and the saving against it mixes a change in spend
  with a change in risk. Both are reported; neither means much alone.
- **Perfect forecast** — the least that can be spent while leaving no more than the target fraction
  $u$ unmet. Truth is not a distribution, so there is no quantile to calibrate: the floor is the
  cost of procuring exactly $(1-u)N_{i,t}$ in every half-hour, put straight through the same price
  formula. Setting it at zero unmet instead would put it *above* a model calibrated to 5%, and
  models would routinely score over 100% of the available saving.

The headline is *"£X less than manual review, which is Y% of the £Z a perfect forecast would
save"*, always alongside the realised unmet fractions. Note that a model whose realised unmet
fraction overshoots the target can still exceed 100%; that is a signal to read the risk column, not
a bug.

## Choosing the limit

Real network limits move with ambient temperature, with how long an overload lasts, with season and
with switching state, so no single number is correct — the fuller version of this caveat is in
[the threshold-choice
discussion](../techniques/evaluation-metrics.md#choosing-the-thresholds-static-per-series-quantile-derived).
We use a **synthetic limit**: the **99th percentile of each series' own full observation history**,
in the constrained direction, labelled `historical_p99` to keep it distinct from the
forecast-quantile label `p99` — one is a fixed power level derived from history, the other a level
of the forecast distribution.

This mirrors a percentile-of-history convention already used for capacity setting, treating winter
as close to the limit throughout, and it is the same single rung the [tail
and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
use, so the leaderboard carries one threshold concept rather than several. The percentile sets the
absolute size of every £ figure on this page — a lower rung would multiply them — which is another
reason to read these numbers as a ranking instrument rather than a total.

`historical_p99` is a closer proxy to a real network rating than an arbitrary percentile would be:
this is methodologically consistent with how comparable network ratings are typically derived, so
the synthetic limit is close to the practice it stands in for, not just a convenient round number.

Where NGED supply a real firm or flex rating we will compute the same metrics against it, as a
case study. A rating never breached during the scored window is not useless here — procurement
volume is driven by the *forecast* crossing the limit, so models still rank — but the unmet
fraction goes undefined, and ratings are not available for every series and sit at different points
of each series' distribution, so they cannot carry the cross-series leaderboard.

## Case studies

- **Flexibility procurement (Metric 1) — Tavistock Primary, `CMZ_T9A_SWE_0050`, South West.**
  Buildable now. Tavistock has genuine winter exceedances, a seasonal transformer rating (13 MVA in
  winter, 10 MVA in the intermediate-cool, intermediate-warm and summer periods — treated as MW,
  assuming negligible reactive power at this transformer), and real procurement history, all from
  public sources. "Winter" here is whatever calendar period NGED's own rating table uses, not a
  fixed date range; the exact month boundaries need confirming against the dataset rather than
  assumed. Procurement history for Tavistock specifically is in National Grid's [flexibility trades
  data and results](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/72b618d2-34c2-4347-8786-111d1cc93ce2)
  dataset; the same dataset's [long-term contracts](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/0e0b3921-c4d0-494c-a5ca-f529ad328ee3)
  resource (excluding any CMZ with "LV" in its name) and [short-term contracts](https://connecteddata.nationalgrid.co.uk/dataset/flexibility-trades-data-and-results/resource/b04ce2c2-8798-486a-8591-48bfdd05d979)
  resource, across all zones, feed the volume-weighted average prices in [what the volumes cost](#what-the-volumes-cost).

- **The curtailment *case study* is blocked, not the curtailment metric.** Tiers 1 and 2 need nothing that does not already exist, so each can rank models as soon as it is implemented. Tier 2 does not wait on Tier 3. What is blocked is validating either tier against a real curtailment event: mapping such an event to a specific series or hierarchy node needs Tier 3 power-flow modelling — see [Tier 3](#tier-3-full-power-flow-modelling). The trial area has two nominal flexibility zones with an export-side constraint, but neither has enough dispatch history to stand in as a case study.

## What these numbers do not capture

- **The limit is invented, and fitted on the scored window.** A percentile of history is not a
  network rating, and it is computed over the full history including the months being scored. It is
  model-independent, so it cannot favour one entrant, but it is not an out-of-sample quantity.
- **The history is already post-intervention.** At a genuinely constrained site the metered power
  reflects flexibility that *was* dispatched and generation that *was* curtailed. So $N$ understates
  true need, and the percentile limit derived from that same history is itself shaped by the
  interventions we are pricing.
- **Ten trial-area sites cannot see direction at all.** They are metered in MVA, which reports the
  magnitude of flow, so reverse power flow appears as a *rise* rather than a sign change (see
  [data quality](../background/data-quality.md)). At those sites an export event would be billed as
  demand-side procurement, and multiplying MVA by half an hour gives MVAh, which is not the
  megawatt-hour a flexibility price is quoted against.
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
- **Ensemble size limits how finely $\tau$ can be tuned.** Manual review has 13 analogues, so its
  quantiles come in coarse steps; a 51-member ensemble is far finer. Models of different ensemble
  size cannot be landed on exactly the same risk.
- **Costs are per fold, and folds are seasonal.** The limit concentrates exceedances at winter peak,
  so annualising a fold that does not span a whole year is meaningless, and a fold with no
  exceedance leaves the unmet fraction undefined.
- **Nothing is validated against real spend**, except the Tavistock flex-procurement [case
  study](#case-studies). The curtailment metric has no equivalent validation yet, and cannot until
  Tier 3 exists.
- **Asset failure and outage costs are excluded.** Outage quantification is valuable but harder,
  and is not in this design.
- **Over-procurement has a deliberate component.** Some over-procurement is understood to be
  deliberate policy — supporting flexibility-market development and the capital programme — rather
  than forecast error, and a better forecast should not be credited with removing it.

## Questions for NGED

**Resolved**, by the July and August 2026 meetings and the public procurement data found since:

- **Flexibility prices, split by contract type** — rather than asking NGED for a single availability
  and utilisation rate, we compute volume-weighted average prices separately for long-term and
  short-term contracts from National Grid's published flexibility-trades data; see [what the volumes
  cost](#what-the-volumes-cost).
- **Curtailment price basis and who bears it** — £100/MWh is a whole-system cost, not one borne
  specifically by NGED or by the curtailed generator; see [curtailment price
  basis](#curtailment-price-basis).
- **Case-study data availability** — Tavistock Primary (`CMZ_T9A_SWE_0050`) has real procurement
  history and a seasonal transformer rating, and anchors the flexibility-procurement metric against
  a real site; see [case studies](#case-studies).
- **Whether curtailment can be validated against a real event today** — no, not below [Tier
  3](#tier-3-full-power-flow-modelling).

**Still open:**

1. **Is the 95th percentile of the 13 analogues the operating point** manual review actually works
   from, and what reliability does NGED target — how much genuinely-needed flexibility may go
   unbought? This was not settled by the prior round of answers and needs a direct follow-up
   question.

## Cost-benefit analysis in the final work package

**NGED will run their own cost-benefit analysis in the project's final work package, and the
choice of method is theirs.** The final work package is tracked as
[WP7](https://github.com/openclimatefix/nged-substation-forecast/issues/684), due February 2028.

**One method worth recommending to NGED is a relative-economic-value curve in the shape of
[Richardson (2000)](https://doi.org/10.1002/qj.49712656313), computed per substation, across the
range of ratios between the cost of acting on a forecast and the loss avoided by acting.** [The
shared idea](#the-shared-idea-same-risk-then-compare-the-spend) above explains why this page took
a different route: £Y, the loss avoided by a breach, is not a figure NGED hold in a form the
leaderboard metrics can use. NGED's own final work package is better placed to supply that figure,
because pricing a breach is a judgement about NGED's business, not a property of a forecast.

**NGED's cost-benefit analysis is not expected to start until late 2027, close to WP7's February
2028 deadline.** This page records the recommendation now, well ahead of that date, so that the
recommendation is not forgotten in the meantime.

## Implementation details (deleted when this ships)

- Two functions in `packages/ml_core/src/ml_core/metrics.py`, sharing a private helper taking the
  limit, the prices and the direction. They consume the same ensemble-member rows as the existing
  quantile metrics.
- **This needs a `Metrics` contract change, still to be reviewed and agreed when we implement**:
  `METRIC_NAMES` gains `flex_procurement_cost_gbp`, `curtailment_cost_gbp` and `unmet_fraction`,
  and `METRIC_PARAMS` gains a `historical_p99` label per constrained direction. Bare `"p99"`
  already exists in `QUANTILE_METRIC_PARAMS` meaning a *forecast* quantile, so a history-derived
  power level has to stay distinct from a level of the forecast distribution.
- **Two things the design needs to store have nowhere to live in `Metrics` today**: the number of
  half-hours a cost row covers (without it, totals cannot be compared across folds of unequal
  length) and the calibrated $\tau$ (which says how conservative a model had to be to reach the
  common risk target, and is interpretable on its own). Both need resolving with the contract
  change.
- Costs are stored **per `time_series_id`**, summed over time only, so they fit the existing primary
  key. Only $\tau$ and the unmet target are pooled across series. **Curtailment Tiers 1 and 2 net
  across series before pricing**, so their cost belongs to a primary or hierarchy node, not a single
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
