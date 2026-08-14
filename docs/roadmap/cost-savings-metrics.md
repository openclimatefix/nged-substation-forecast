# Estimating the money a better forecast saves

> **Status: 🚧 Planned (v0.3).** Epic:
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6); issue:
> [#606](https://github.com/openclimatefix/nged-substation-forecast/issues/606). This page is the
> plan for two leaderboard metrics that express forecast skill in pounds. It is written to be read
> by anyone numerate. See the [roadmap index](index.md) for status conventions.

## Read this first: these pounds rank models, they do not cost anything

These metrics put a **£** figure against every model we train. That figure rests on an invented
network limit and a single average price per megawatt-hour, so it is a **rough proxy**: not a cost
analysis, not a business case, and not quotable as either. Its one job is to rank forecasts on the
axis NGED care about, turning "this model has a lower threshold-weighted continuous ranked
probability score" into "this model would have spent less to keep the network within limits". We
use pounds rather than a unitless score because the parameters genuinely are prices, and because a
pound figure is the only forecast-quality number most readers can act on.

## Two savings, measured separately

A better forecast creates value through two different mechanisms, with different prices and
different beneficiaries. We compute them as **two metrics, reported as two numbers**, and never add
them up:

1. **Flexibility procurement.** NGED pay flexible customers to reduce demand when a site risks
   running beyond its limit. They are risk-averse and knowingly over-procure. A sharper forecast
   buys less flexibility for the same security. This is money NGED spend.
2. **Curtailment of generation.** Generators are curtailed to keep exports within network limits.
   Curtailment avoided is generation sold. Who this saves money *for* — NGED, or the connected
   generator under a non-firm connection — is [question 3](#questions-for-nged) below, and it
   changes how the number should be read.

A third saving — the engineer-hours freed by replacing a manual review of time-series plots with an
automated forecast — is real, but it is **not a leaderboard metric**: it is identical for every
model we train, so it cannot rank them. It belongs in the project's final report, priced in
engineer-hours.

## The shared idea: same risk, then compare the spend

The textbook way to price a forecast charges it for what goes wrong: £X per action taken, £Y per
limit breach nobody saw coming. We cannot do that, because £Y — the cost of a breach — is not a
figure NGED hold in a form we can use.

So we invert the question. **Models are aimed at equal safety, and we compare what each one spends
to get there.** Each model may be as conservative as it likes, and we tune that conservatism until
it leaves the same small amount of risk unaddressed. Then the only thing left to compare is cost.
This matches NGED's account of the problem: they are not trying to avoid a breach they currently
suffer, they are trying to stop over-buying to avoid one.

The knob is the **procurement quantile** $\tau$ — how far up its own forecast distribution a model
looks when deciding to act. A timid model uses a high $\tau$, buys a lot, and is rarely caught out.
Calibration picks each model's $\tau$ so that its **unmet fraction** — the share of
genuinely-needed megawatt-hours it failed to cover — hits a common target (5% to begin with):

$$
\text{unmet fraction} = \frac{\sum_{i,t} \max(0,\; N_{i,t} - V_{i,t})}{\sum_{i,t} N_{i,t}}
$$

where $V_{i,t}$ is the volume the model would have bought (or curtailed) for time series $i$ in
half-hour $t$, and $N_{i,t}$ is the volume that turned out to be needed. Measuring unmet *energy*
rather than counting missed events matters: at a p99 limit the events are rare, and a count of them
is too noisy to rank models by.

**$\tau$ is calibrated on training folds only.** Tuning it on the fold being scored would let a
model see its own future, and every pound of the resulting "saving" would be lookahead.

**Equal risk is a target, not a guarantee, and this is the design's main weakness.** Because $\tau$
is fixed in advance, what a model *realises* on the scored fold is whatever its tail calibration
delivers there. An underdispersed model overshoots the target, spends less, and can top the
leaderboard while being materially less safe. The **realised out-of-sample unmet fraction is
therefore reported beside every cost, and a cost read without it is meaningless.** Two models are
only comparable on cost when their realised unmet fractions are close.

## Metric 1 — flexibility procurement cost

For time series $i$ and half-hour $t$, with **limit** $L_i$ and flexibility price $p_{\text{flex}}$
(£/MWh):

| Quantity | Definition |
|---|---|
| Volume procured | $V_{i,t} = \max(0,\; \hat q_{i,t}(\tau) - L_i) \times 0.5$ MWh |
| Volume needed | $N_{i,t} = \max(0,\; y_{i,t} - L_i) \times 0.5$ MWh |
| Cost | $C = p_{\text{flex}} \sum_{i,t} V_{i,t}$ |

$\hat q_{i,t}(\tau)$ is the model's $\tau$-quantile forecast, $y_{i,t}$ the observed power measured
in the **constraint-side direction** for that series (below), and $\times 0.5$ converts MW held for
a half-hour into MWh.

**Worked example.** A primary substation whose limit sits at 28 MW, on one winter evening
half-hour. Manual review forecasts 31 MW, so it procures $(31 - 28) \times 0.5 = 1.5$ MWh. Demand
turns out to be 28.6 MW, so only 0.3 MWh was needed. At £750/MWh that half-hour cost £1,125, of
which £225 was useful. A model forecasting 29.0 MW procures 0.5 MWh — £375, saving £750 in that
half-hour. The metric sums this over every half-hour and every series.

## Metric 2 — curtailment cost

Identical arithmetic on the export side, with the export limit $E_i$ and the curtailment price
$p_{\text{curt}}$ (£/MWh of network access). $V_{i,t}$ is the volume curtailed, $N_{i,t}$ the
volume that needed curtailing, and the cost is $p_{\text{curt}} \sum_{i,t} V_{i,t}$.

**Worked example.** A generation-dominated feeder with an export limit of 8.5 MW. The forecast at
its calibrated quantile says 9.6 MW, so 0.55 MWh is curtailed; actual export is 8.7 MW, so 0.1 MWh
needed curtailing. At £100/MWh that is £55 against £10 of real constraint — £45 of generation
curtailed for nothing.

The two ship as **two functions returning two numbers**, sharing a private helper, because their
prices, limits, direction and beneficiaries differ.

### Which direction is the constraint on?

There is no single sign rule. This repo carries two conventions — at a substation, positive power
flows towards end-users; at a customer's meter, positive means the customer is *generating* (see
[sign convention](forecast-building-blocks.md#sign-convention)) — and the trial area contains both,
plus battery sites that both charge and discharge. Constraint-side direction is therefore resolved
**per `time_series_type`**, reusing the mapping the [tail and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
already need, with the ambiguous types confirmed by NGED. Applying one global rule would silently
score £0 for every generator meter in the trial area.

## What each number is compared against

Every model's cost is reported beside two reference points, computed on the same series and
half-hours:

- **Manual review** — NGED's method today: the 13-analogue ensemble, read off a plot, taking the
  95th percentile if a single number is needed ([the incumbent
  forecast](../background/nged-incumbent-forecast.md); we have not confirmed with NGED that the
  95th percentile is what they use, and [question 5](#questions-for-nged) asks). It is scored at
  that **actual operating point, not calibrated to the common risk target**, because the point is
  to measure what NGED do today. Its realised unmet fraction is therefore an output — the number
  that says what risk level they currently work to — and the saving against it mixes a change in
  spend with a change in risk. Both are reported; neither is meaningful alone.
- **Perfect forecast** — truth used as the forecast, held to the **same unmet target** as every
  model. This is the floor: the least that can be spent at that risk level. Holding it to zero
  unmet instead would let a calibrated model spend less than "perfect" and score above 100% of the
  available saving.

The headline is *"£X less than manual review, which is Y% of the £Z a perfect forecast would
save"*, always alongside the realised unmet fractions. Both metrics are computed for every
experiment.

## Choosing the limit

Real network limits move with ambient temperature, with how long an overload lasts, with season and
with switching state, so no single number is correct. We use a **synthetic limit** — a percentile
of each series' own full observation history, for the same full-history stability reason the
[NMAE denominator](metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity) uses — and
report the metric at **both the 95th and 99th percentiles**, labelled `hist_p95` and `hist_p99`.

NGED offered real firm and flex capacities for the trial area, and we want them — for the case
studies below, and to check that the synthetic limits land somewhere sensible. They cannot replace
the synthetic limit for ranking, because a real rating that was never breached during the scoring
window produces zero exceedance events, and no model can be graded on events that never happened.

The 99th percentile is the better stand-in for a network close to its limit only at winter peak.
The 95th exists because 99th-percentile exceedances may be too rare to separate models, and that is
a question for data rather than assumption. If the two rank models identically we keep the 99th
alone.

**This ladder differs from the one the tail and exceedance metrics use** (the 90th and 98th
percentiles, `hist_p90` / `hist_p98`). NGED endorsed the 95th/99th for procurement decisions; the
older rungs were our own choice, made before that conversation. Two ladders on one leaderboard is
not a good end state and [#254](https://github.com/openclimatefix/nged-substation-forecast/issues/254)
should reconcile them.

## What these numbers do not capture

- **The limits are invented.** A percentile of history is not a network rating. Sites that are
  genuinely unconstrained get a limit anyway, and are scored as though flexibility were bought
  there.
- **The history is already post-intervention.** At a genuinely constrained site the metered power
  reflects flexibility that *was* dispatched and generation that *was* curtailed. So $N$ understates
  true need, and the percentile limit derived from that same history is itself shaped by the
  interventions we are pricing.
- **The risk target binds only in-sample**, as set out above; the realised unmet fraction is the
  guard against this and must be read with every cost.
- **Unmet energy is pooled across series and half-hours.** A model can hit the 5% target by
  covering the largest site well and abandoning many small ones, and 5% concentrated in one deep
  breach is far worse operationally than the same 5% spread thinly. Harm grows faster than depth;
  equalising energy does not equalise harm. The per-series distribution of unmet energy is reported
  for this reason.
- **The prices are single averages.** One £/MWh figure stands in for a tendered market with
  availability payments, utilisation payments, zone-by-zone clearing prices and finite liquidity.
- **Procurement is not per-half-hour.** NGED tender flexibility ahead, in blocks and windows. Our
  arithmetic assumes perfectly granular buying, which flatters every model equally but overstates
  the achievable saving.
- **Ensemble size limits how finely $\tau$ can be tuned.** Manual review has 13 analogues, so its
  quantiles come in coarse steps; a 51-member ensemble is far finer. Models of different ensemble
  size cannot be landed on exactly the same risk.
- **Costs are per fold, and folds are seasonal.** A 99th-percentile limit concentrates exceedances
  at winter peak, so annualising a fold that does not span a whole year is meaningless. A fold with
  no exceedance at all leaves the unmet fraction undefined.
- **Nothing is validated against real spend**, except at the trial-area sites that sit in an actual
  flexibility zone. There are a couple, and they are the case studies that tell us whether these
  numbers are the right order of magnitude.
- **Asset failure and outage costs are excluded.** NGED identified outage quantification as
  valuable but harder; it is not in this design.
- **Over-procurement has a deliberate component.** NGED over-buy partly to stimulate the
  flexibility market and to support their capital programme. That portion is policy, not forecast
  error, and a better forecast should not be credited with removing it.

## Questions for NGED

1. **Flexibility price** — is £500–1000/MWh right for dispatched flexibility, which published
   dataset should we take it from, and does it cover availability payments as well as utilisation?
2. **Curtailment price** — is £100/MWh of network access right, and is the £2M saved last year on
   the same basis, so we can check our totals against it?
3. **Who bears the curtailment cost** — NGED, or the generator under a non-firm connection? This
   decides whether metric 2 measures NGED's saving or the connected customer's.
4. **How much do you currently over-procure?** The entire saving is measured against this. Even
   procured-versus-needed volumes for a single zone would anchor it.
5. **Is the 95th percentile of the 13 analogues the operating point** you actually work from, and
   what reliability do you target — how much genuinely-needed flexibility may go unbought?
6. **Which trial-area sites have curtailable generation**, which sit in a real flexibility zone with
   procurement history we can use as a case study, and can we have the firm and flex capacities?

## Implementation details (deleted when this ships)

- Two functions in `packages/ml_core/src/ml_core/metrics.py`, sharing a private helper taking the
  limit, the price and the direction. They consume the same ensemble-member rows as the existing
  quantile metrics.
- **This needs a `Metrics` contract change, to be agreed before it is written**: `METRIC_NAMES`
  gains `flex_procurement_cost_gbp`, `curtailment_cost_gbp` and `unmet_fraction`, and
  `METRIC_PARAMS` gains `hist_p95` / `hist_p99`. The `hist_` prefix is not decoration — bare
  `"p95"` and `"p99"` already exist in `QUANTILE_METRIC_PARAMS` meaning *forecast* quantiles, and
  the tail-metric design chose the prefix precisely to keep a history-derived power level distinct
  from a level of the forecast distribution.
- Costs are stored **per `time_series_id`**, summed over time only, so they fit the existing primary
  key; the portfolio headline is their sum. Only $\tau$ and the unmet target are pooled across
  series.
- Costs are sums over time, unlike every other metric in the table, so a row must record the number
  of half-hours it covers or the totals cannot be compared across folds of unequal length.
- The calibrated $\tau$ is worth storing: it says how conservative a model had to be to reach the
  common risk target, which is interpretable on its own.
- Prices, the risk target and the limit percentiles are configuration, not constants in code, so
  NGED's answers can be applied without a retrain.
- Scored on a single lead-time slice (`day_ahead` until we know when the procurement decision is
  actually made) rather than all horizons pooled.
