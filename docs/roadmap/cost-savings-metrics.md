# Estimating the money a better forecast saves

> **Status: 🚧 Planned (v0.3).** Epic:
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6); issue:
> [#606](https://github.com/openclimatefix/nged-substation-forecast/issues/606). This page is the
> plan for two leaderboard metrics that express forecast skill in pounds. It is written to be
> read by anyone numerate, and is the page to send NGED for review. See the
> [roadmap index](index.md) for status conventions.

## Read this first: the pounds are a yardstick, not an audit

These metrics put a **£** figure against every model we train. That figure is a **rough proxy**,
built on a synthetic network limit and a single average price per megawatt-hour. It is not a cost
analysis, it is not a business case, and it must not be quoted as either. Its one job is to rank
forecasts on the axis NGED actually cares about, so that "this model has a lower threshold-weighted
CRPS" becomes "this model would have spent less to keep the network within limits".

We use pounds anyway, rather than a unitless score, for two reasons: the parameters genuinely are
prices in pounds, and a pound figure is the only forecast-quality number most readers can act on.
The risk we accept is that a skim-reader mistakes it for rigour — hence this warning, the health
warning on every headline number, and the [limitations](#what-these-numbers-do-not-capture)
section.

## Two savings, measured separately

A better forecast saves NGED money through two different mechanisms, with different prices and
different physics. We compute them as **two metrics, reported as two numbers**, and never add them
up:

1. **Flexibility procurement.** NGED pay flexible customers to reduce demand when a substation
   risks running beyond its limit. They are risk-averse and knowingly over-procure. A sharper
   forecast buys less flexibility for the same security.
2. **Curtailment of generation.** NGED curtail connected generators to keep exports within network
   limits. Curtailment forgone is generation sold. A sharper forecast curtails less.

A third saving — the engineer-hours freed by replacing a manual review of time-series plots with an
automated forecast — is real and was raised by NGED, but it is **not a leaderboard metric**: it is
the same for every model we train, so it cannot rank them. It belongs in the project's final
report, priced in engineer-hours.

## The shared idea: same risk, then compare the spend

The obvious way to price a forecast is to charge it for what goes wrong: £X per action taken, £Y
per limit breach that nobody saw coming. We cannot do that, because £Y — the true cost of a breach
— is a number NGED do not have in a usable form, and it is politically loaded (deferred
reinforcement, transformer loss-of-life, regulatory exposure).

So we invert the question. **Every model is required to be equally safe, and we compare what each
one spends to get there.** Concretely, each model is allowed to be as conservative as it likes, and
we tune that conservatism until every model leaves the same small amount of risk unaddressed. Then
the only thing left to compare is cost. This matches NGED's own description of the problem: they
are not trying to avoid a breach they currently suffer, they are trying to stop over-buying to
avoid one.

The knob we tune is the **procurement quantile** $\tau$ — how far up its own forecast distribution
a model looks when deciding to act. A timid model uses a high $\tau$, buys a lot, and is rarely
caught out; an aggressive model uses a low $\tau$ and is caught out often. Calibration finds each
model's $\tau$ such that its **unmet fraction** — the share of genuinely-needed megawatt-hours it
failed to cover — equals a common target (5% to begin with).

$$
\text{unmet fraction} = \frac{\sum_{s,t} \max(0,\; N_{s,t} - V_{s,t})}{\sum_{s,t} N_{s,t}}
$$

where $V_{s,t}$ is the volume the model would have bought (or curtailed) for substation $s$ in
half-hour $t$, and $N_{s,t}$ is the volume that actually turned out to be needed. Measuring unmet
*energy* rather than counting missed events matters: at a p99 limit the events are rare, and a
count of them is too noisy to rank models by.

**$\tau$ is calibrated on training folds only.** Tuning it on the fold being scored would let the
model see its own future, and every pound of the resulting "saving" would be lookahead.

## Metric 1 — flexibility procurement cost

For each substation $s$ and half-hour $t$, with **demand limit** $L_s$ and flexibility price
$p_{\text{flex}}$ (£/MWh):

| Quantity | Definition |
|---|---|
| Volume procured | $V_{s,t} = \max(0,\; \hat q_{s,t}(\tau) - L_s) \times 0.5$ MWh |
| Volume needed | $N_{s,t} = \max(0,\; y_{s,t} - L_s) \times 0.5$ MWh |
| Cost | $C = p_{\text{flex}} \sum_{s,t} V_{s,t}$ |

$\hat q_{s,t}(\tau)$ is the model's $\tau$-quantile forecast and $y_{s,t}$ the observed power; the
$\times 0.5$ converts MW held for a half-hour into MWh. Power is positive towards end-users, so
demand exceedance is power above $L_s$ (see the
[sign convention](forecast-building-blocks.md#sign-convention)).

**Worked example.** A primary substation with a p99 demand limit of 28 MW, on one winter evening
half-hour. Manual review forecasts 31 MW, so it procures $(31 - 28) \times 0.5 = 1.5$ MWh. Demand
turns out to be 28.6 MW, so only 0.3 MWh was needed. At £750/MWh that half-hour cost £1,125, of
which £225 was useful. A model forecasting 29.0 MW at the same calibrated risk procures 0.5 MWh —
£375, saving £750 in that half-hour. The metric is this sum over every half-hour and every
substation, scaled to £/year.

## Metric 2 — curtailment cost

Identical in shape, applied to exports, with the **export limit** $E_s$ and curtailment price
$p_{\text{curt}}$ (£/MWh of network access). Let $g_{s,t} = \max(0,\, -y_{s,t})$ be the export
magnitude (power is negative when generation flows back into the grid) and $\hat g_{s,t}(\tau)$ its
$\tau$-quantile forecast:

| Quantity | Definition |
|---|---|
| Volume curtailed | $V_{s,t} = \max(0,\; \hat g_{s,t}(\tau) - E_s) \times 0.5$ MWh |
| Volume needed | $N_{s,t} = \max(0,\; g_{s,t} - E_s) \times 0.5$ MWh |
| Cost | $C = p_{\text{curt}} \sum_{s,t} V_{s,t}$ |

**Worked example.** A generation-dominated feeder with a p99 export limit of 8.5 MW. The forecast
at its calibrated quantile says 9.6 MW, so 0.55 MWh is curtailed; actual export is 8.7 MW, so 0.1
MWh needed curtailing. At £100/MWh that is £55 spent against £10 of real constraint — £45 of
generation curtailed for nothing.

The two metrics share their arithmetic but ship as **two functions returning two numbers**, because
their prices, their limits and their direction differ, and because reporting one combined figure
would hide which mechanism a model is good at.

## What each number is compared against

A cost in isolation means nothing, so every model's cost is reported beside two reference points,
computed the same way on the same substations and half-hours:

- **Manual review** — NGED's method today: the 13-analogue ensemble read off a plot at its
  95th percentile ([the incumbent forecast](../background/nged-incumbent-forecast.md)). This is
  the bar. "Manual review" is NGED's own name for it and is more accurate than calling it a
  forecast.
- **Perfect forecast** — truth used as the forecast, procuring exactly $N_{s,t}$ with nothing
  unmet. This is the floor: no forecast can spend less and stay within limits, so it bounds the
  saving that is available to win at all.

The headline presentation is therefore *"£X/year less than manual review, which is Y% of the £Z/year
that a perfect forecast would save"*. Both metrics are computed for **every experiment** and carried
on the leaderboard.

## Choosing the limit: p95 and p99, both reported

Real network limits move with ambient temperature, with how long an overload lasts, with season and
with switching state, so no single number is correct. We use a **synthetic limit** — a percentile of
each substation's own history — and report the metric at **both p95 and p99**, side by side.

p99 is the more realistic stand-in for a network that is close to its limit only at winter peak.
p95 exists because p99 exceedances are rare enough that the resulting metric may be too noisy to
separate models, and that is a question to settle with data rather than assumption. If the two rank
models identically, we keep p99 alone.

This is the same choice, for the same reasons, as the static thresholds used by the
[tail and exceedance metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks).

## What these numbers do not capture

- **The limits are invented.** A percentile of history is not a network rating. Substations that
  are genuinely unconstrained get a limit anyway, and are scored as though flexibility were bought
  there.
- **The prices are single averages.** One £/MWh figure stands in for a tendered market with
  availability payments, utilisation payments, zone-by-zone clearing prices and finite liquidity.
- **Procurement is not per-half-hour.** NGED tender flexibility ahead, in blocks and windows, not
  half-hour by half-hour on the day. Our arithmetic assumes perfectly granular buying, which
  flatters every model equally but overstates the achievable saving.
- **The decision rule is mechanical.** A real operator applies judgement, local knowledge and
  discretion that a quantile threshold does not model.
- **Nothing is validated against real spend**, except where a trial-area substation sits in an
  actual flexibility zone. There are a couple of those, and they are the case studies that tell us
  whether the synthetic numbers are the right order of magnitude.
- **Asset failure and outage costs are excluded.** NGED identified outage quantification as
  valuable but harder; it is not in this design.
- **Over-procurement has a deliberate component.** NGED over-buy partly to stimulate the
  flexibility market and to support their capital programme. That portion is policy, not forecast
  error, and a better forecast should not be credited with removing it.

## Questions for NGED

1. **Flexibility price** — is £500–1000/MWh the right range for dispatched flexibility, which
   published dataset should we take it from, and does it cover availability payments as well as
   utilisation?
2. **Curtailment price** — is £100/MWh of network access the right figure, and is the £2M saved
   last year on the same basis, so we can sanity-check our totals against it?
3. **How much do you currently over-procure?** The entire saving is measured against this. Even
   historical procured-versus-needed volumes for a single zone would anchor it.
4. **What reliability are you actually targeting?** We assume 5% of needed megawatt-hours may go
   uncovered. What figure do you work to?
5. **How far ahead is the procurement decision made?** We assume day-ahead, which sets the forecast
   lead time the metric scores.
6. **Which trial-area substations have curtailable generation**, and which sit in a real flexibility
   zone with procurement history we can use as a case study?

## Implementation details (deleted when this ships)

- Two functions in `packages/ml_core/src/ml_core/metrics.py`, sharing a private helper that takes
  the limit, the price and the direction. They consume the same ensemble-member rows as
  [CRPS and the quantile metrics](../techniques/evaluation-metrics.md).
- Results land in the `forecast_metrics` Delta table as `flex_procurement_cost_gbp` and
  `curtailment_cost_gbp`, with `metric_param` carrying the limit level (`"p95"` / `"p99"`), so the
  two limits occupy separate rows rather than separate columns.
- Costs are **sums over time**, not means like every other metric in the table. Rows must therefore
  record the number of half-hours summed, or the totals cannot be rescaled to £/year or compared
  across folds of unequal length.
- The calibrated $\tau$ per model is an output worth storing — it says how conservative a model had
  to be to hit the common risk target, which is interpretable on its own.
- Prices and the risk target are configuration, not constants in code, so NGED's answers can be
  applied without a retrain.
- Scored on a single lead-time slice (`day_ahead` until question 5 is answered) rather than all
  horizons pooled, because a procurement decision happens once at a known lead time.
