# Probabilistic Forecasting from NWP Ensembles

> **Status: durable method explainer.** This page explains how to turn an ensemble of weather
> forecasts into an honest probability distribution over power — why the obvious approach is
> systematically overconfident, what the principled fix looks like, and the one tempting
> shortcut to avoid. It is applied by the
> [probabilistic evaluation & calibration plan](../roadmap/metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
> and shapes the percentile representations in the
> [delivery tables](../roadmap/delivery-tables.md#table-1-power_forecast).

## The intuitive sketch

Start with what we do today. ECMWF gives us **51 plausible weather futures** (ensemble members)
— 51 equally believable stories about next week's weather. We feed each story through our power
model and get 51 power curves. Where the curves fan out wide, we claim to be uncertain; where
they bunch together, we claim to be confident.

The problem: each of those 51 curves is the model's **single best guess** given that weather
story. The model is asked "what will power be if the weather does *this*?" and answers with one
number — as if, once the weather is known, it could predict power perfectly. It can't. Two of
the three reasons a forecast misses have been silently dropped, and the fan is too narrow: all
51 curves can miss reality *on the same side*.

The failure is worst exactly where it is most embarrassing. For tomorrow, the 51 weather
stories barely differ — weather models only diverge over days — so the fan collapses to nearly
a single line. The forecast claims near-certainty at the horizon where anyone can see we still
make errors:

```text
 the fan we produce today                 an honest forecast band
 (spread from weather only)               (weather + model error + noise)

 power                                    power
   │            ____.--~~                   │         ░░░░░░░░░░░░
   │      ═════      ~~--__                 │     ░░░░▒▒▒▒▒▒▒▒░░░░░░
   │ ═════x    ¯¯--__                       │ ▒▒▒▒x▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░
   │            x        x                  │     ▒▒▒▒x▒▒▒▒▒▒▒▒x▒▒▒░░
   │                                        │         ░░░░░░░░░░░░░░
   └────────────────────────▶ time          └──────────────────────▶ time
    day 1      day 4      day 7              day 1    day 4     day 7

    x = what actually happened. On day 1 the members haven't diverged yet,
    so the fan is a confident thin line — and reality falls outside it.
```

The fix, in one sentence: instead of asking the model for one number per weather story, ask it
for a **range** per weather story — "given this weather, power will most likely land between
here and here, with this shape" — and then **overlay all 51 fuzzy bands on top of each other**.
Where many bands pile up, the combined forecast is confident; the outermost edges of the pile
are the tails. Reading percentiles off that pile gives one honest distribution that contains
*both* kinds of spread: the weather stories disagreeing with each other, *and* each story's own
admission that weather doesn't fully determine power.

The rest of this page makes that sketch precise.

## Where forecast uncertainty actually comes from

Total predictive uncertainty decomposes into three terms:

1. **Weather uncertainty** — we don't know what the weather will do. This is what the 51 NWP
   members sample: their divergence *is* the weather uncertainty, growing with lead time.
2. **Weather→power model error** — even given the true weather, our model's mapping from
   weather to power is imperfect (finite training data, missing features, wrong functional
   form).
3. **Intrinsic noise** — even a perfect model given perfect weather couldn't predict power
   exactly: meter noise, occupant behaviour, unmodelled local events. Irreducible randomness.

A deterministic model (XGBoost trained with squared error learns the **conditional mean**)
pushed through 51 members captures term 1 *only*. Terms 2 and 3 are dropped entirely, at *every*
lead time. The resulting ensemble is **under-dispersed** — systematically overconfident — and the
failure is most conspicuous at short horizons, where term 1 has barely grown, so the fan
collapses toward the single line of the sketch above while our errors do not. The
[spread-skill ratio](evaluation-metrics.md#spread-skill-ratio) is the metric
that catches this: spread ÷ error ≪ 1 means the fan is too thin.

This matters commercially, not just statistically: flexibility procurement keys off the
**tails** (P90+ peaks), and under-dispersion is precisely the error that makes tails look safer
than they are.

### Long horizons are not automatically safe

The long-horizon end looks like the safe one. By day 8–14 the 51 weather stories have diverged
enormously, so the fan is anything but a thin line — the opposite of the short-horizon failure
above. But the same two dropped terms still bite here; they just hide better. A wide *input* fan
does not guarantee an honest *output* fan, because the learned weather→power mapping sits between
them, and squared-error training quietly narrows what that mapping passes through as lead time
grows.

Far out, ECMWF's individual trajectories are themselves close to noise: the model was trained on
forecast weather as its input, and at long lead times that input is wrong by a large,
lead-time-dependent amount. Squared-error training responds to noisy inputs by hedging toward the
mean — attenuating the fitted weather→power sensitivity — and today's model applies one
sensitivity across all lead times, because it has no feature telling it how trustworthy the
weather input is. So the day-8–14 fan can be merely *wide-looking*: large NWP spread pushed
through a damped, horizon-blind sensitivity, with the dropped term-2 residual (largest exactly
where the weather input is least trustworthy) never added back in. This is why the
[spread-skill ratio and PICP](evaluation-metrics.md#probabilistic-metrics) are computed per
horizon slice and not just overall: a visibly fanned-out `extended_range` can still be
under-dispersed, and only the numbers, not the eye, can say so.

The mechanism also sharpens the case for giving the model `nwp_lead_time_hours` as a feature
([#230](https://github.com/openclimatefix/nged-substation-forecast/issues/230)). Beyond the
double-counting mitigation discussed in the caveat below, the lead-time feature is what lets the
model learn *how much* to attenuate its weather sensitivity per horizon — narrow when the weather
input is trustworthy, wide when it isn't — instead of the one horizon-averaged compromise it is
forced into today.

## The fix, formally: a mixture of conditional distributions

Ask the model for a full conditional distribution per member — "the distribution of power
*given* weather scenario $m$", written $F_m(y) = P(\text{power} \le y \mid \text{weather}_m)$.
In practice $F_m$ arrives as a set of quantiles (e.g. p1…p99 from a quantile-regression model);
that per-member quantile set is exactly
[Representation 3](../roadmap/delivery-tables.md#representation-3-ensemble-of-percentile-forecasts)
in the delivery tables. Each conditional distribution carries terms 2 and 3; the *disagreement
between* members carries term 1.

Combining them is the law of total probability. The members are designed to be equiprobable, so
the total predictive distribution is the equal-weight **mixture** — in the forecast-combination
literature, the **linear (opinion) pool**:

$$
F(y) \;=\; \frac{1}{51} \sum_{m=1}^{51} F_m(y)
$$

Read it as: *the probability that power lands below $y$ is the average, over the 51 weather
stories, of the probability of landing below $y$ within each story.* Nothing deeper than that —
but it is the decomposition that guarantees both kinds of spread survive into the final
distribution. This pooled distribution, expressed as percentile columns with one row per
`valid_time`, is
[Representation 2](../roadmap/delivery-tables.md#representation-2-percentiles) — i.e.
Representation 2 is *derived from* Representation 3 by pooling.

## Turning 51 quantile sets into one: the pooling recipe

Inverting the mixture CDF sounds like numerical work, but with quantile forecasts there is a
trick that makes it three lines of Polars:

1. **Sort each member's quantiles.** Quantile-regression models fit each quantile level
   independently, so a member's p80 can come out *below* its p65 ("quantile crossing").
   Sorting the values within each member — "monotonic rearrangement" — is the standard,
   theoretically sound fix.
2. **Treat every quantile value as an equiprobable pseudo-sample.** A member's quantiles at
   equally-spaced levels are, by construction, equally likely places for power to land within
   that weather story. So pool all 51 × K values (51 members × K quantile levels) for a given
   `(time_series_id, valid_time)` into one bag.
3. **Take empirical percentiles of the bag.** The bag's p95 *is* the p95 of the mixture — this
   recipe is exactly the inverse of the averaged CDF above (a `group_by` + `pl.quantile`).

One practical note on step 2: the recipe is exact when the K levels are equally spaced. Our
delivery levels are deliberately tail-heavy (p1, p2, p5, …, p98, p99), so either have the model
emit a denser equally-spaced set internally and reduce to the delivery levels at the end, or
weight the pseudo-samples by the probability gap each one represents.

## The tempting shortcut that doesn't work: averaging the quantiles

The "obvious" way to combine 51 percentile forecasts is to average them level by level: the
combined p90 is the mean of the 51 members' p90s (this has a name: **Vincentization**). It
looks harmless and is subtly wrong for our purpose.

Averaging quantile-by-quantile averages the *locations* of the 51 distributions and keeps
roughly their *individual* widths — the disagreement **between** members (term 1, the weather
uncertainty!) is discarded rather than added. Picture 51 narrow bands scattered up and down the
power axis: the mixture says "the truth is somewhere in this whole scattered pile"; Vincent
averaging says "the truth is in one narrow band at the pile's centre". Vincentization is a
respectable combiner in settings where the members are near-identical copies shifted slightly —
but here it would *re-create the exact under-dispersion this whole design exists to fix*. Use
the mixture.

(A useful mnemonic: the mixture **adds** the between-member spread to the within-member spread;
Vincentization **averages it away**. For fattening tails, you want addition.)

## Caveat: double-counting weather uncertainty

There is one way the mixture can overshoot. The conditional models are trained against
*forecast* weather (a member's trajectory), not the weather that actually happened. So the
spread each model learns — "given this weather input, how uncertain is power?" — already
includes some weather-forecast error, because at training time the weather input itself was
wrong by some amount. Mixing over 51 members then adds weather spread *again*. At long
horizons, where NWP error is large, the pooled distribution can come out **over**-dispersed.

Mitigations, in order of practicality:

- **Give the model the lead time as a feature** (`nwp_lead_time_hours`, planned in [#230](https://github.com/openclimatefix/nged-substation-forecast/issues/230)), so the learned
  conditional spread can be narrow when the weather input is trustworthy and wide when it
  isn't, rather than one horizon-averaged width.
- **Train on many ensemble members, not just the control** (planned in [#148](https://github.com/openclimatefix/nged-substation-forecast/issues/148)) — the model then sees "one plausible
  scenario" as its actual input distribution, which is what a member is at inference time.
- **Measure, then correct**: check the [spread-skill ratio and PICP](evaluation-metrics.md#probabilistic-metrics) of the *pooled* forecast per
  horizon slice, and apply a per-horizon affine recalibration of the pooled quantiles (fit on
  the training window) only if the numbers demand it.

Keep the asymmetry in mind: for a tails-driven product, mild **over**-dispersion (slightly
over-stated risk) is a far safer failure mode than the under-dispersion it replaces
(under-stated risk). Fix double-counting if measured, but don't fear it more than the disease.

## How this maps onto the project

- **Diagnose first**: the spread-skill / PICP / pinball / CRPS metrics that quantify all of the
  above are implemented — see the [evaluation-metrics reference](evaluation-metrics.md) — as
  Phases A–B of the
  [probabilistic evaluation plan](../roadmap/metrics-and-leaderboard.md#delivering-the-probabilistic-metrics).
- **Cheap stopgap**: post-hoc spread inflation of the existing deterministic ensemble ([Phase C
  of the probabilistic evaluation plan](../roadmap/metrics-and-leaderboard.md#phase-c-cheap-calibration-after-b-proves-the-diagnosis)) buys calibration without any of this machinery — the full mixture pipeline must beat
  it to earn its keep.
- **The full pipeline** ([Phase D of the probabilistic evaluation plan](../roadmap/metrics-and-leaderboard.md#phase-d-ensemble-of-quantile-forecasts-representation-3-pooled-representation-2)): a quantile-objective XGBoost emits Representation 3;
  the pooling recipe above derives Representation 2 for
  [delivery](../roadmap/delivery-tables.md#table-1-power_forecast).
