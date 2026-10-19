# Differentiable physics: background

**This page explains, for a reader who has never fitted a generalised additive model, how the model
of Part 2 works and how the planned full model would extend it.** The six-part
differentiable-physics series builds models of [Primary
substation](../glossary.md#primary-substation) demand from physical terms: a daily profile,
holidays, heating, and solar output. The [techniques
page](../../techniques/differentiable-physics.md) explains the method in general. [Experiment
103](../00103-differentiable-physics-1-data-look/result.md) looked at the data, and [experiment
104](../00104-differentiable-physics-2-simple-structural-model/result.md) fitted the Part 2 model.
Only what those two experiments built counts as built. Everything under "The full model" is a plan,
and the plan may change with each result.

## How the Part 2 model works

**A generalised additive model (GAM) says that demand is a sum of simple pieces.** Each piece
describes one effect on its own: the time of day, the day of the week, a holiday, the temperature,
or the sun. The model adds the pieces to get the forecast. Any single piece can be pulled out and
plotted, so each effect stays separate and easy to read. That readability is why people who forecast
demand like GAMs.

**In a textbook GAM, each piece is a smooth curve whose shape the model learns from the data.** The
usual tool is a spline. Picture a flexible ruler pinned at regular points, called knots. The model
chooses how high the ruler sits at each knot, and the curve bends smoothly between knots. A penalty
on wiggliness stops the curve chasing noise. Lower error pulls the curve towards the data, and the
penalty pulls the curve towards a straight line. The balance between the two sets how bendy each
curve ends up.

**The Part 2 model is a stricter cousin of the textbook GAM.** The model fixes the shape of most
pieces in advance, using physics and common sense, and learns only how big each piece is. The whole
fit is then one penalised least-squares problem with exactly one best answer. The model measures
demand minus solar output, in points of effective capacity, and has seven pieces:

- **Daily profile.** One level for every half-hour of the day, with separate profiles for weekdays,
  Saturdays, Sundays, and bank holidays.
- **Holidays.** Each named holiday period, such as Easter or the school summer break, gets its own
  adjustment for each block of hours in the day. A small tent-shaped term, rising and falling around
  Christmas Day, is added on top.
- **Heating.** The model takes how far the temperature sits below 15.5 °C. At or above 15.5 °C, the
  heating term is zero. Below 15.5 °C, demand rises in a straight line as the temperature falls.
  Buildings warm up and cool down slowly, so the model uses the temperature averaged over 6 hours, 1
  day, 3 days, and 1 week, and learns a slope for each average in each block of hours. Every slope
  is constrained to be zero or positive, because a colder day cannot reduce heating demand.
- **Lighting.** Four small terms measure how dark it is, so the model can add demand on winter
  evenings, when lights come on earlier.
- **Solar.** For each half-hour, the model calculates how much a panel of unit capacity would
  produce, from the measured sunlight and the position of the sun, and reduces that output by 0.4%
  per kelvin of panel heating. The model then learns the capacity of solar panels hidden behind the
  substation, one capacity per season-year, and subtracts the panels' output. The capacity may grow
  from one season-year to the next, but never shrinks.
- **Drift.** Each block of hours in the day gets a level that can move slowly. The level is set at
  monthly knots, with a penalty on sharp bends. Drift is the one piece that is a true spline. Drift
  absorbs slow change, such as a new housing estate or a factory closing, and a forecast holds the
  last value.
- **Steps (one arm of experiment 104 only).** Occasional sudden level changes, kept rare by a
  penalty on their total size (an L1 penalty).

**The yearly cycle comes from several pieces together, not from one piece.** Heating carries the
winter rise, and is driven by the real temperature rather than by the date. Lighting adds the dark
evenings. Solar takes away the summer middays. Underneath sit two smooth waves across the year, one
a year long and one half a year long, which pick up any seasonal pattern that weather and daylight
do not explain. Because the season comes mostly from weather and daylight, an unusually cold March
gets a cold-weather forecast and not an average-March forecast.

**Fitting is one large linear regression, because every piece is a number multiplied by a column of
inputs.** Build a table with one row per half-hour and one column per input: the profile slots,
holiday flags, temperature hinges, darkness ramps, solar shape, and drift knots. Then find the
coefficients that minimise squared error plus the penalties. The penalties act like priors: each
penalty pulls a coefficient towards zero, or towards a smooth path, unless the data argues
otherwise. Some coefficients must stay positive (the heating slopes and the solar capacity
increments), so the fit cannot be a one-line matrix solution. Instead, a small constrained solver
runs on the graphics card. In the experiment 104 benchmark, one fit on six substations took 1.6
seconds on the graphics card and 3.3 seconds on the processor.

**The uncertainty bands come from a separate step.** Fit the same model 12 months before the end of
the training data, measure how wrong that fit was over the following 12 months, and take quantiles
of those errors for each block of hours and each quarter of the year. Those quantiles become the
bands around the forecast.

**The Part 2 model is the starting point for the full model.** The full model described next keeps
this additive structure and replaces some fixed shapes with learned ones. Experiment 104 found that
the Part 2 model forecast better when trained on every usable year than on the last 12 months only,
and that adding drift halved that gain. The verdict was `inconclusive`.

## The full model

**The full model is a plan with four changes, and each change can be switched on alone, so a later
experiment can tell which change helps.** The framework module built for experiment 104
(`packages/studies/src/studies/dp_pooled.py`) contains switches for several of these changes. At the
time of writing, no merged experiment has scored them. The 6.6-times GPU speed-up reported in
experiment 104 came from a proxy for the pooled model and not from a scored forecast.

1. **Learn shapes that Part 2 fixes.** The heating threshold becomes a parameter with a prior (15.5
   ± 2 °C), so each substation can move its threshold, and the model learns how to blend the four
   temperature averages. The fit then loses its single guaranteed best answer and runs by gradient
   descent on the graphics card.
2. **Pool across substations, which is the heart of the series.** All substations would share a
   small library of daily shapes, perhaps four: one residential with a big evening peak, one
   commercial and flat through the working day, and so on. Each substation learns its own mix of
   those shapes, plus a small correction of its own that a prior pulls towards zero. The substations
   would also share one holiday module and one heating-curve shape, which each substation scales. A
   new substation with a few weeks of history would borrow almost everything, and would learn only
   its own mix and scale.
3. **Add a small interaction neural network, the first truly neural part.** A cold Sunday is not a
   cold day plus a Sunday, because people at home heat differently from offices. The neural network
   would take calendar and weather inputs together and add a penalised correction. The neural
   network would never see sunlight or the trend in time, so it could not quietly absorb the solar
   or drift pieces and break interpretability.
4. **Wrap two layers around the demand model.** A Kalman filter would track, for each block of
   hours, a persistent level (new loads, or transfers that stay) and a short-lived anomaly that
   fades (a misjudged cold snap, a football match, Saharan dust). The Kalman filter plays the role
   that lag features play in XGBoost. A meter layer would predict real and reactive power and
   calculate what an [MVA meter](../glossary.md#mw-and-mva) would read, which handles the "bounce"
   through zero when a substation exports solar power. The framework module contains the meter
   layer, but only a placeholder for the Kalman filter's state.

**Two smaller additions are planned.** One is an optional flat unmetered generator, included only if
the data argues strongly for one. The other is a soft ceiling on solar output where inverters clip.
The framework module has switches for both, and turning either on currently raises an error.

## Is the full model still a GAM?

**Mostly, but the full model would go beyond a GAM in three places.** The demand core stays
additive: each piece can still be pulled out and plotted. Priors do not change that, because the
wiggliness penalty of a GAM is mathematically a Gaussian prior. A GAM with priors on its pieces,
shared across substations, is a Bayesian hierarchical GAM. An interaction term learned by a small
neural network also stays in the family: an additive model built from small neural networks is a
[neural additive model (Agarwal et al., 2021)](https://arxiv.org/abs/2004.13912).

**The three places beyond a GAM are the shared shapes, the Kalman filter, and the meter layer.**
Shared shapes multiplied by each substation's mix form a low-rank, varying-coefficient model. The
Kalman filter is a state-space model with the GAM as its expected value. The meter layer applies a
square root to the GAM's output, so the GAM sits inside a physical model of the measurement. A fair
short description is "a hierarchical additive model with a small neural network for interactions,
wrapped in a Kalman filter and a model of the meter". Each piece keeps a physical meaning and can be
plotted, and that interpretability is what the series aims to keep, whatever else it adds.

## The six parts

**The series is planned as six parts, and each part is planned from the result of the part before.**
The list below is the current intention and not a commitment. Each theme's research plan page holds
the plan for its own parts.

1. **Data look, with no model.** Which substations, rows, and dates are trustworthy: start dates,
   the fixed lockdown window, and the dark rows of the MVA substations ([experiment
   103](../00103-differentiable-physics-1-data-look/result.md), merged).
2. **A GAM for each substation.** Does the model gain from older years, with and without drift?
   ([Experiment 104](../00104-differentiable-physics-2-simple-structural-model/result.md), merged,
   verdict `inconclusive`.)
3. **Pooling.** Does one shared model match separate models when there is plenty of history, and
   beat them when there are only 4 weeks or 3 months of history?
4. **Long history.** Does the pooled model, trained back to 2019 with drift, beat its best shorter
   window? This part would bring in the Kalman filter, as a 2×2 comparison of drift against Kalman
   filter, plus a running-average baseline.
5. **Embedded generation.** A soft solar clip, a flat unmetered generator without inventing one, and
   the MVA meter layer.
6. **A real-world test on the [ENS](../glossary.md#ens-era5-and-cams) forecast, against
   [BG2](../glossary.md#bg2).** Four arms: the differentiable-physics model alone, the
   differentiable-physics model as the starting point for XGBoost, the GAM with the Kalman filter,
   and the GAM as the starting point for XGBoost.
