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
and the plan may change with each result. The six parts are listed at the end of this page.

## How the Part 2 model works

**A generalised additive model (GAM) says that demand is a sum of simple pieces.** Each piece
describes one effect on its own: the time of day, the day of the week, a holiday, the temperature,
or the sun. The GAM adds the pieces to get the forecast. Any single piece can be pulled out and
plotted, so each effect stays separate and easy to read. That readability is why people who forecast
demand like GAMs.

**In a textbook GAM, each piece is a smooth curve whose shape the GAM learns from the data.** The
usual tool is a spline. Picture a flexible ruler pinned at regular points, called knots. The GAM
chooses how high the ruler sits at each knot, and the curve bends smoothly between knots. A penalty
on wiggliness stops the curve chasing noise. Lower error pulls the curve towards the data, and the
penalty pulls the curve towards a straight line. The balance between the two sets how bendy each
curve ends up.

**The Part 2 model is a stricter cousin of the textbook GAM.** The Part 2 model fixes the shape of
most pieces in advance, using physics and common sense, and learns only how big each piece is. The
whole fit is then one penalised least-squares problem with exactly one best answer. The Part 2 model
forecasts demand minus solar output, in points of [effective
capacity](../glossary.md#capacity-and-points-of-capacity) (1% of a series' effective capacity). It
has five pieces, plus two optional pieces that only some arms of experiment 104 used. The pieces use
four blocks of hours in the day: night (00:00 to 05:59), morning (06:00 to 10:59), midday (11:00 to
15:59), and evening (16:00 to 23:59).

- **Daily profile.** One level for every half-hour of the day, with separate profiles for weekdays,
  Saturdays, Sundays, and bank holidays.
- **Holidays.** Each named holiday period, such as Easter or the school summer break, gets its own
  adjustment for each block of hours. A small tent-shaped term, rising and falling around Christmas
  Day, is added on top.
- **Heating.** The Part 2 model takes how far the temperature sits below 15.5 °C. At or above 15.5
  °C, the heating term is zero. Below 15.5 °C, demand rises in a straight line as the temperature
  falls. That bent line is called a hinge. Buildings warm up and cool down slowly, so the Part 2
  model uses the temperature averaged over 6 hours, 1 day, 3 days, and 1 week, and learns a slope
  for each average in each block of hours. Every slope is constrained to be zero or positive,
  because a colder day cannot reduce heating demand. The temperature is an estimate of past weather
  (ERA5, see the [glossary](../glossary.md#ens-era5-and-cams)).
- **Lighting.** Four small terms measure how dark it is, so the Part 2 model can add demand on
  winter evenings, when lights come on earlier.
- **Solar.** For each half-hour, the Part 2 model calculates how much a panel of unit capacity would
  produce, from the estimated sunlight (CAMS irradiance, also an estimate of past weather) and the
  position of the sun, and reduces that output by 0.4% per kelvin of panel heating. The Part 2 model
  then learns the capacity of solar panels hidden behind the substation, one capacity per
  [season-year](../glossary.md#season-year) (a year from July to June), and subtracts the panels'
  output. The capacity may grow from one season-year to the next, but never shrinks.
- **Drift (optional; used only in the arms of experiment 104 that include drift).** Each block of
  hours gets a level that can move slowly. The level is set at monthly knots, with a penalty on
  sharp bends. Drift is the one piece that is a true spline. Drift absorbs slow change, such as a
  new housing estate or a factory closing, and a forecast holds the last value.
- **Steps (optional; used only in one arm of experiment 104).** Occasional sudden level changes,
  kept rare by a penalty on their total size (an L1 penalty, which keeps most steps at exactly
  zero).

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
increments), so the fit cannot be a one-line matrix solution. Instead, a small solver for
least-squares problems with sign limits on some coefficients runs on the graphics card. In the
experiment 104 benchmark, one fit on six substations took 1.6 seconds on the graphics card and 3.3
seconds on the processor.

**The uncertainty bands come from a separate step.** Fit the same model 12 months before the end of
the training data, measure how wrong that fit was over the following 12 months, and take quantiles
of those errors for each block of hours and each quarter of the year. Those quantiles become the
bands around the forecast.

**The Part 2 model is the starting point for the full model.** The full model described next keeps
this additive structure and replaces some fixed shapes with learned ones. Experiment 104's verdict
was `inconclusive`: its scores with all usable years, with and without drift, could not be told
apart from its score with the last 12 months only.

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
   the daily shapes, plus a small correction of its own that a prior pulls towards zero. The
   substations would also share one holiday module and one heating-curve shape, which each
   substation scales. A new substation with a few weeks of history would borrow almost everything,
   and would learn only its own mix and scale.
3. **Add a small interaction neural network, the first truly neural part.** A cold Sunday is not a
   cold day plus a Sunday, because people at home heat differently from offices. The neural network
   would take calendar and weather inputs together and add a penalised correction. The neural
   network would never see sunlight or the trend in time, so it could not quietly absorb the solar
   or drift pieces and break interpretability.
4. **Wrap two layers around the demand model.** A Kalman filter (a method that updates an estimate
   of a hidden quantity each time a new reading arrives) would track, for each block of hours, a
   persistent level and a short-lived anomaly. A persistent level is a change that stays, such as a
   new load or a permanent transfer. An anomaly is a deviation that fades, such as a misjudged cold
   snap or a football match. The Kalman filter plays the role that lag features play in XGBoost. A
   meter layer would predict real and reactive power and calculate what an [MVA
   meter](../glossary.md#mw-and-mva) would read. An MVA meter reads apparent power, which is never
   negative, so the reading "bounces" back up after the real power falls through zero, as it does
   when a substation exports solar power at midday. The framework module contains the meter layer,
   but only a placeholder for the Kalman filter's state.

**Two smaller additions are planned.** One is an optional flat unmetered generator, included only if
the data argues strongly for one. The other is a soft ceiling on solar output: when many inverters
(the devices that convert panel output to alternating current) each cap their output at their own
rating, the total flattens gradually and not abruptly. The framework module has switches for both
additions and for drift in the pooled model, and turning any of them on currently raises an error
(`DpConfig.check_supported` rejects them).

## Is the full model still a GAM?

**Mostly, but the full model would go beyond a GAM in three places.** The demand core stays
additive: each piece can still be pulled out and plotted. Priors do not change that, because the
wiggliness penalty of a GAM is mathematically a Gaussian prior. A GAM with priors on its pieces,
shared across substations, is a Bayesian hierarchical GAM. An interaction term learned by a small
neural network is close to a neural additive model, [Agarwal et al.
(2021)](https://arxiv.org/abs/2004.13912), which builds an additive model from one small neural
network per input. The full model would extend that design with a network that takes calendar and
weather inputs together.

**The three places beyond a GAM are the shared shapes, the Kalman filter, and the meter layer.**
Shared shapes multiplied by each substation's mix form a low-rank model, meaning that many
substations are described by a few shared curves. Because the coefficients differ by substation, it
is also a varying-coefficient model. The Kalman filter makes the full model a state-space model, in
which the GAM gives the expected demand and the filter tracks the hidden persistent level and
anomaly. The meter layer applies a square root to the GAM's output (apparent power is the square
root of the sum of the squares of real and reactive power), so the GAM sits inside an equation for
what the meter reads. A fair short description is "a hierarchical additive model with a small neural
network for interactions, wrapped in a Kalman filter and a meter equation". Each piece keeps a
physical meaning and can be plotted, and that interpretability is what the series aims to keep,
whatever else it adds.

## The six parts

**The series is planned as six parts, and each part is planned from the result of the part before.**
The list below is the current intention and not a commitment. Each theme's research plan page holds
the plan for its own parts.

1. **Data look, with no model.** Which substations, rows, and dates are trustworthy. Experiment 103
   ([merged](../00103-differentiable-physics-1-data-look/result.md)) decided each substation's start
   date, a fixed lockdown window (23 March 2020 to 19 July 2021) that training leaves out, and which
   rows of the substations metered in MVA count as dark rows (rows where CAMS irradiance is below 1
   W/m², so solar output is zero and the rows show demand alone).
2. **A GAM for each substation.** Does the model gain from older years, with and without drift?
   ([Experiment 104](../00104-differentiable-physics-2-simple-structural-model/result.md), merged,
   verdict `inconclusive`.)
3. **Pooling.** Does one shared model match separate models when there is plenty of history, and
   beat them when there are only 4 weeks or 3 months of history?
4. **Long history.** Does the pooled model, trained back to 2019 with drift, beat its best shorter
   window? This part would bring in the Kalman filter. The comparison is a 2×2 table: with and
   without drift, crossed with with and without the Kalman filter. A baseline that tracks a running
   average of recent demand is added for reference.
5. **Embedded generation.** A soft solar clip, a flat unmetered generator without inventing one, and
   the MVA meter layer.
6. **A real-world test on the [ECMWF ensemble forecast (ENS)](../glossary.md#ens-era5-and-cams),
   against the [second best-guess recipe (BG2)](../glossary.md#bg2).** Four arms: the
   differentiable-physics model alone, the differentiable-physics model supplying XGBoost's
   [base margin](../glossary.md#base-margin) (the starting prediction for each row), the GAM with
   the Kalman filter, and the GAM supplying XGBoost's base margin.
