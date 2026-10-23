# Differentiable physics (demand, calendar, and weather): research plan

**The cheap screens of inputs to the convex model are bundled into one experiment.** Experiment 104
found that all years of history without drift scored worse than 12 months, and that a drift held
constant recovered that. The recent-level drift and the day-type filter are covered by the KFE
experiment, so this page lists the input screens only.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| N3 and N4, interaction terms and heating threshold | Do interaction terms in the convex model, and a heating threshold chosen per substation, lower its error? | Done, partly supported | None | Interaction terms (N3) gained, mostly at one substation, and the per-site heating threshold (N4) tied (experiment 114) |
| S1, cheap screens added to GAM-K (experiment 126) | Do cheap additions to the day-ahead GAM-K lower the score, winter first? | Done, inconclusive | None | None of 13 screens showed a trusted gain; E-sol (-0.036 on dark winter evenings) is the lead for a retest on ENS weather; W-basis worse in every month, site and year; F-mix, W-stn undecided on 2 to 3 months |
| E7F, Economy 7 night-step term in the forecast (experiment 137) | Does an additive night term, fitted on the base forecast's own past errors, lower the day-ahead score on cold winter nights from 01:00 UTC? | Running | None | Planned; plan committed before any code |
| Weather encoder retest | Does a small learned encoder over a patch of weather cells lower the error on hard cases (convective and frontal days, wind and solar sites) once the base forecast is recalibrated? | 3 | A plain recalibration of GAM-K's quantiles, which experiment 119 found accounts for most of the encoder's gain | Queued, after the recalibration |

Later parts are listed in their own themes.

## Bundled cheap screens

**The six screens share one experiment and run on 6 CPU cores in about 2 hours.** Each has one
hypothesis:

- **Free temperature spline.** A free spline of temperature fits better than the fixed 15.5 °C
  hinge.
- **Smoothed temperatures.** Temperature smoothed at two decay rates, plus smoothed daily minimum and
  maximum, adds skill to the convex model's means of 6 hours to 1 week. The GAM papers on regional
  load use these terms: [Obst et al. (2021)](https://arxiv.org/abs/2009.06527) for French load, and
  [Fasiolo et al. (2021)](https://arxiv.org/abs/1707.03307) for the winning GEFCom2014 entry. No paper
  isolates the gain from the smoothing alone.
- **Humidity and wind chill.** Feeding humidity and wind chill into the heating hinge lowers the
  winter error.
- **School-term calendar.** A school-term term lowers the error in term-time weeks, once data
  availability is checked.
- **Lighting from solar elevation.** Lighting modelled from solar elevation beats a fixed clock curve
  on dark winter evenings.
- **Filter-noise mixture.** A filter noise that depends on the regime (cold snap, mild, holiday)
  beats one fixed noise.

## Weather encoder retest

**The retest repeats the weather-encoder comparison on the hard cases, with a recalibrated base
forecast.** The encoder replaces single-cell inputs with a learned view of a patch of cells, and the
aim is a lower error on convective and frontal days, when a single cell misses the neighbouring
weather. Experiment 119, the first encoder experiment, ended inconclusive, and about two-thirds of
its gain survived when the network read the wrong date's weather. The retest therefore waits for a
plain recalibration of GAM-K's quantiles, and runs on the GPU. The [encoders
technique page](../../../techniques/encoders.md) is the first thing to read.

## Planned experiments

<!-- planned-table -->
