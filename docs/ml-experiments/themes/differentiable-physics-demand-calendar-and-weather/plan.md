# Differentiable physics (demand, calendar, and weather): research plan

**The next step is a drift term that re-estimates its level after the end of the training data.**
Experiment 104 found that all years of history without drift scored worse than 12 months, and that a
drift held constant recovered that and helped in every scored month, leaving all years with drift
level with 12 months.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Drift from recent demand (proposed) | Does a drift that re-estimates its level from recent demand beat a drift held constant? | 1 | Part 3, which pools the model across substations | Proposed |
| N3 and N4, interaction terms and heating threshold | Do interaction terms in the convex model, and a heating threshold chosen per substation, lower its error? | Running | Nothing | Running (experiment 114) |
| Smoothed temperatures at two time constants | Do exponentially smoothed temperatures with two decay rates, plus smoothed daily minimum and maximum, lower the convex model's error? | 2 | Experiment 114, which changes the same temperature terms | Proposed |
| [E7, storage-heating night step](../../00123-differentiable-physics-e7-storage-heating-night-step-estimation/result.md) | How large is the winter night step at Primary substations, pooled over the six decision series, and how does it grow with cold? | Running | Winter 2025-26 waits for experiment 114 to merge | Running (experiment 123) |
| UTC-indexed night term | Does a small level per hour band in UTC hours, from 22:00 to 08:00 UTC, remove error that the local-time daily profile leaves around the clock changes? | 3 | A descriptive check of residuals against UTC hour | Waiting on a descriptive check |
| N11, weather encoder | Does a small learned encoder over a 5 by 5 patch of weather cells around each substation lower the error on convective and frontal days? | 4 | GPU hours (4 to 8 for the run) | Proposed |

Later parts are listed in their own themes.

## Smoothed temperatures at two time constants

**The proposal adds two exponentially decayed temperature columns and smoothed daily extremes to the
convex model.** The GAM papers on regional load use temperature smoothed at two decay rates per
half-hourly step, 0.95 and 0.99, plus smoothed daily minimum and maximum temperatures.
[Obst et al. (2021)](https://arxiv.org/abs/2009.06527) use these terms for French load, and [Fasiolo
et al. (2021)](https://arxiv.org/abs/1707.03307) report the same smoothing for the winning
GEFCom2014 entry. No paper isolates the gain from the smoothing alone. At half-hourly steps the two
rates correspond to roughly 10 hours and 50 hours. The convex model already uses means of 6 hours to
1 week, which are the same idea, so the experiment tests whether the two decay rates and the daily
extremes add anything to those means. It runs in minutes on the CPU.

## UTC-indexed night term

**Economy 7 and storage-heater timers run on Greenwich Mean Time all year, so their load switches on
at fixed UTC hours, while the convex model's daily profile follows local clock time.** The proposal
adds one small level per hour band in UTC hours from 22:00 to 08:00 UTC, on top of the local-time
profile. Around the two clock changes, the local profile places the timer-driven load an hour away
from where it falls. The first step is a descriptive check: plot the model's residuals at night
against UTC hour, separately for the weeks either side of each clock change. The term is only worth
fitting if those residuals shift with the clock change.

## N11: weather encoder

**The proposal replaces the single-cell ERA5 and CAMS inputs with a small learned encoder over a 5
by 5 patch of cells around each substation.** The aim is a lower error on convective and frontal days,
when a single cell misses the neighbouring weather. The run needs 4 to 8 GPU hours, so it waits for the
GPU budget. The [encoders technique page](../../../techniques/encoders.md) is the first thing to read.

## Planned experiments

<!-- planned-table -->
