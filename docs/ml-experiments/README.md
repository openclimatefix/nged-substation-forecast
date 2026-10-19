# ML experiments

**Each folder here is one small experiment that tests one hypothesis about improving the production
forecast, and the experiments are grouped into themes.** A theme has three pages: a Summary that
states the answer so far, an Experiments page that lists every experiment tried, and a Research plan
that lists what is planned next. The folders are named `<id>-<slug>`, where the id is a five-digit
integer that orders the experiments by the order in which they were started. A folder holds
`result.md` and its figures.

## Themes

- **[Calibration and uncertainty](themes/calibration-and-uncertainty/index.md):** Whether the
  forecast's quantiles and ensemble members are well calibrated, and how to widen or sharpen them.
- **[Curtailment cost](themes/curtailment-cost/index.md):** Scoring a forecast by the curtailment
  cost of its errors, and by how well its tails are calibrated.
- **[Demand inputs](themes/demand-inputs/index.md):** Which extra inputs lower the error of the
  substation demand forecast.
- **[Generator forecasting](themes/generator-forecasting/index.md):** Forecasting the output of
  solar and wind farms.
- **[Hierarchy and pooling](themes/hierarchy-and-pooling/index.md):** Forecasts that share
  information across substations or across levels of the electricity network hierarchy.
- **[Lagged power and error correction](themes/lagged-power-and-error-correction/index.md):** Where
  recent measured power should enter the forecast, and how to correct the forecast's recent errors.
- **[Production reference and best guess](themes/production-reference-and-best-guess/index.md):**
  How the production forecast compares with the best-guess recipes and with a perfect weather input.
- **[Retraining and training history](themes/retraining-and-training-history/index.md):** How often
  to refit the XGBoost models, and how much training history to give them.
- **[Solar disaggregation](themes/solar-disaggregation/index.md):** Separating embedded solar from
  substation demand.
- **[Switching detectors](themes/switching-detectors/index.md):** Detecting load transfers between
  substations, called switching events, from power data alone.
- **[Switching labels in the forecast](themes/switching-labels-in-the-forecast/index.md):** Using
  switching labels to forecast the demand under the normal running arrangement.
- **[Weather products and blending](themes/weather-products-and-blending/index.md):** Which weather
  data improves the forecast, and whether blending several weather products helps.

## How an experiment is recorded

**The front matter of `result.md` is the only place that records an experiment's theme, verdict, and
status, and the theme pages build their tables from it.** The Experiments table of each theme, the
planned-experiments table of each Research plan, and the candidates table below are generated from
the front matter when the documentation is built, so adding a finished experiment means writing its
`result.md` and editing nothing else. The `ml-experiment` skill
(`.claude/skills/ml-experiment/SKILL.md`) lists the front matter fields and their allowed values.

`uv run python scripts/mkdocs_hooks/experiment_tables.py --check` validates every front matter
block, and the documentation build runs the same check. `uv run python
scripts/mkdocs_hooks/experiment_tables.py --theme <theme>` prints a theme's Experiments table as
markdown.

## Planned experiments and their order

**Each theme's Research plan page holds its own priority table and the detailed proposals for that
theme.** A planned experiment has a `result.md` with status `planned`, committed before any code
runs. An idea that has no folder yet is marked "proposed" in the priority table and carries a
tentative id, P1, P2, and so on, which the proposal sections use as headings. A proposal receives
its numeric id when its plan is committed, and the tentative id then disappears. The design under
test is on the [staged probabilistic forecast](../roadmap/staged-probabilistic-forecast.md) page.

**Resolution limits.** The Research plan pages flag the experiments whose planned contrasts the
available sample cannot resolve. The sample is 16 Primary substations, 9 generator series, and one
validation year with 12 months, and it resolves gains of roughly 0.3 points of capacity or more.

## Candidates for the next release

**These experiments are flagged `candidate` and are not yet adopted.** The flags are proposals for
the maintainer to confirm. The maintainer reviews the
candidates every one to two weeks and ports the best into a new model release. The [release
log](../live_service/model-releases.md) records which experiments each release adopted.

<!-- candidates-table -->

## What the experiments taught us so far

- **A 14-day error correction put N1 below `manual_heuristic` for Primary demand.** The correction
  lowered the quantile score by 0.99 points on N1, a model frozen at its training end, and the
  0.5-year half-life added nothing to that frozen model (00019).
- **Use the 14-day additive error correction for Primary demand, with power 2 days old, and do not
  yet use it for BSP and GSP.** It scored the same with 2-day-old power as with 3.5-hour-old power,
  a 7-day window and a ratio form were worse, and on 4 BSP and GSP series it did not lower the
  quantile score (00021).
- **A weather product as accurate as ERA5 would lower the Primary demand score by 1.9%.** ERA5
  weather in place of ENS weather lowered the quantile score by 0.10 points, nearly all from
  shortwave radiation and none up to 2 days ahead (00022).
- **Check for calendar memorisation.** When the same valid time repeats across many ENS runs, a
  model with calendar columns can memorise it. Give each experiment a climatology row, and group
  folds by valid time.
- **Choose a row set that covers every weekday.** The 53 Tuesday runs of 00001 and 00002 put every
  case of the first day on a Tuesday or Wednesday, which does not suit demand.
- **Training on ERA5 weather helps solar and wind, but a longer history does not add to it, and for
  demand a long history hurt.** The solar and wind gain appeared even when the ERA5-trained model
  saw only the ENS-era months, so the gain comes from the weather it was trained on and not from the
  years. For demand, training from 2019 was worse than training from 2023, probably because the
  customer mix and embedded generation drifted.
- **Put recent power in the stage 2 calibrator, and calibrate before judging it.** For Primary
  substations, the calibrator with recent-origin inputs beat a stage 1b correction, and a lag-free
  calibrator is worth about 1 point of capacity at short leads (00005).
- **The term and return-to-work columns showed no detectable gain on top of the Christmas column for
  Primary demand.** Over the three validation years, the return and term columns moved the Primary
  error by +0.06 points from 2 to 20 January (99.2% interval -0.17 to +0.30), +0.20 at Easter (-0.19
  to +0.59), and -0.29 over the half terms (-0.68 to +0.07), and Both was not 0.2 points better than
  a calendar shifted by 17 days (00011). The arm with the full history and the COVID indicator
  scored 0.26 points higher on the whole year than the base arm.
- **A change that helps the point forecast of solar or wind need not help the pooled mixture's
  CRPS.** CAMS-trained solar lowered the member-average error by 0.14 points (00012) and the
  mixture's CRPS by 0.009 (00020), and a stage 0 mapping widened the member spread by a quarter. A
  power-curve base margin lowered the wind CRPS by 0.055 (00020).
- **Attach the forecast solar term to the XGBoost model as a base margin, with a weight fitted on
  the training window and zero beyond 10 days.** The base-margin arm lowered the Primary demand
  quantile score by 0.152 points (98.75% interval -0.199 to -0.111) against the best guess `d0` and
  0.256 against `d1l`, where the fixed-coefficient shift of 00046 lowered `d1l` by 0.056, and
  subtracting the forecast solar from a model trained on gross demand was not shown to match the
  best guess (00064).
- **Do not expect a zero start to help the production XGBoost models.** Experiment 78 found that a
  zero base margin lowered its reference model `d0` by 0.071 points, but on the production recipe
  the zero start changed the Primary score by -0.017 (-0.072 to +0.054) and the solar score by
  -0.010 (-0.036 to +0.028), and worsened wind models without a power-curve margin by 0.049 (00080).
- **Refresh the capacity table that feeds the base margin once per season-year.** A table refreshed
  that often captured almost all of the gain from a current capacity (00078).
