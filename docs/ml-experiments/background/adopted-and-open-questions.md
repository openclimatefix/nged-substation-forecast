# Adopted and open questions

**No experiment result is adopted in production, a set of results are candidates for the next model
release, and each open question belongs to one theme.** This page gives one line per item and links
the theme that owns it. A planner checks here before proposing an experiment that a theme already
holds as settled or planned. Evidence tags follow the [update rule](index.md#update-rule).

## Adopted

- **Nothing is adopted.** The release log lists no numbered model release, and no `result.md`
  carries `production_status: adopted` ([model releases](../../live_service/model-releases.md))
  (docs).

## Candidates for the next release

- **Primaries and supply points:** weekly refits, and a five-member blend at Primaries (experiment
  87). Owner: [production reference and best
  guess](../themes/production-reference-and-best-guess/index.md) (docs).
- **Retraining:** weekly refitting without the 14-day correction (experiments 34, 48, 81). It holds
  at 16 Primaries on two sets of runs, but 3 series carry much of the gain. Owner:
  [retraining and training history](../themes/retraining-and-training-history/index.md) (exp 34, 48,
  81; 16 Primaries; provisional).
- **Solar and wind farms:** the pooled mixture over ensemble members, the solar spread factor, and
  the wind power-curve base margin (experiments 2, 16, 20). Owner: [calibration and
  uncertainty](../themes/calibration-and-uncertainty/index.md) (docs).
- **Solar offset:** a forecast of the unmetered solar output as an XGBoost base margin at Primaries
  with a clean capacity estimate (experiments 64, 78). Owner: [solar
  disaggregation](../themes/solar-disaggregation/index.md) (docs).
- **Physics-based demand model:** the convex model with a Kalman filter (experiment 111), a
  recent-error correction recipe (experiment 155), and a correction neural network with a two-number
  solar summary (experiment 220). Owner: [differentiable physics: forecast
  engine](../themes/differentiable-physics-forecast-engine/index.md) (docs).

## Open questions by theme

- **Do the spread factor, the blend, and the solar offset hold on a second validation year?** The
  first year chose their settings. Owner:
  [production reference and best guess](../themes/production-reference-and-best-guess/index.md)
  (docs).
- **Which training source suits solar and wind farms, the ERA5-trained mixture or the ensemble's own
  history?** Needs more than 6 solar and 3 wind farms. Owner: [production reference and best
  guess](../themes/production-reference-and-best-guess/index.md) (docs).
- **Does weekly refitting beat the stack on an unscored year and with the full ensemble, and why do
  older training years score worse?** Owner: [retraining and training
  history](../themes/retraining-and-training-history/index.md) (docs).
- **Do old years of history help the physics-based demand model?** The answer changed sign between
  two scored years. Owner: [differentiable physics: long history and
  drift](../themes/differentiable-physics-long-history-and-drift/index.md) (docs).
- **Does one model pooled across Primaries match a model per substation, and does pooling hold at
  hundreds of series?** Owners: [hierarchy and pooling](../themes/hierarchy-and-pooling/index.md)
  and [differentiable physics: pooling across
  substations](../themes/differentiable-physics-pooling-across-substations/index.md) (docs).
- **Does a calibrator that sees recent power help solar and wind farms, and does a missing-lag flag
  widen the band where a lag is missing?** Owner: [calibration and
  uncertainty](../themes/calibration-and-uncertainty/index.md) (docs).
- **Does lagged power help once the 14-day correction is in place?** No experiment has combined them.
  Owner: [lagged power and error
  correction](../themes/lagged-power-and-error-correction/index.md) (docs).
- **Does the live Met Office UKV feed give wind farms the gain the research archive did?** Owners:
  [generator forecasting](../themes/generator-forecasting/index.md) and [weather products and
  blending](../themes/weather-products-and-blending/index.md) (docs).
- **Does a better sunshine forecast close the gap to the true-sunshine oracle?** Owner: [solar
  disaggregation](../themes/solar-disaggregation/index.md) (docs).
- **Which extra inputs help Primary demand beyond the calendar and solar set?** A heating term gave
  0.14 points in cold hours on one winter, under its 0.3 target. Owner: [demand
  inputs](../themes/demand-inputs/index.md) (docs).
- **Does any forecast curtail less than another once about 2,500 series are scored, and are the
  forecast tails too narrow in general?** Owner: [curtailment
  cost](../themes/curtailment-cost/index.md) (docs).
- **What physical plant reproduces each metered solar farm's output, and how much rooftop solar sits
  behind each substation?** Owners: [metered solar farms](../themes/metered-solar-farms/index.md) and
  [differentiable physics: embedded generation](../themes/differentiable-physics-embedded-generation/index.md)
  (docs).
- **Does the physics-based demand model, driven by ensemble weather, forecast Primary demand up to 7
  days ahead as well as BG2?** Experiment 111 could not tell them apart. Owner: [differentiable
  physics: forecast engine](../themes/differentiable-physics-forecast-engine/index.md) (exp 111; 5
  Primaries, one year; provisional).
- **How should the physics-based demand model combine its daily profile with holiday, heating, and
  solar terms?** Owner:
  [differentiable physics: demand, calendar and
  weather](../themes/differentiable-physics-demand-calendar-and-weather/index.md) (docs).

Last checked against experiment 225.
