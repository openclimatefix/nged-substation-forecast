# How the Kalman filter corrects the demand forecast

**This page hosts an animated explainer of how a Kalman filter corrects the forecast that a
generalised additive model (GAM) makes of Primary substation demand.** The explainer uses real data
from one Primary substation (anonymised as D11) for one winter week, 4 to 10 February 2026. It shows
no series from a metered generator. Press Play, drag the slider, or use the arrow keys to change
scene. [Open the explainer full-screen](kalman-gam-explainer.html) on a phone, where the frame below
is small.

<!-- pyml disable-num-lines 2 no-inline-html-->
<iframe src="../kalman-gam-explainer.html" title="How the Kalman filter corrects the GAM"
        width="100%" height="420" style="border:1px solid #D5D9E3" loading="lazy"></iframe>

**The explainer has six scenes.**

- Scene 1, "GAM alone": the forecast of the additive model without any correction.
- Scene 2, "The error": observed demand minus the additive model's forecast.
- Scene 3, "Filter state": how each new reading moves the filter's two-part state.
- Scene 4, "Forecast": the corrected forecast at one issue time.
- Scene 5, "Bands": the uncertainty band around the corrected forecast.
- Scene 6, "Not fixed": the error pattern by time of day that the filter leaves behind, and the
  time-of-day terms of stage 2 that shrink the pattern. The pattern shrinks, but the mean absolute
  error barely changes (4.17, 4.12, and 4.16 points).

**The explainer departs in three ways from a single-state Kalman filter whose variance sets the
forecast band.**

- **The state has two parts, and four filters run side by side.** Each filter holds a slow level and
  a fast part that fades with a half-life of 19.1 hours. One filter exists per hour band: night
  (00:00 to 05:59), morning (06:00 to 10:59), midday (11:00 to 15:59), and evening (16:00 to 23:59),
  in local time. Each reading updates both parts of its band's filter, with a gain (the share of the
  surprise that the filter takes on) averaging about 8% for the level and about 42% for the fast
  part over the week. The correction holds the level flat across the lead time and fades only the
  fast part. The filter code is
  [`dp_assimilation.py`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/studies/src/studies/dp_assimilation.py),
  with the noise parameters that [experiment
  133](00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/result.md)
  learned.
- **The band comes from a table of past forecast errors, not from the filter's variance.** The table
  holds quantiles of past forecast errors for each lead-time band, hour band, and calendar quarter
  (`LeadHourSpread` in
  [`dp_ens.py`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/studies/src/studies/dp_ens.py)).
  Neither the filter's variance nor the spread of the weather ensemble enters the band. [Experiment
  169](00169-calibration-combined-fixes-and-ensemble-spread/result.md) found that adding the
  ensemble spread improved nothing. The week shown uses the forecasts stored by [experiment
  136](00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md). The current Primary demand
  forecast uses the smoothed table of [experiment
  160](00160-calibration-band-width-fix-on-stage-2-winner/result.md), widened on forecast-cold days
  by [experiment 164](00164-calibration-cold-night-tail-under-coverage/result.md).
- **The explainer leaves out the neural-network head and the solar summary.** The scenes draw the
  additive model plus the filter, and the time-of-day terms of stage 2 from [experiment
  155](00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md). Primary demand
  research default 3 also holds a small neural-network head and a solar summary, as the [log of
  research default versions](research-default-log.md) lists.

**The week was chosen to show the filter working.** The additive model under-forecast on 99% of the
half-hours of the week, and the filter lowered the mean absolute error from 6.6 to 3.0 points of
capacity at the last of the six issue times. A week where the additive model had no persistent error
would show less.
