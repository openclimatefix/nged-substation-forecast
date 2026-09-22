# Cleaning the trial-area telemetry

> **Status: 🚧 Planned (v0.4).** Epic:
> [#150](https://github.com/openclimatefix/nged-substation-forecast/issues/150). Clean historical
> dataset: [#734](https://github.com/openclimatefix/nged-substation-forecast/issues/734);
> shape-change detection:
> [#553](https://github.com/openclimatefix/nged-substation-forecast/issues/553).

**The faults to clean are catalogued in [NGED's network and its
data](../background/network.md#data-quality-in-the-trial-area); this page holds what has since been
measured about them and what that implies for the cleaning step.** Versions 0.1 to 0.3 train on and
forecast from uncleaned telemetry, so every finding here is currently absorbed by the models rather
than removed.

## A generator's commissioning ramp has to be cut, and one cut-off is already known

**The site labelled E in the [beam/diffuse
results](../studies/beam-diffuse-split.md) produced less than its capacity record says until
6 October 2024, so rows before that date do not describe the plant that exists now.** The site
climbed to its settled output through eight months of discrete steps, each held for days at a time —
the step levels, the evidence and the figure are under [how a new solar farm reaches full output in
stages](../background/network.md#a-new-solar-farm-reaches-full-output-in-stages-over-months). Which
series carries the label is recorded in the private data store rather than here, because this page
is published.

**Use 6 October 2024 as the cut-off, and treat it as a bracket rather than a sharp date.** The step
is fitted from daily output ratios, and the winter that follows it carries a fifth of summer's
usable half-hours, so anywhere between October 2024 and March 2025 fits the data equally well. The
earliest date in that bracket keeps the most rows, and every later date would remove more. A
cleaning step that hard-codes the cut-off should say which bracket it came from, so that a later
re-measurement can move it without anyone having to rediscover why it was there.

**The beam/diffuse experiment removes those rows rather than masking them out of training alone,
and the same reasoning applies here.** A curtailed hour can be kept and scored against the cap the
operator set. Nothing in the record says what fraction of an array was energised on a given day, so
a commissioning hour has no ceiling to compare a prediction against.

## Detecting a ramp needs a reference series, not a threshold

**A part-built solar farm under a clear sky produces exactly what a whole solar farm produces under
cloud, so no threshold on a single series separates them.** Dividing a site's output by the median
output of its neighbours cancels cloud, season and time of day, and leaves the plateaus visible; a
clear-sky irradiance model would do the same job where no neighbour is available. That requirement
is what makes ramp detection a different job from the false-zero and stuck-value detections, which
each look at one series alone.

**This is the same detection problem as [#553](https://github.com/openclimatefix/nged-substation-forecast/issues/553)**,
which asks how to notice a feed that keeps reporting while what it measures changes underneath. A
commissioning ramp is that fault run forwards; a generator switching off behind a monitored
substation is the same fault run backwards. Both are invisible to a freshness check and both show
up as a step in error against a reference.

## The export cap is a mask on the target, not a cleaning rule

**An hour in which the network operator capped a generator's export is a real measurement of a real
export, so it is not dirty data.** It belongs out of a *training* target, because no weather feature
can explain a network instruction — that argument, and the guardrail against feeding the cap to a
model as a feature, are in [drop curtailed hours from the training
target](xgboost-improvements.md#drop-curtailed-hours-from-the-training-target). A cleaning step
that deleted capped hours from the stored telemetry would throw away what actually happened on the
network.

**One generator in the trial area is connected under active network management, and NGED has
confirmed there are no others**, so within the trial area a missing setpoint record means a
generator that ran free. That confirmation does not carry to the wider rollout, where the
population of flexible connections has to be established again.
