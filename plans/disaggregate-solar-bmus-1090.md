# Plan: separating solar output from aggregate balancing mechanism units (issue 1090)

**Question.** Can the solar output of a balancing mechanism unit (BMU) that pools many sites be
separated from its other generation and demand, using CAMS irradiance as an input? The study also
tests the disaggregation ideas in `docs/techniques/` on a problem with public data, before the main
NGED work meets private data.

**Size.** Complex: it publishes a page (a stored result), adds tested modules to `packages/studies`
(no production path, no degradation rule, no asset graph), admits several designs, and spans new
code. Only `studies/`, `packages/studies/`, and `docs/studies/` change. Reviews: this plan had one
Opus review (its findings are folded in below); the scripts get an Opus code review before the final
run; two Opus science reviews, a diff review, and the page reviews follow. The plan had one plan
review, not two, because the study runs autonomously and the science reviews follow the results.

## Stages

**Stage 0: the decision rule.** For each of the 26 aggregate BMUs, measure how well the census's
envelope model describes the output: the share of half-hours with negative output, the output at
night, and the correlation with the sun. Fixed in advance: a BMU is *clean* if under 5% of its
half-hours are negative and its night-time output is under 5% of its 99th-percentile output;
otherwise it is *structured*. Add up the registered Generation Capacity and the envelope estimate of
each group, and report both, because one BMU (`2__HTGPL000`, 912.5 MW registered, 125 MW at the 99th
percentile) holds half of the registered capacity. Rule from the issue: stop at the envelope fit if
most capacity (by either weight) is clean; otherwise run stages 2 and 3 on the structured BMUs. The
five TotalEnergies BMUs are net importers in 77 to 95% of half-hours, so stage 2 is expected to run.

**Stage 1: infer the physical parameters of single sites.** Forward model per site: CAMS hourly
global horizontal irradiance (GHI), shared across the two half-hours by the cosine of the solar
zenith; Erbs decomposition into direct and diffuse; isotropic-sky transposition onto a plane with
tilt and azimuth (or a single-axis tracker); a scale called the effective DC capacity; a clip at the
AC capacity. The effective DC capacity absorbs the CAMS bias, losses, and temperature derating (the
model has no temperature), so it is never compared with a nameplate DC capacity. Fitted by robust
least squares from nine starting orientations, on daytime half-hours (sun above 5 degrees), with the
registered capacity hidden.

- Sites: the 11 single-site BMUs that follow the sun (3 pure PV, 8 hybrid), each scored as pure PV
  or hybrid. A co-located battery can charge from the PV and dent its output.
- Folds: four contiguous three-month blocks (fit on three, score the fourth).
- Time-offset diagnostic: refit with the output shifted by one and two half-hours each way. A fitted
  azimuth is only trustworthy if zero offset wins, because a half-hour error moves azimuth by about
  7.5 degrees.
- Identifiability: the AC capacity is identified only at sites whose output clips. A site is
  *clipping* if its output sits within 2% of its 99th percentile in at least 20 half-hours on days
  with clear-sky midday irradiance. For a non-clipping site the fitted AC capacity is a lower bound
  and is reported as unidentified, not scored. Tilt is treated as unidentified unless the synthetic
  recovery test shows otherwise.
- Synthetic recovery (positive control): simulate output from known parameters at the real sites'
  positions with a *different* generator from the fit (the DISC decomposition, a temperature
  derate, and block-bootstrapped real residuals instead of independent noise), refit, and report the
  error in each parameter.
- No site has a surveyed tilt or azimuth, so recovered tilt and azimuth are diagnostics only. The
  tests are recovered AC capacity against Generation Capacity (with the CfD and REPD capacity as a
  sensitivity at the two sites whose Generation Capacity is doubtful) and out-of-block error.

**Stage 2: separate solar from a synthetic sum of real series (Spoke 1 of the evaluation page).**
The non-solar half is made only from BMUs with no PV behind them: wind farms, gas units, pumped
storage, and a standalone battery. Supplier BMUs net rooftop PV (BSC K3.3.8), so a supplier BMU is
only a labelled extra scenario whose true solar share is unknown. The solar half is the sum of
several real single-site BMUs, scaled to a stated share of the aggregate's peak. Regressors are
solar basis curves (east, south, west, and tracker) from CAMS, with the DC:AC ratio fixed in advance
at the median fitted in stage 1 leaving out the paired site. The primary regressor is the mean of
the CAMS points in the aggregate's GSP group with the solar sites' own points removed; the GB-wide
mean of the 18 grid points is the comparison. The non-solar part is a calendar baseline (three
flexibilities, with `seasonal` the one planned for P3). Solved as bounded linear least squares, the
convex twin of the differentiable fleet node.

- Weather-correlated confounds the sum of two real series leaves out: a pairing with a battery
  BMU, and a pairing with an injected demand term that rises on dull days (proportional to
  one minus the clear-sky index). The solar coefficient's bias is reported on its own, apart from
  the series error.

**Stage 3: apply to the real aggregates** that fail the decision rule. No ground truth: check the
sum constraint, zero at night, the recovered capacity against the envelope estimate and the
registered capacity, and the stage-2 detection limit.

## Planned contrasts (named before any result)

- P1: recovered AC capacity at the clipping single sites, the physical model against the census
  envelope fit (cosine and CAMS shapes) and against the largest and 99th-percentile output: mean
  absolute relative error against Generation Capacity. Interval: resample sites.
- P2: out-of-block error of the free-orientation model against the same model with tilt and
  azimuth fixed at 30 and 180 degrees, and against a tracker: does inferring orientation improve
  held-out power? Interval: the spread across the four folds and resampled sites.
- P3: stage-2 recovery of the solar series (mean absolute error as a share of the solar half's
  capacity) at solar shares of 10%, 25%, and 50% of the aggregate's peak, for the separation with
  CAMS regressors against the envelope-scaled curve. P3b: the same separation with a cosine-of-zenith
  basis instead of CAMS, which answers whether CAMS helps.
- P4: stage-2 negative control: an aggregate with no solar. The recovered solar capacity, as a share
  of the aggregate's peak, must be under 2% (fixed now), or the method's false-positive size is
  reported as it stands.

Every other number is exploratory (baseline flexibility, tracker, DC:AC, supplier-BMU scenarios).

## Files

`packages/studies/src/studies/pv_physics.py`, `pv_fit.py`, `pv_separation.py` with tests in
`packages/studies/tests/`. `pv_physics` calls `pvlib` directly for sun position, as
`studies.solar` does. Scripts in `studies/solar_disaggregation/`. Page
`docs/studies/solar-bmu-disaggregation.md`, nested under the census page in the nav.

## What the study cannot show

No surveyed orientation exists for any BMU. Single-site validation does not test aggregates that mix
technologies or net demand, which is what stage 2 is for, and the synthetic sums of stage 2 are
idealised (Spoke 1 caveat). Stage 3 has no ground truth.

## Added after the first results (post hoc)

The difference separation (weights fitted to four-hour changes), the physical fit to four-hour
changes (`fit_plant_to_changes`), a dull-day demand control, a detection statistic (share of
four-hour change variance explained, with a threshold from no-solar controls), and Gaussian priors
on tilt, azimuth, and DC:AC ratio were added after the first results. The page labels each as post
hoc or exploratory. Stage 2c has no month bootstrap.
