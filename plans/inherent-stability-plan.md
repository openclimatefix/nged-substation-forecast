# Inherent stability — design principle and plan

The forecasting service should degrade gracefully and legibly rather than fail, and should
communicate that degradation in the answer itself rather than through a side channel. This document
states the principle, records where the code already implements it and where it does not, defines
the engineering hypotheses it implies, and lists the work to close the gap.

Nothing here is implemented yet. Five decisions are open, in [§7](#7-open-decisions). Delete this
file when the work lands, pasting a summary into the PR body.

**Contents**

1. [The principle](#1-the-principle)
2. [Vocabulary](#2-vocabulary)
3. [Mechanisms](#3-mechanisms)
4. [Where the code stands](#4-where-the-code-stands)
5. [Engineering hypotheses](#5-engineering-hypotheses)
6. [The plan](#6-the-plan)
7. [Open decisions](#7-open-decisions)

---

## 1. The principle

> **We never stop answering. We get less certain instead — and we say so in the answer itself.**

### The incumbent is the floor

`docs/background/nged-incumbent-forecast.md` records how NGED forecasts today: for each substation
and target half-hour, 13 historical analogues at the same time-of-day on the same weekday — 6 from
the last 6 weeks, 7 from 49–55 weeks back. No weather, no ML, no holiday alignment, no load-growth
scaling. An operator reads a forecast off the plot.

Two properties matter to us. It **consumes no NWP**, so an NWP outage does not degrade it — which
makes NWP outage the hard test for us: can our weather-blind fallback beat a method that never used
weather? And it **survives a power-data outage**, because the 49–55-week-back analogues are
unaffected by recent staleness.

So the incumbent already embodies this philosophy, which is why it is the right floor. It also gives
a better failure criterion than any arbitrary staleness threshold:

> **We should only fail when we can no longer beat the incumbent.**

That is measurable. `nged_incumbent` is designed in
`docs/roadmap/metrics-and-leaderboard.md` → "The headline baseline", though not yet implemented
([#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147)).

The consequence, once verified, is the strongest claim in this document:

> **At our worst we degrade to roughly the incumbent. At our best we beat it substantially. There
> is no state in which NGED is worse off than they are today.**

That is currently an intention, not a measured fact — see [§3.2](#32-the-weather-blind-problem).

### The degradation ladder

| Rung | Available inputs | Expected behaviour |
|---|---|---|
| 0 | Everything fresh | Best skill; narrowest bands |
| 1 | One or two daily NWP runs missed | Slightly worse; bands widen slightly |
| 2 | No NWP for days/weeks | Weather-blind: lags, calendar, per-series structure. **Should still beat the incumbent** |
| 3 | No NWP *and* no recent power | Calendar + climatology + year-old history. Converges toward *being* the incumbent |
| 4 | Nothing at all | Physical envelope (clear-sky) + climatology. Very wide bands, still bounded and still true |

Rung 4 matters because it shows there is no input state in which we have nothing true to say:
clear-sky irradiance needs only latitude, longitude and time.

### Missing versus wrong

"Always output" is not "always trust". The distinction:

- **Absent or stale input** → always produce a forecast. Degrade, widen, declare.
- **Detectably wrong input** → do not consume it. Treat it as missing, which routes it back into the
  always-output path.

A stuck meter reporting 2.1 MW for 52 hours is not missing data; it is actively misleading, and a
lag-feature model will propagate it happily. The incumbent has the identical vulnerability.
`docs/background/data-quality.md` already documents both classes empirically — false zeros, stuck
values, and missing data as separate phenomena with real plots — so that page is the evidence base
for this distinction, not something to restate.

### Restoring force and damping

From vehicle and aircraft dynamics: **static stability** is whether a disturbed system's initial
tendency returns it toward correct; **dynamic stability** is whether that return settles rather than
oscillating with growing amplitude. A system can be statically stable and dynamically unstable —
always heading the right way, but hunting and never settling.

So the principle has two halves:

> **Restoring force** — when inputs degrade, the system's normal path already moves toward a
> sensible output.
>
> **Damping** — that correction settles rather than oscillating.

Damping is where bounded retries with backoff, rate limits on retraining, and hysteresis on model
promotion and demotion belong. The data-engineering failures it prevents — retry storms, backfill
cascades, a model flapping between promoted and demoted — are what actually generate on-call pages.
For the operability argument this half matters more than the first.

### What this is not: Postel's law

"Be liberal in what you accept" sounds like this principle but has fallen out of favour because
liberal acceptance is how silent corruption propagates (RFC 9413 on protocol maintenance). Our
stance is sharper:

> **Liberal about missing inputs. Strict about malformed ones.**

The Patito contracts layer is the strict half, and it is what stops inherent stability from decaying
into "accept anything and hope". `docs/architecture/forecast-delivery.md` § "Strict data contracts
(machine-verifiable)" is the anchor.

### The R&D / production asymmetry

This resolves what would otherwise look like a contradiction between "fail loudly" and "never fail
if any data is retrievable". The two contexts have different costs of being wrong:

| | Production | Model R&D |
|---|---|---|
| Cost of no output | High — a user is waiting | Nil — rerun it |
| Cost of a quietly-degraded output | Moderate, **if** flagged in the data | High — silently poisons a model and every comparison built on it |
| Correct posture | Fail-operational: degrade and declare | Fail-fast: refuse to proceed |

> **Fail in the direction where being wrong is cheapest to recover from.**

In production that is forward; in R&D it is backward. This is not the same axis as Dagster's
WARN-versus-ERROR severity: R&D lives in the CV and training assets, so the natural mechanism is a
strict-mode flag on the feature and validation layer, plus the asset tagging in
[#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423).

### Where complexity should live

> **When a capability can be built into the training loop or into the production service, build it
> into the training loop.**

The service runs unattended at 06:00 on the day the inputs are strangest. Training runs in front of
a human who can read the traceback and re-run it. Complexity in the two places therefore carries
very different risk, and the reasoning that puts fail-fast in R&D and fail-operational in production
says the same thing about code placement: keep the serving path as close to "load a model, call
`predict`" as we can.

This is descriptive as much as aspirational. `promoted_model` already copies the champion to local
disk so inference makes no MLflow call. Regime-conditional conformal calibration
([§3.6](#36-neural-nets-and-differentiable-physics)) has the same shape — computed offline,
production does a table lookup.

Two qualifiers keep it honest:

- **It is a tie-break, not an override.** A single model spanning every degradation regime may spend
  capacity on regimes that occur one day in a hundred. Where skill is comparable, prefer the simpler
  service; where the gap is measured and real, skill wins.
- **It relocates the branch rather than removing it.** A fallback cascade's `if` is reviewable and
  directly testable; a model that "handles anything" holds the same branch internally as a learned
  default direction, which can only be measured. So the principle is safe to apply only once the
  failure-scenario suite exists to measure it — which makes items 1–3 a precondition for it rather
  than merely adjacent work.

It does not license unbounded training complexity: H2 and H3 depend on retraining staying cheap and
promotion staying one-click, so a training harness nobody can run is also a production risk. And it
has no bearing on work that *must* live in production — items 13 and 14 among them.

### Naming, and the analogy to use

**Use "inherent stability", and lead with the steering analogy.** Avoid nuclear fission: in a
reactor *safe means off*, so a reader told the system is "passively safe like a reactor" will
reasonably infer it fails to a stopped, cold state — the opposite of the thesis. Avoid "passive
safety" as a term, because in the automotive industry it already means crashworthiness (seatbelts,
airbags, crumple zones), and self-centring steering is filed under inherent stability, not passive
safety.

Steering points the right way. Front-end geometry tilts the steering axis back so the tyre contact
patch trails behind where that axis meets the ground, like a shopping-trolley caster; any sideways
force generates a torque swinging the wheel back into line. Let go of the bars and the bike keeps
going, correctly, without an operator. Cars add kingpin inclination and pneumatic trail, and
together these give the **self-aligning torque** — which is also the driver's feedback channel: as
the front tyres approach the grip limit, pneumatic trail collapses and the steering goes light. The
car reports its own degradation through the same mechanism that keeps it centred. That is exactly
what widening confidence bands do for us.

---

## 2. Vocabulary

| Term | Origin | Fit |
|---|---|---|
| **Graceful degradation** | General | The plain-English name: lose fidelity, not availability |
| **Static stability** | AWS Builders' Library, borrowed from vehicle dynamics | A system keeps working using state it *already has* when dependencies fail. Exactly `live_forecasts` reading NWP already on disk |
| **Fail-operational vs fail-passive** | Avionics autoland; ISO 26262 | The sharpest available distinction. Fail-passive: disengage cleanly, hand back to the human. Fail-operational: keep delivering through the fault. Ours is fail-operational |
| **Blast radius** | SRE | A **different axis** — *how much* fails, not *which way*. Our partitioning is a blast-radius story; keep the two separate in the prose |
| **Bulkhead / circuit breaker** | Nygard, *Release It!* | The canonical stability-patterns reference |
| **Stale-while-revalidate** | RFC 5861 | Serve stale while refreshing behind the scenes — the NWP fallback's cousin |
| **Write-Audit-Publish** | Data engineering | The **opposite stance**: validate, then block publication on failure. Fail-closed. We are deliberately fail-open on freshness |
| **Postel's law** | RFC 761 | Named only to disown |

Dagster supplies first-class vocabulary for the fail-open choice — `AssetCheckSeverity.WARN` versus
`ERROR`, and `blocking=True/False` — so the docs can be concrete rather than philosophical.

---

## 3. Mechanisms

### 3.1 XGBoost's NaN handling, and its limit

XGBoost uses sparsity-aware split finding (Chen & Guestrin). At every split it learns not just a
threshold but a **default direction** for rows where that feature is missing: it tries sending all
missing rows left, then right, and keeps whichever gives better gain. Missingness is routed, not
imputed. This is why `_nullify_leaky_lags` can null a feature rather than drop a row.

The limit is that the default direction is learned **from the missingness present in the training
data**. If a feature is never missing during training, XGBoost still picks a direction, but that
choice was never evaluated against any data. In production, the first time that feature is missing,
every affected row takes an untested path. So the guarantee is narrower than "XGBoost handles NaN":

> **XGBoost handles missingness patterns it saw during training.**

For a failure like "the whole ECMWF run is absent" — which may never occur in a clean training set
— we have no evidence about behaviour at all.

### 3.2 The weather-blind problem

The claim "with no NWP for weeks we still beat the incumbent" is not delivered by NaN routing. A
model trained *with* NWP features and run *without* them falls back on arbitrary default directions,
not on a well-trained weather-blind model. Two ways to make the claim true:

| Option | Mechanism | Verdict |
|---|---|---|
| **A. Train for it** | Include outage-shaped scenarios in training so the default directions are learned | **The default.** One model, nothing added to the serving path, genuinely inherent |
| **B. Fallback cascade** | Keep a cheap weather-blind model trained alongside; use it when NWP is absent | A branch, but a deterministic function of what data exists — no detection step, so defensible if it is needed |

Try A, keep B in reserve, and let the failure-scenario suite decide — but §1's where-complexity-lives
principle sets the burden of proof. A is the default; B has to earn its place in production with a
measured skill gap, not merely by being easier to implement. That makes the choice empirical rather
than aesthetic.

### 3.3 Widening bands: the in-band signal

A stale forecast **looks identical to a fresh one**. Staleness columns and warning tables are side
channels: they require the consumer to go and look. Uncertainty bands are not. A forecast whose
P5–P95 spread has doubled has already told the consumer to be more cautious, through the only number
they were going to read anyway. The machine reports its own degradation through the same mechanism
it uses to do its job, so **no separate monitoring system has to work for the safety property to
hold**.

Two caveats. `XGBoostConfig.objective` currently defaults to `reg:squarederror`, so today's model is
a point forecast; quantile output
([#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263)) is a prerequisite.
And bands widening *correctly* under degradation is not automatic even with quantile regression —
see [§3.6](#36-neural-nets-and-differentiable-physics).

### 3.4 The physical envelope

Clear-sky irradiance needs only latitude, longitude and time, so as data degrades the forecast can
relax toward a physical envelope that is always computable — wide, honest, and still bounded by what
the sky can deliver.

Clear-sky irradiance is already a named deliverable of
[#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168), designed in
`docs/roadmap/xgboost-improvements.md` → "Linearised physics features for solar and wind", with
further physics in `docs/techniques/differentiable-physics.md`. Only the **floor** framing is new:
nothing currently says clear-sky is what we fall back to when we have nothing.

### 3.5 Three audiences, three channels

| Audience | Question | Channel |
|---|---|---|
| **Forecast users** (NGED) | "How much should I trust *this row*?" | In-band: quantile spread + `nwp_init_time` (already on the row) |
| **Data providers** | "Is *your* feed broken, and since when?" | Aggregated and **attributable**: `power_forecast_warnings`, the freshness check's late-series table |
| **Us, the developers** | "Is *our* system at fault?" | Out-of-band: Sentry, plus the missed-check-in alarm |

Inherent stability creates a specific hazard for the third channel: **a system that always succeeds
looks identical to a system that is not running at all.** Both produce zero failures. That is why
the Sentry missed-check-in alarm, firing from outside the deployment, is load-bearing rather than
belt-and-braces — it is the one piece of active monitoring the design cannot do without.

For the provider channel, a warning is only actionable if it names *whose* NWP and *which* run,
which implies a `warning_source` field on `power_forecast_warnings`.

It must also count the right thing. We ingest **one ECMWF run per day** (00Z, downloaded at 08:30
UTC by `ecmwf_ens_schedule`) and forecast at 00/06/12/18, so healthy NWP age at forecast time ranges
from 12 h at the 12:00 slot to **30 h at the 06:00 slot**, just before the day's download lands. Raw
age is therefore not a health signal — 18-hour-old NWP is exactly what the 18:00 slot is supposed to
use, and any absolute threshold low enough to catch a real outage would fire on two of the four
slots every day. The signal is **missed runs**: how many daily runs are absent between the freshest
run on disk and the freshest that should exist by now. That is zero in every healthy slot, whichever
slot it is. The degradation ladder's rungs are counted the same way.

### 3.6 Neural nets and differentiable physics

**Do not zero-fill; remove from the set.** Zero is a meaningful value in physical units — 0 MW,
0 W/m², 0 °C are all real states — so replacing a missing value with zero asserts something false.
Treat inputs as a **set of tokens** (value embedding + feature-identity embedding + time embedding)
and simply do not emit tokens for absent inputs. Attention is natively permutation-invariant and
variable-length, so a missing input is structurally absent rather than encoded as a sentinel. Mask
the attention matrix for padding only.

The dense alternative is **value + mask channels**, so the network can distinguish "zero" from
"unknown". GRU-D is the precedent: irregularly-sampled clinical time series with heavy missingness,
using masks plus learned decay of the last observation toward an empirical mean.

**Our missingness comes in two kinds, and only one of them needs scenarios.**

*Chronic and fine-grained.* Three de-accumulated variables — `precipitation_surface`,
`downward_short_wave_radiation_flux_surface`, `downward_long_wave_radiation_flux_surface` — are
legitimately null at lead-0 in **every** run, and beyond lead-0 carry *scattered per-pixel* nulls
rooted in **corrupt ECMWF source accumulation**: some fields report physically-impossible negative
accumulation, which Dynamical's de-accumulation step correctly surfaces as null rather than
clamping corrupt data to zero
([dynamical-org/reformatters#722](https://github.com/dynamical-org/reformatters/issues/722), WONTFIX
upstream — a looser clamp would only convert visibly-null corrupt data into invisibly-zeroed corrupt
data). Empirically a few percent of a slice; see `contracts/weather_schemas.py:231` and
`docs/architecture/ecmwf-ens-known-issues.md`. This is element-wise, not blocky. But it is present in **every** training run, so it is in-distribution:
per §3.1 this is the one case where "XGBoost handles the missingness it saw during training"
genuinely holds. It needs no scenario, and the main risk is that someone later "fixes" it by
imputing.

*Episodic and coarse-grained.* Missed or stale runs, a wholesale-absent variable, a telemetry
stall. These are rare or wholly absent from training data, which is exactly why they must be
enumerated — and here the combinatorics stay tractable: NWP {fresh, *n* runs missed, absent} ×
telemetry {present, partial, absent} × metadata is on the order of ten to twenty realistic regimes,
not 2ⁿ. So structured, outage-shaped dropout is feasible and matches reality far better than
element-wise random dropout.

The ingest gate already keeps the two apart. A *whole-slice* null in a de-accumulated variable is
fatal in `Nwp.validate`, so wholesale corruption never lands as silently-broken data — it manifests
downstream as a **missed run**, which is rung 1 of the ladder. Fine-grained catastrophic corruption
is converted into coarse-grained absence, the form the rest of this design already handles.

**Differentiable physics is the strongest piece.** A physical model has a defined output for any
input state; where an input is absent, substitute a climatological prior or a physical bound and let
the physics propagate it. Physics gives the envelope, the learned residual sharpens it, and as data
degrades the residual head has less to work with so the answer relaxes toward the prior. No
branching, no fallback logic, no `if data_is_missing:` — the same code path does the right thing
because of how it is arranged.

It is also the **nearest**. Differentiable physics arrives at **v0.7** as Candidate B in the
capacity-estimation head-to-head
([#141](https://github.com/openclimatefix/nged-substation-forecast/issues/141),
[#157](https://github.com/openclimatefix/nged-substation-forecast/issues/157),
[#158](https://github.com/openclimatefix/nged-substation-forecast/issues/158)), two milestones
ahead of the v0.9 neural-net spike. So none of this is v2 work: the first model family we must
*build* for missingness is a DP estimator, and the requirement has to reach its design pages now
rather than when we get there.

**For honest bands, use conformal prediction per regime.** Split/Mondrian conformal calibration is
post-hoc: calibrate interval widths separately for each degradation regime using held-out residuals
from that regime, giving finite-sample coverage guarantees per regime with no retraining. It works
with **XGBoost today**, so widening bands can ship before any PyTorch work exists, and it makes "the
spread widens when inputs are missing" a measured fact rather than a hope.

### 3.7 Input dropout for regression

Standard dropout is a regulariser, and its effect is not a classification-versus-regression
question. It is used in regression networks routinely — MC dropout for regression uncertainty (Gal &
Ghahramani) is a whole line of work, and Wager, Wang & Liang showed input-layer dropout is a form of
adaptive regularisation. The two real problems are different, and both apply here:

1. **Dropout-to-zero without a mask corrupts the target.** If the network cannot distinguish
   "dropped" from "genuinely zero", it learns `E[y | x with random zeros]`, biasing the conditional
   mean. With an explicit mask channel or token removal, dropout training is well-posed. This is a
   representation bug, not a law about regression.
2. **Dropout trains for MCAR; production missingness is not MCAR.** In Rubin's taxonomy a value is
   **MCAR** (*missing completely at random*) when its absence is independent of everything, observed
   or not; **MAR** (*missing at random*) when the absence depends only on data we can see; and
   **MNAR** (*missing not at random*) when it depends on the missing value itself. Random dropout
   simulates MCAR. Our outages are at best MAR — they correlate with time of day, weather systems
   and provider incidents, all observable — and some are MNAR: a meter that drops out during the
   storm that caused the extreme reading is missing *because* the value was extreme. MNAR is the
   dangerous case, since the missing values are then systematically unlike the observed ones. Either
   way, a model trained on random dropout is calibrated for a world it does not live in, and the
   miscalibration surfaces as over-confident bands *during a real outage* — the worst possible time.

So do not lean on MCAR dropout as the primary mechanism. Use architecture (token removal, masks,
physics fallback) for capability under missingness, structured outage-shaped dropout for learned
behaviour, and regime-conditional conformal calibration for honest widths. Only the last is strictly
necessary for the bands to be trustworthy.

---

## 4. Where the code stands

Audited 2026-08-04 across `defs/checks.py`, `defs/assets.py`, `defs/production_assets.py`,
`defs/schedules.py`, `definitions.py` and `_sentry.py`.

### Already in line with the principle

| Piece | Why it matters |
|---|---|
| `promoted_model` copies the champion to **local disk; inference makes no MLflow call** (`production_assets.py:120`) | The whole experiment-tracking stack can be down and 06:00 inference is unaffected. The best single production-stability example |
| `live_forecasts` selects "the freshest NWP run present as of `power_fcst_init_time`" — a **relative** query, not "today's run" | A missed download degrades to an older run through the normal path. No fallback branch exists to get wrong |
| `power_data_is_fresh` — `blocking=False`, `severity=WARN`, reading **on-disk data recency** not materialisation time | The canonical warn-don't-block pattern; its docstring explains why materialisation-freshness would miss the real failure |
| `nwp_has_no_unexpected_nulls` — in-asset WARN from the frame already in memory (`assets.py:199`, `:275`) | Same pattern, without a second read |
| **No ERROR-severity asset check exists anywhere in the repo** | The fail-open posture is applied consistently, with inline comments giving the reasoning |
| Sentry's log-to-event capture **deliberately disabled** — `LoggingIntegration(event_level=None)` | Subtle and correct: otherwise every `ERROR` log becomes an event, so a fail-open design would flood Sentry with events for conditions it deliberately tolerates |
| The failure hook is attached to the **three scheduled jobs only** (`schedules.py`) | The R&D/production asymmetry, already implemented at the telemetry layer |
| Three distinct Sentry channels — exceptions, freshness (`capture_message`, `level="warning"`), and a **success-only** heartbeat skipped on replay | Maps cleanly onto §3.5; a replay backfill is not evidence the service is alive now |
| `report_power_freshness` **never raises** | Its docstring recognises that a bug in the *warning* path would trip the failure hook and silently convert fail-open into fail-closed |
| `nwp_init_time` on `PowerForecast` (`power_schemas.py:245`) | Provenance travels in-band with every row |
| All five `raise` sites in `cv_assets.py` are in the training and metrics path | R&D fails fast, as it should |
| Daily and 6-hourly partitioning; Patito contracts at every boundary | Blast radius, and the strict half of §1 |
| `METRIC_NAMES` covers `mae`, `nmae`, `rmse`, `mbe`, `crps`, `spread_skill_ratio`, `pinball_loss`, `mean_pinball_loss`, `picp`, `interval_width` | T1.3 needs no new metric, only a scenario dimension to slice by |

### Divergences, all in `live_forecasts`

1. **NWP more than ~15 days stale causes a hard failure.** A weeks-old run no longer covers the
   forecast window, so its rows join to a null `ensemble_member`, are filtered out at
   `production_assets.py:280`, and the asset raises on `forecasts.height == 0` at `:286`. This is
   rung 2 of the ladder, and NGED gets nothing at all.
2. **NWP between 0 and ~15 days stale degrades silently.** The forecast is built from an
   increasingly ancient run, with `nwp_init_time` recorded on the row but no warning anywhere — no
   check, no Sentry event, no widened bands. Arguably worse than the hard failure, because it is
   undetectable from the consumer side without deriving staleness by hand.
3. **`select_nwp_init_time` raises when nothing qualifies** (`_production_helpers.py:71`). Narrower
   in practice, since it fires only when the NWP table holds no run at or before the cutoff, but it
   is the same class and it bites `replay` backfills older than the retention window.

The raise on empty `trained_ids` (`production_assets.py:241`) is **correctly fail-fast** and should
stay: an empty model is a promotion bug, not a data outage. The codebase already distinguishes the
two cases, just not consistently.

**Ordering constraint:** fixing divergence 1 without fixing 2 would convert a loud failure into a
silent one, so the `live_forecasts` check must land first.

### Designed but untracked

- **`power_forecast_warnings` and `asset_health_history`** — two of the five contractual v1.0
  delivery tables, fully designed in `docs/roadmap/delivery-tables.md`, with **no GitHub issue**.
  The first is the user-facing half of this whole principle.
- **Degradation-responsive bands.**
  [#262](https://github.com/openclimatefix/nged-substation-forecast/issues/262)–[#264](https://github.com/openclimatefix/nged-substation-forecast/issues/264)
  turn an *ensemble* into percentiles; nothing makes the spread respond to stale or missing inputs.
- **Failure-scenario evaluation** — absent from docs and GitHub alike, so every v0.5 XGBoost
  experiment would pick a champion on clean-data skill alone.
- **An intervention log** — see [§5](#5-engineering-hypotheses).

---

## 5. Engineering hypotheses

All three headline claims already exist as prose in `docs/background/requirements.md`: "Uptime:
lenient by design" (L142), "ML experimentation at scale" (L76), and "A short, safe path from R&D to
production" (L94). What they lack is a threshold and a measurement. So the hypotheses page extracts
and elevates that prose, linking to it rather than restating it.

One gap the framing exposes: "Uptime: lenient by design" is a *defensive* argument — outages do not
cause much damage. H1 is a stronger *positive* claim: interventions will be rare. The second is the
one actually in dispute, and a sceptic is not moved by "it's fine when it breaks".

### Why hypotheses rather than aims

NIA funding is for transferable learning, including negative results, and six report issues
([#128](https://github.com/openclimatefix/nged-substation-forecast/issues/128),
[#130](https://github.com/openclimatefix/nged-substation-forecast/issues/130),
[#131](https://github.com/openclimatefix/nged-substation-forecast/issues/131),
[#132](https://github.com/openclimatefix/nged-substation-forecast/issues/132),
[#135](https://github.com/openclimatefix/nged-substation-forecast/issues/135),
[#156](https://github.com/openclimatefix/nged-substation-forecast/issues/156)) are natural
consumers. It also converts a disagreement about devops burden into something that resolves itself,
which is a better outcome than a document arguing one side — and pre-registering a number signals
confidence in a way prose cannot. Finally, it forces the measurement artifacts to exist in advance.

The commitment this entails: a hypothesis without a number is an aim wearing a lab coat. Each needs
a threshold and a window, and we must be willing to record a falsification.

### The claims

**H*n*** is a hypothesis; **T*n.m*** is the *m*th **test** of hypothesis *n*. H1 bundles three
separable claims that each need their own measurement, so its row is a header and the three tests
beneath it carry the thresholds. H2 and H3 make a single claim each, so their test sits inline and
needs no number. Labels are citable from issues and reports, so **append, never renumber**.

| | Claim | Test | Threshold | Source | Resolvable |
|---|---|---|---|---|---|
| **H1** | Manual attention only for upstream format changes; graceful, legible degradation; faithful uncertainty | | | | |
| T1.1 | *Operability* | Interventions per quarter, classified by cause | ≥90% attributable to upstream format or contract change; zero out-of-hours | Intervention log | ~2 quarters of v1.0 |
| T1.2 | *Graceful degradation* | Forecast emitted for every series under every failure scenario, **and still beats `nged_incumbent`** | 100% emitted; beats incumbent at rungs 0–2 | Failure-scenario suite | v0.3, after [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) |
| T1.3 | *Faithful uncertainty* | PICP and pinball computed **per degradation regime** | PICP within tolerance of nominal in every regime | Leaderboard, scenario dimension | v0.5 |
| **H2** | Hundreds of experiments per month | Registered leaderboard experiments per month, **and** median human-minutes each | ≥200/month; ≤5 human-min each | MLflow + timestamps | v0.5 |
| **H3** | One-click promotion of the winner | Commands from "leaderboard says X won" to "X is serving" — **and the same for rollback** | ≤1 each way | Runbook + `promoted_model` | v0.3 |

T1.3 is the sharpest test available and needs no new metric — only the scenario dimension, which is
the same machinery that serves T1.2. H3 must include rollback: one-click promotion without one-click
demotion is not safe, and it is the damping half of §1. H2's real claim is its second number, since
a 200-config sweep trivially clears "hundreds per month"; the transferable claim is throughput of
*decision-grade* experiments with negligible human time, and it is worth pairing with
cost-per-experiment.

### The intervention log is time-critical

H1 is the only hypothesis that **cannot be measured retrospectively**. Experiment counts live in
MLflow forever and promotion steps can be counted at any time, but "how many times did a human have
to intervene, and why?" is unrecoverable unless recorded as it happens — and the service is already
running on AWS.

The artifact is cheap: an append-only log with date, trigger, cause category, human-minutes, and
whether a runbook existed. The cause taxonomy is the point, since H1 predicts that essentially all
entries fall into "upstream format or contract change". **This should ship before the docs pages.**

### Candidate further hypotheses

Proposals only; six is the sensible ceiling, because each carries a measurement cost.

- **H4 (cost)** — the service runs under £X/month at v1 scale and £Y at v2.
  `docs/architecture/aws-costs.md` already estimates **~£25–35/month for the whole v1 stack** at 32
  time series. The most transferable NIA finding of the lot, and a second independent answer to the
  devops-team worry.
- **H5 (operability by a non-expert)** — an NGED operator can run the service from runbooks alone.
  Already designed as the operator contract in `docs/roadmap/handover.md`; framing it as a
  hypothesis turns the game days into a measurement rather than a training exercise.
- **H6 (scale without redesign)** — the architecture goes from 32 to ~2,500 series without
  structural change. The central engineering bet of the project, currently unstated anywhere, and
  only resolvable at v2 — which is an argument for writing it down now.

---

## 6. The plan

### 6.1 New pages

**`docs/architecture/inherent-stability.md`**, in the nav directly after `Overview`. It sits in
`architecture/` because it is durable design rationale, and because CLAUDE.md permits code to link
to `docs/architecture/`, so docstrings can point at it without rotting.

Structure, serving the three audiences in sequence:

1. The principle in plain language, the incumbent-as-floor statement, and the steering analogy.
2. A **failure-mode table** — what breaks, what the system does, whether a human is paged.
3. The **rules**, numbered and imperative. This is what future Claude sessions will act on, and a
   numbered list is far better instruction than prose.
4. Mechanisms, condensed from §3.
5. What this deliberately is not: Postel's law, and the R&D/production asymmetry.

The page **spans the architecture/roadmap line**, because the philosophy is partly built and partly
not. The resolution, which the page should state: the principle and the built mechanisms live here;
unbuilt mechanisms stay in their roadmap pages and are **linked, not copied**. Otherwise the page
becomes a roadmap mirror and rots.

**`docs/engineering-hypotheses.md`**, in the nav immediately after "Documentation Guide" — above
Background, because these are our claims, not NGED's context.

The three-page division of labour, with no duplication:

| Page | Answers |
|---|---|
| `background/requirements.md` | *Why* we need this (NGED-derived). Existing prose stays put |
| `engineering-hypotheses.md` | *What we claim*, how it is tested, what would falsify it |
| `architecture/inherent-stability.md` | *How the design delivers H1* |

**`docs/techniques/conformal-prediction.md` — deferred, written when item 8 reaches the roadmap.**
Conformal prediction currently appears **nowhere** in `docs/`, so item 8 would introduce an
unexplained term into a roadmap page. The techniques section exists for exactly this — a durable
explainer of a solution method — so item 8 must not land without one, plus its bullet in
`techniques/index.md`. Three things the page must cover beyond a generic tutorial:

- **The guarantee and what it costs.** Distribution-free, finite-sample coverage computed from
  held-out residuals, with no retraining and no distributional assumption about the model.
- **Why *Mondrian*, not plain split conformal.** The guarantee rests on exchangeability between
  calibration and test data, and a degradation regime is precisely a violation of it. Conditioning
  on the regime restores exchangeability *within* each regime, which is why §3.6 says "per regime"
  rather than simply "conformal". This is the project-specific point, and the one a generic
  explainer would miss.
- **The precondition and its limit.** Each regime needs enough held-out residuals of its own, and
  the rarest regimes — a weeks-long NWP outage — may have too few. That is a real constraint on
  item 8, not a footnote.

It also needs cross-links to `probabilistic-forecasting.md` (which produces the intervals conformal
calibrates) and `evaluation-metrics.md` (PICP and interval width, which measure whether it worked).

### 6.2 Edits to existing pages

| Page | Edit |
|---|---|
| `documentation-guide.md` | **Both new pages need a home in its four-place model.** `engineering-hypotheses.md` fits none of the four buckets — it is neither unbuilt design, nor an explanation of built code, nor NGED-derived background, but a fifth kind of content: our own falsifiable claims. It needs a new row in both of the guide's tables |
| `live_service/operations.md` | **Two missing runbooks.** The page contains no occurrence of "stale", "outage", or "missing", and offers no way back from a promotion. Both gaps block the H1 and H3 measurements |
| `background/requirements.md` | Cross-links to the hypotheses page, **plus a fourth bullet under "Uptime: lenient by design"**: its three bounded-damage arguments all assume the outage is *our compute stopping*. An extended NWP outage keeps compute running while forecasts hard-fail, and the last good forecast ages out too, so the 14-day-horizon argument does not cover it |
| `architecture/overview.md` | Cross-link. It is currently billed as the design-philosophy page, so without one the two pages compete for the same job |
| `background/data-quality.md` | A pointer that false zeros and stuck values are *wrong* while missing data is *missing*, and the two are handled differently. The evidence already lives there |
| `architecture/production-deployment.md` | An upward link to the principle. **Do not restate** — it already carries the WARN-not-failure reasoning and the inside/outside monitoring complementarity almost verbatim |
| `roadmap/delivery-tables.md` | Degradation-conditional band widening under Table 1; `warning_source` on Table 2 |
| `roadmap/metrics-and-leaderboard.md` | The failure-scenario suite, how it is scored, and the incumbent as acceptance criterion |
| `roadmap/engineering-health.md` | Degradation smoke-tests in the scientific-rigor section |
| `roadmap/xgboost-improvements.md` | The NaN default-direction limit (§3.1) |
| `roadmap/capacity-estimation.md` | **Missingness robustness as a head-to-head judging criterion**, under "What every candidate must get right". Both candidates ingest metered generation that really does have gaps, and the winner's capacity estimate feeds v1.0 forecasting, so an estimator that mis-estimates under an outage propagates downstream. The page already reasons about one kind of absent data — "Identifiability: the data goes silent at night" — so this extends existing reasoning rather than importing a new concern |
| `techniques/differentiable-physics.md` | The mechanism from §3.6: a physical forward model has a defined output for any input state, so substitute a climatological prior or a physical bound for an absent input and let the physics propagate it. The page currently says **nothing** about missing data, yet it is the durable home for the strongest missingness story we have |
| `techniques/encoders.md` | Token removal, not zero-fill (§3.6) — zero is a real physical value, so a zero-filled encoder input asserts something false |
| `architecture/testing.md` | The degradation smoke-tests, under "notable test suites", once built |
| `background/nged-incumbent-forecast.md` | A note that the incumbent is our degradation floor |
| `docs/index.md` | Two entries in the Documentation list; one sentence in "More than a forecast" |
| `roadmap/index.md` | Milestone bullets for the new issues, plus a missingness clause on the v0.7 capacity bullet and the v0.9 neural-net bullet |
| `CLAUDE.md` | A short entry under Architecture. A docs page alone will not reliably reach future Claude sessions; CLAUDE.md is what is always in context |
| `architecture/why-dagster-not-airflow.md` | Three edits — see §6.5 |

### 6.3 Issues

The sequencing argument is that **the evaluation machinery must precede the model experiments it is
meant to judge.**

"Milestone" below means the roadmap milestone and parent epic, not a GitHub milestone field: the
repo has **zero** GitHub milestones, and ordering lives in epic sub-issue lists and the OCF project
board. Each row means "attach as a sub-issue of that epic, positioned by execution order".

| # | Issue | Milestone | Rationale |
|---|---|---|---|
| 0 | **Intervention log** — artifact, cause taxonomy, runbook line | **now, ahead of everything** | Evidence is being lost daily |
| 1 | Degradation smoke-tests: ablate input groups; assert output exists, stays in physical bounds, does not explode | **v0.2** | Cheap, CI-fast, no MLflow. Sibling of [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229) |
| 2 | Canonical failure-scenario suite — named, versioned degradation transforms over `AllFeatures` | **v0.3** | Shared by tests, leaderboard, and later training. Must exist before v0.5. Enumerate only §3.6's *episodic* class; the chronic de-accumulated scatter is in-distribution and needs no scenario, though an **elevated** scatter fraction is a good candidate — `assess_nwp_quality` already computes it |
| 3 | Score every leaderboard experiment under each scenario, **against `nged_incumbent`** | **v0.3**, after [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) | Otherwise v0.5 picks a champion blind to degradation behaviour |
| 4 | `power_forecast_warnings` **Phase 1**: `STALE NWP` + `STALE POWER`, with `warning_source` | **v0.3** | No dependency on v0.4/v0.6/v0.7. The user-facing half, buildable now |
| 5 | Rollback path for `promoted_model` | **v0.3** | H3's second direction |
| 6 | `power_forecast_warnings` **Phase 2**: meter-error types | **v0.4** | Depends on improved cleaning |
| 7 | `asset_health_history` table | **v0.4** | Same dependency |
| 8 | Degradation-conditional interval calibration — conformal per regime | **v0.5** | Directly after #263/#264. Must ship with the `techniques/conformal-prediction.md` explainer (§6.1) — the term appears nowhere in `docs/` today |
| 9 | Clear-sky as the zero-data **floor** — extend [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168) | **v0.3** for the shared primitive; feature use stays with #168 | #168 already delivers clear-sky irradiance. Only the floor framing is new, and the scenario suite needs something to degrade *to* |
| 10 | Cost-per-experiment instrumentation | **v0.5** | Piggybacks on the aws-costs machinery |
| 11 | Weather-blind guarantee: outage-shaped training augmentation (§3.2 option A) | **v0.5** | "Never worse than the incumbent" depends on it |
| 12 | Missingness contract on `BaseForecaster` | **v0.9**, note on [#362](https://github.com/openclimatefix/nged-substation-forecast/issues/362) | Forces the NN spike to answer the question rather than discover it in v2. It binds `BaseForecaster` implementers only — the v0.7 DP estimators are not forecasters, so they are covered by the capacity-estimation judging criteria instead |
| 13 | Extend [#424](https://github.com/openclimatefix/nged-substation-forecast/issues/424) — the `live_forecasts` check — to report **missed NWP runs at forecast time** (not raw age, per §3.5), WARN and non-blocking | **v0.2**, where #424 already sits | Every production asset has a check except the one NGED consumes. #424 needs the degradation dimension and a severity decision |
| 14 | Make `live_forecasts` **degrade rather than raise** when NWP is absent or out of coverage; keep the `trained_ids` raise | **v0.3**, after 13 | §4 divergences. Ordering is load-bearing |
| 15 | Runbook: degraded input data — NWP dark, telemetry stalled, reading the freshness check | **v0.3** | H1's "recovery next business day, via runbook" threshold is unmeasurable without it |
| 16 | Runbook + mechanism: roll back a promoted model | **v0.3** | The docs half of item 5 |

Remaining v0.6/v0.7 warning types attach as sub-tasks of the existing epics rather than new issues.
[#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423) gains a note that the
R&D/production tag is the mechanism behind the fail-fast/fail-forward asymmetry, and
[#141](https://github.com/openclimatefix/nged-substation-forecast/issues/141),
[#157](https://github.com/openclimatefix/nged-substation-forecast/issues/157) and
[#158](https://github.com/openclimatefix/nged-substation-forecast/issues/158) gain a note that
missingness robustness is a judging criterion, scored against item 2's scenario vocabulary.

**Existing issues to attach to rather than duplicate:**

| Issue | Relationship |
|---|---|
| [#424](https://github.com/openclimatefix/nged-substation-forecast/issues/424) | *Is* item 13. Currently `enhancement` only, no epic, no project fields |
| [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168) | Already delivers clear-sky irradiance; item 9 extends it |
| [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) | `nged_incumbent` baseline — **blocks T1.2** |
| [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161) | NWP ingestion validation — item 13's sibling on the ingest side |
| [#420](https://github.com/openclimatefix/nged-substation-forecast/issues/420) | Silencing warnings for dead series — warning fatigue, downstream of items 4/6/7 |
| [#374](https://github.com/openclimatefix/nged-substation-forecast/issues/374) | "Add more data-validation functions" — **empty body**; the natural home for the missing-versus-wrong distinction |

A cross-cutting **`inherent-stability` label** across all of these gives a single legible workstream
view (`gh issue list --label inherent-stability`) without a new epic, which matters because the work
spans v0.2–v0.9 while CLAUDE.md requires epics to map 1:1 to milestones.

### 6.4 Effort and sequencing

The expensive items are not deferred by choice; they are blocked by the v0.5 quantile pipeline. The
items that must be early are the cheap ones. So there is no real trade-off between "groundwork now"
and "implementation later".

| Item | Size | Why |
|---|---|---|
| 0. Intervention log | **XS** | A file, a taxonomy, one runbook line |
| 1. Degradation smoke-tests | **S** | Trivial once item 2 exists; pure functions, no MLflow |
| 2. Failure-scenario suite | **S** | Pure transforms over an `AllFeatures` frame. The vocabulary design is the work; the code is small |
| 3. Leaderboard scenario scoring | **M** | Costs N× *predict* + N× metrics, **not** N× train — you train once and predict per scenario |
| 4. Warnings Phase 1 | **M** | New Delta table and writer, but only two warning types |
| 5. `promoted_model` rollback | **S** | The forward path exists; mostly config and runbook |
| 9. Clear-sky primitive | **M** | Well-trodden; also useful as a feature, so it earns its keep twice |
| 13. `live_forecasts` check | **S** | Reuses the `_to_asset_check_result` shape established twice already |
| 14. `live_forecasts` degrades | **S–M** | Removing the raises is trivial; deciding *what* a no-NWP forecast contains is the real work, and it overlaps item 11 |
| 15, 16. The two runbooks | **S** each | Prose, but writable only once 13, 14 and 5 define what the operator does |
| 6, 7. Warnings Phase 2, health history | **M–L** | Contractual, genuinely gated on v0.4 cleaning |
| 10. Cost-per-experiment | **S** | A tag and a query over machinery `aws-costs.md` already describes |
| 8. Degradation-conditional calibration | **L** | Real ML work, **blocked** on quantile output |
| 11. Outage-shaped training augmentation | **L** | Real ML work, **blocked** on items 2 and 3 existing to evaluate against |
| 12. `BaseForecaster` missingness contract | **S** | Meaningful only once a second model family exists |

So items 2–5, 9 and 14–16 land in **v0.3** as one focused chunk, not a milestone-sized programme,
with items 1 and 13 in v0.2 ahead of them and item 0 starting now. Items 8, 10 and 11 land in v0.5
because that is when their prerequisites exist, and item 12 in v0.9 with the NN spike.

Two orderings are constrained. **Item 13 must precede item 14**, and the v0.2/v0.3 split satisfies
that naturally: after 13 ships the check reports NWP age while `live_forecasts` still fails loudly
on very stale NWP, which is strictly better than today and never opens the window the constraint
guards against — silent degradation. The requirement this places on #424 is that it ships **WARN and
non-blocking**, like the two existing checks; a blocking check would both contradict the principle
and force item 14 to revisit it. Separately, [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147)
must precede item 3, both within v0.3.

**Two things to front-load, because retrofitting them is painful:**

1. **Add the scenario dimension to the metrics schema immediately, populated with a single `"none"`
   value.** Adding the column later means backfilling every historical metric or accepting a
   leaderboard discontinuity. A column holding one value costs almost nothing and removes a
   migration.
2. **Design the scenario vocabulary before writing any code, and treat it as a contract.** It gets
   stamped into metrics rows, so changing it later invalidates historical comparisons — exactly the
   re-runnability property `requirements.md` calls load-bearing. It is also the yardstick the v0.7
   capacity estimators and the v0.9 neural-net spike are both judged against, so it must be stable
   well before either.

One thing needs **no** change: `PowerForecast` requires no new degradation column, because a
consumer can derive the regime from `nwp_init_time` versus `power_fcst_init_time`, both already
present.

### 6.5 Dagster-versus-Airflow: three edits, verdict unchanged

Option C (stay on Dagster) still stands. But this work reweights one argument, adds a cost to
Option B, and makes the page's own central premise falsifiable.

**Reweight the asset-checks row.** It currently reads as an observability nicety. It is a
design-principle row: non-blocking WARN checks are the mechanism by which the system stays
fail-operational while still telling the truth. In Airflow, data-quality checks are ordinary tasks
and **blocking by default** (as of 3.3.0), so fail-open behaviour requires writing every check as a
task that deliberately never fails and reports out-of-band. That makes inherent stability depend on
developer discipline on every check, rather than on a first-class severity flag. Link the row to the
new page and say the gap is architectural rather than cosmetic.

**Add a cost to Option B.** Option B ports *only the live service* to Airflow — which is precisely
the half where the non-blocking check primitive matters most, since `power_data_is_fresh` and
`nwp_has_no_unexpected_nulls` are both production-side and the warning channel to NGED originates
there. It would move the fail-open half onto the orchestrator lacking a fail-open primitive, while
leaving the R&D half — which wants fail-fast and would be content with blocking tasks — on Dagster.
That is backwards. It does not kill Option B, but it belongs in the "Against" list.

**Add a trigger under "What would change this assessment".** The page's central argument rests on
experiment volume, which is now H2, with a threshold and a measurement. That cuts both ways: **if H2
is falsified — 20 experiments a month rather than 200 — the central argument for Dagster weakens
materially.** Saying so makes an architectural decision re-testable rather than permanent, which is
what the page's "documented seam" framing is for.

A fourth edit is conditional on [decision 2](#7-open-decisions): if scenarios become part of the
partition key rather than a metrics column, the all-time partition catalog gets roughly ten times
more cells, strengthening the existing "Airflow has the data model but no partition-status UI" gap.
Resolve decision 2 before touching this page.

### 6.6 Scope

Of the seventeen items, 4, 6 and 7 are on the critical path to a contractual v1.0 deliverable, and
**13 and 14 are production-correctness fixes rather than quality work** — divergence 1 is a live
hard-failure mode that would cut NGED off entirely during an extended NWP outage. The rest is
quality and stability work. So the widening is smaller than it looks, and part of it is overdue
rather than new. It is still a widening against the 2026-07-01 "live service first"
reprioritisation, and that call is Jack's — but 13 and 14 sit *inside* that priority, not against
it.

---

## 7. Open decisions

1. **Is v0.3 the right home for the failure-scenario suite?** It widens v0.3, which currently
   carries leaderboard, baselines and production monitoring, and it delays v0.5. The alternative is
   to let v0.5 run on clean-data skill and retrofit scenario scoring, which is cheaper now but means
   re-judging every v0.5 experiment later. *Recommendation: take the delay — per §6.4 it is small,
   because the v0.3 tranche is mostly pure functions plus two schema decisions.*

2. **How do failure scenarios fit the evaluation model?** `EVALUATION_SCOPES` is currently
   `("leaderboard", "production_monitoring", "ad_hoc")`, with `EvalScopeType` a deliberately
   narrower `Literal["leaderboard", "ad_hoc"]` — the subset the asset handles today, documented to
   expand when Phase 8 lands. Any new scope must follow that two-name pattern. The choice is a
   fourth scope, or a new dimension within `leaderboard`. *Recommendation: the dimension, as a
   metrics **column**, not part of the partition key* — it makes degradation behaviour a first-class
   property of every experiment, and leaves partition counts unchanged. This also settles the
   conditional edit in §6.5.

3. **Does the inherent-stability page cover the whole system, or production only?** The
   operability audience cares about production, but omitting the R&D asymmetry makes the page read
   as "we never fail", which is the wrong instruction for future Claude sessions. *Recommendation:
   one page, production-first in ordering, with R&D as a clearly-marked contrasting section near the
   end.*

4. **Do we commit to the weather-blind guarantee?** If "never worse than the incumbent" is to be
   load-bearing rather than aspirational, item 11 belongs in v0.5 and we accept the training cost.
   *Recommendation: commit — it is the strongest sentence in the whole principle and should not be
   left unbacked.* The where-complexity-lives principle makes this more than a skill question: item
   11 is what keeps the weather-blind fallback out of the production service.

5. **Six hypotheses, or three?** H4–H6 are proposals only.
