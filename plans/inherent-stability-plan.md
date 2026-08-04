# Inherent stability — discussion summary and plan

Consolidation of the 2026-08-04 design session. Nothing in the repo has been changed yet. This
document is the working plan for the docs pages and issues described in [§7](#7-the-plan); delete it
once that work lands, pasting a summary into the PR body.

**Contents**

1. [The philosophy](#1-the-philosophy)
2. [Terminology, and what the industry calls this](#2-terminology-and-what-the-industry-calls-this)
3. [Mechanisms](#3-mechanisms)
4. [What already exists in the code](#4-what-already-exists-in-the-code)
5. [Engineering hypotheses](#5-engineering-hypotheses)
6. [Gaps found](#6-gaps-found)
7. [The plan](#7-the-plan)
8. [Decisions needed from Jack](#8-decisions-needed-from-jack)

---

## 1. The philosophy

### The core statement

> **We never stop answering. We get less certain instead — and we say so in the answer itself.**

And, since the incumbent is the floor:

> **At our worst we degrade to roughly NGED's incumbent method. At our best we beat it
> substantially. There is no state in which NGED is worse off than they are today.**

That second sentence is currently an *intention*, not a measured fact. See
[the weather-blind problem](#32-the-weather-blind-problem-and-a-real-design-choice).

### The incumbent is the floor — and it is itself inherently stable

`docs/background/nged-incumbent-forecast.md` records the incumbent recipe: for each substation and
target half-hour, 13 historical analogues at the same time-of-day on the same weekday — 6 from the
last 6 weeks, 7 from 49–55 weeks back. No weather, no ML, no holiday alignment, no load-growth
scaling. An operator reads a forecast off the plot; ~P95 when a single number is needed.

Two consequences that matter to us:

- **It consumes no NWP at all**, so an NWP outage does not degrade it. That makes NWP outage the
  *hard* test for us: can our weather-blind fallback still beat a method that never used weather?
- **It survives a power-data outage**, because the 49–55-week-back analogues are unaffected by
  recent staleness. Only the 6 recent analogues age.

So the incumbent already embodies the philosophy we are articulating. That is not a criticism of it
— it is why it is the right floor. It also gives us a far better failure criterion than an
arbitrary staleness threshold:

> **We should only fail when we can no longer beat the incumbent.**

That is measurable, and `nged_incumbent` is *already planned* as a leaderboard baseline
(`docs/roadmap/metrics-and-leaderboard.md` → "The headline baseline: `nged_incumbent`"). So the
machinery to test it is already on the roadmap.

### The degradation ladder

| Rung | Available inputs | Expected behaviour |
|---|---|---|
| 0 | Everything fresh | Best skill; narrowest bands |
| 1 | NWP a few hours stale | Slightly worse; bands widen slightly |
| 2 | No NWP for days/weeks | Weather-blind: lags, calendar, per-series structure. **Should still beat incumbent.** |
| 3 | No NWP *and* no recent power | Calendar + climatology + year-old history. Converges toward *being* the incumbent. |
| 4 | Nothing at all | Physical envelope (clear-sky) + climatology. Very wide bands, still bounded and still true. |

Rung 4 is worth stating explicitly because it demonstrates there is no input state in which we have
nothing true to say. Clear-sky irradiance needs only latitude, longitude and time.

### The precise rule: missing vs wrong

Jack's hardening position — always output — is right, but needs one distinction to be safe:

- **Absent or stale input** → always produce a forecast. Degrade, widen, declare.
- **Detectably wrong input** → do *not* consume it. Treat it as missing, which routes it back into
  the always-output path above.

A stuck meter reporting 2.1 MW for 52 hours is not missing data; it is actively misleading, and a
lag-feature model will propagate it happily. The incumbent has the identical vulnerability. The
vocabulary for this already exists in the delivery-table design: `STUCK TIMESERIES`,
`INVALID TIMESERIES VALUE`, `MISSING VALUE`.

So "always output" is not "always trust". Detection of *wrong* data is what makes "always output"
safe rather than reckless.

### Static and dynamic stability — the damping half

Borrowed from vehicle and aircraft dynamics, and genuinely useful here:

- **Static stability** — when disturbed, does the initial tendency return the system toward correct?
- **Dynamic stability** — does that return settle, or overshoot and oscillate, possibly with growing
  amplitude?

A system can be statically stable and dynamically unstable: always heading the right way, but
hunting and never settling. The data-engineering versions are exactly the things that generate
on-call pages — retry storms, backfill cascades where a repair triggers more repairs, a model
auto-demoted on a bad day and auto-promoted on a good one, flapping between the two.

So the principle has two halves, and the second is the one people forget:

> **Restoring force** — when inputs degrade, the system's normal path already moves toward a
> sensible output.
>
> **Damping** — that correction settles rather than oscillating.

Damping is where bounded retries with backoff, rate limits on retraining, and hysteresis on
model promotion/demotion belong. For the operability argument this half matters *more*: teams get
woken by oscillation far more often than by clean failure.

### What this is deliberately not: Postel's law

"Be liberal in what you accept" sounds like this principle but has fallen out of favour precisely
because liberal acceptance is how silent corruption propagates (see RFC 9413 on protocol
maintenance). Our stance is sharper:

> **Liberal about missing inputs. Strict about malformed ones.**

The Patito contracts layer is the "strict" half, and it is what stops inherent stability from
decaying into "accept anything and hope".

### The R&D / production asymmetry

This resolves what would otherwise read as a contradiction ("fail loudly" vs "never fail if any
data is retrievable"). The two contexts have different costs of being wrong:

| | Production | Model R&D |
|---|---|---|
| Cost of no output | High — a user is waiting | Nil — rerun it |
| Cost of a quietly-degraded output | Moderate, **if** flagged in the data | High — silently poisons a model and every comparison built on it |
| Correct posture | Fail-operational: degrade and declare | Fail-fast: refuse to proceed |

The general rule:

> **Fail in the direction where being wrong is cheapest to recover from.**

In production that is forward; in R&D that is backward. Same principle, opposite direction — a
much better story than two rules that appear to conflict.

Note this is *not* the same axis as Dagster's WARN-vs-ERROR severity. R&D lives in the CV/backtest
assets and the training path, so the natural mechanism is a strict-mode flag on the
feature/validation layer, plus the asset tagging already proposed in
[#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423).

### Which analogy to use in the docs

**Recommendation: lead with steering. Do not use nuclear fission.**

*Why not nuclear.* In a reactor, **safe means off**. Negative reactivity coefficients, gravity-drop
control rods, passive decay-heat removal — every mechanism drives toward shutdown. Our thesis is
the opposite: keep producing, because a downstream consumer needs an answer. A reader told "passively
safe like a reactor" will reasonably infer "fails to a stopped, cold, quiet state", which is exactly
what we are arguing against.

*Why steering.* Front-end geometry tilts the steering axis back so the tyre contact patch trails
behind where that axis meets the ground — like a shopping-trolley caster. Any sideways force
generates a torque that swings the wheel back into line. Let go of the bars and the bike keeps
going, correctly, without an operator. That is *fail-operational*, which is what we want.

Cars add kingpin inclination (the body rises slightly when steering off-centre, so gravity pulls
back to straight) and pneumatic trail. Together these are the **self-aligning torque** — and here
is the part that maps beautifully: that torque is *also* the driver's feedback channel. As the front
tyres approach the grip limit, pneumatic trail collapses and the steering goes light. The car reports
its own degradation **through the same mechanism that keeps it centred**. That is precisely what
widening confidence bands do for us.

*Terminology hazard, worth one sentence in the docs.* In the automotive industry "passive safety"
already means **crashworthiness** — seatbelts, airbags, crumple zones — as opposed to "active
safety" (ABS, ESC, AEB). Self-centring steering is filed under inherent stability, not passive
safety. So "passive safety" plus a car analogy is, by automotive vocabulary, backwards.

**Jack's preference for "inherent stability" is the right call.** It avoids the nuclear *safe = off*
implication and the automotive *airbags* implication, and a non-expert parses it correctly on first
read: the stability is a property of how the thing is built.

*Optional one-clause cameo.* The reactor idea worth keeping, if any, is the negative fuel-temperature
coefficient: as fuel heats, Doppler broadening of the U-238 resonances absorbs more neutrons and
reactivity drops — the system automatically does less as conditions worsen, with **no detection
step**. One clause, no section.

---

## 2. Terminology, and what the industry calls this

The concept exists in the industry, scattered across several terms. Ones worth naming in the docs:

| Term | Origin | Fit |
|---|---|---|
| **Graceful degradation** | General | The plain-English name. Lose fidelity, not availability. **Jack wants this in the docs.** |
| **Static stability** | AWS Builders' Library (borrowed from vehicle/aircraft dynamics) | Best technical fit for one specific part: a system keeps working using state it *already has* when dependencies fail — no new control-plane calls. Exactly `live_forecasts` reading NWP already on disk. |
| **Fail-operational vs fail-passive** | Avionics autoland; ISO 26262 / autonomous driving | The sharpest vocabulary for the distinction we are drawing. Fail-passive: disengage cleanly, hand back to the human. Fail-operational: keep delivering through the fault. Ours is fail-operational. |
| **Blast radius** | SRE / infra | Related but a **different axis** — *how much* fails, not *which way* it fails. Our daily/6-hourly partitioning is a genuine blast-radius story. Keep the two separate in the prose. |
| **Bulkhead / circuit breaker** | Nygard, *Release It!* | The canonical stability-patterns reference. |
| **Stale-while-revalidate** | RFC 5861 (HTTP caching) | Serve stale while refreshing behind the scenes. Close cousin of the NWP fallback. |
| **Write-Audit-Publish** | Data engineering | Worth naming as the **opposite stance**: stage, validate, block publication on failure. Fail-closed. We are deliberately fail-open on freshness. |
| **Postel's law** | RFC 761 | Named only to disown — see above. |

Dagster gives first-class vocabulary for the fail-open choice: `AssetCheckSeverity.WARN` vs `ERROR`,
and `blocking=True/False`. That lets the docs be concrete rather than philosophical.

---

## 3. Mechanisms

### 3.1 XGBoost's NaN handling — and the trap

Currently this carries most of the load, so the docs should explain it plainly.

**How it works.** XGBoost uses sparsity-aware split finding (Chen & Guestrin). At every split it
learns not just a threshold but a **default direction** for rows where that feature is missing: it
tries sending all missing rows left, then right, and keeps whichever gives better gain. Missingness
is not imputed; it is *routed*. This is a real inherent-stability mechanism, and it is why
`_nullify_leaky_lags` can null a feature rather than drop a row.

**The trap — document this.** The default direction is learned **from the missingness present in
the training data**. If a feature is never missing during training, XGBoost still picks a direction
for it, but that choice was never evaluated against any data — it is effectively arbitrary. In
production, the first time that feature *is* missing, every affected row takes an untested path.

So the guarantee is narrower than "XGBoost handles NaN". It is:

> **XGBoost handles missingness patterns it saw during training.**

For a production failure like "the whole ECMWF run is absent" — which may never occur in a clean
training set — we currently have *no evidence at all* about behaviour.

### 3.2 The weather-blind problem, and a real design choice

This is the direct consequence of Jack's incumbent insight combined with §3.1.

The claim "with no NWP for weeks we still beat the incumbent" is not delivered by XGBoost's NaN
routing. A model trained *with* NWP features, run *without* them, falls back on arbitrary default
directions — not on a well-trained weather-blind model. To make the claim true rather than hoped-for,
there are two options:

| Option | Mechanism | Verdict |
|---|---|---|
| **A. Train for it** | Include outage-shaped scenarios in training so the default directions are learned, not arbitrary | Purist: one model, no branch, genuinely inherent. Preferred if it performs. |
| **B. Fallback cascade** | Keep a cheap weather-blind model trained alongside; use it when NWP is absent | Pragmatic. Technically a branch, but a deterministic function of what data exists — no detection step, so it is defensible. |

**Recommendation: try A, keep B as the fallback, and let the failure-scenario suite decide.** That
makes the choice empirical rather than aesthetic. It also argues for moving outage-shaped training
augmentation earlier than v0.9 if we want the "never worse than incumbent" claim to be load-bearing.

### 3.3 Widening bands — the in-band degradation signal

The problem with "keep forecasting on stale data" is that a stale forecast **looks identical to a
fresh one**. Every other fix — staleness columns, warning tables — is a *side channel*: it requires
the consumer to go and look.

Uncertainty bands are not a side channel. They are expressed in the output's own units. A forecast
whose P5–P95 spread has doubled has *already told* the consumer to be more cautious, through the only
number they were going to read anyway. This is the self-aligning-torque property: the machine reports
its own degradation through the same mechanism it uses to do its job. **No separate monitoring system
has to work for the safety property to hold.**

Two honest caveats:

- `XGBoostConfig.objective` currently defaults to `reg:squarederror` — today's model is a **point**
  forecast. Quantile output ([#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263))
  is a prerequisite, not a consequence.
- Bands widening *correctly* under degradation is not automatic even with quantile regression. See
  §3.6.

### 3.4 The physical envelope floor

Clear-sky irradiance needs only latitude, longitude and time, so there is **no input-failure state
in which we have nothing true to say**. As data degrades, the forecast should relax toward the
physical envelope — wide, honest, and still bounded by what the sky can possibly deliver.

This is the reactor's negative reactivity coefficient in our domain: the worse the situation, the
less the system asserts, automatically. It is also useful as an ordinary feature, so it earns its
keep twice.

### 3.5 Three audiences, three channels

The three failure-signal use-cases have genuinely different requirements and resolve to different
mechanisms — not one `forecast_warnings` table serving all three:

| Audience | Question | Channel |
|---|---|---|
| **Forecast users** (NGED) | "How much should I trust *this row*?" | In-band: quantile spread + `nwp_init_time` (already on the row) |
| **Data providers** (NGED telemetry, ECMWF) | "Is *your* feed broken, and since when?" | Aggregated and **attributable**: `power_forecast_warnings` / the freshness check's late-series table, keyed by source |
| **Us, the developers** | "Is *our* system at fault?" | Out-of-band: Sentry + the missed-check-in alarm |

**The developer channel carries a specific hazard created by inherent stability**: a system that
always succeeds looks identical to a system that is not running at all. Both produce zero failures.
That is why the Sentry missed-check-in alarm firing from *outside* the deployment is load-bearing
rather than belt-and-braces — it is the one piece of active monitoring the design genuinely cannot
do without. The docs should say so explicitly.

Design rule for the provider channel: a warning saying "NWP was 18 hours old" is only actionable if
it also says *whose* NWP and *which* run. **Warnings must name the upstream source, not just the
symptom.** This implies a `warning_source` field on Table 2.

### 3.6 Neural nets and differentiable physics — the forward look

**Don't zero-fill; remove from the set.** Zero is a meaningful value in physical units — 0 MW,
0 W/m², 0 °C are all real states. Replacing a missing value with zero asserts something false. The
clean formulation is to treat inputs as a **set of tokens** (value embedding + feature-identity
embedding + time embedding) and simply not emit tokens for absent inputs. Attention is natively
permutation-invariant and variable-length, so a missing input is *structurally absent* rather than
encoded as a sentinel. Mask the attention matrix for padding only. Set Transformer / Perceiver shape.

**Dense alternative: value + mask channels.** Pass a binary "is present" indicator alongside every
value so the network can distinguish "zero" from "unknown". GRU-D is the well-known precedent —
irregularly-sampled clinical time series with heavy missingness, using masks plus learned decay of
the last observation toward an empirical mean. Structurally the same problem as our telemetry gaps.

**Our missingness is blocky, which changes the combinatorics.** The explosion only bites if
missingness is arbitrary *per element*. Ours is not: NWP {fresh, stale, absent} × telemetry
{present, partial, absent} × metadata {present}. That is on the order of ten to twenty realistic
regimes, not 2ⁿ — so training with **structured** dropout that mimics real outage patterns is
entirely tractable, and is a far better match for reality than element-wise random dropout.

**Differentiable physics is the easiest and strongest piece.** A physical model has a defined output
for any input state; where an input is absent, substitute a climatological prior or a physical bound
and let the physics propagate it. Structurally: physics gives the envelope, the learned residual
sharpens it. As data degrades, the residual head has less to work with and the answer relaxes toward
the physics prior — which is always computable. **No branching, no fallback logic, no
`if data_is_missing:`.** Same code path, right behaviour, because of how it is arranged.

**For honest bands without solving the combinatorics inside the model: conformal prediction, per
regime.** Split/Mondrian conformal calibration is post-hoc — calibrate interval widths separately
for each degradation regime (full data / stale NWP / no NWP / no telemetry) using held-out residuals
from that regime. Finite-sample coverage guarantees per regime, no retraining. Two things make this
attractive: it works with **XGBoost today**, so widening bands could ship before any PyTorch work
exists; and it makes "the spread widens when inputs are missing" a *measured* fact rather than a hope.

### 3.7 Correction on data dropout for regression

Jack's understanding was that dropout is fine for classification but strictly harmful for regression.
That framing is not quite the real distinction, and the accurate version is more useful.

Standard dropout is a regulariser; its effect is not classification-vs-regression. It is used in
regression networks routinely — MC dropout for regression uncertainty (Gal & Ghahramani) is a whole
line of work, and Wager, Wang & Liang showed input-layer dropout is a form of adaptive regularisation
(roughly ridge on variance-scaled features, for linear models).

The two **real** problems, both of which apply here:

1. **Dropout-to-zero without a mask corrupts the target.** If the network cannot distinguish
   "dropped" from "genuinely zero", it learns `E[y | x with random zeros]`, which biases the
   conditional mean. With an explicit mask channel (or token removal), dropout training is well-posed.
   This is probably the source of the "harmful for regression" intuition — it is a **representation
   bug**, not a regression-vs-classification law.
2. **Dropout trains for MCAR; production missingness is not MCAR.** Randomly dropping inputs teaches
   the model that data goes missing independently of everything else. Real outages correlate with
   time of day, weather systems, and provider incidents. A model trained on random dropout and
   deployed under structured missingness is calibrated for a world it does not live in — and the
   miscalibration shows up as over-confident bands *during a real outage*, the worst possible time.

**Conclusion:** do not lean on MCAR input dropout as the primary mechanism. Use architecture
(token removal / masks / physics fallback) for *capability* under missingness, structured
outage-shaped dropout for *learned behaviour*, and regime-conditional conformal calibration for
*honest widths*. Complementary; only the last is strictly necessary for the bands to be trustworthy.

---

## 4. What already exists in the code

Verified by reading the repo during this session.

| Piece | Status | Serves |
|---|---|---|
| `promoted_model` → champion copied to **local disk, zero MLflow at inference** (`src/nged_substation_forecast/defs/production_assets.py:120`) | ✅ | The single best production-stability example. The whole experiment-tracking stack can be down and 06:00 inference is unaffected. |
| `live_forecasts` selects "the freshest NWP run present as of `power_fcst_init_time`" — a **relative** query, not "today's run" | ✅ | A missed download degrades to an older run through the *normal* path. No fallback branch exists to get wrong. |
| `power_data_is_fresh` — `blocking=False`, `severity=WARN`, reads **on-disk data recency** not materialisation time (`src/nged_substation_forecast/defs/checks.py`) | ✅ | The canonical warn-don't-block pattern. Its own docstring explains why materialisation-freshness would miss the real failure. |
| `nwp_has_no_unexpected_nulls` — in-asset WARN, computed from the frame already in memory (`defs/assets.py:178`, `:275`) | ✅ | Same pattern, cheaper: no second read of the table. |
| `nwp_init_time` on `PowerForecast` (`packages/contracts/src/contracts/power_schemas.py:245`) | ✅ | Provenance already travels in-band with every row. |
| Sentry missed-check-in alarm, fired from **outside** the deployment | ✅ | The necessary complement (§3.5). |
| Daily / 6-hourly asset partitioning | ✅ | Blast radius — the separate axis. |
| Patito contracts at every boundary | ✅ | The strict half that stops this decaying into Postel's law. |
| `DELIVERY_QUANTILES` (13 levels) + pinball / PICP / CRPS / spread-skill metrics | ✅ | Everything needed for §3.3 *except* a quantile-emitting model. |
| `XGBoostConfig.objective = "reg:squarederror"` | ⚠️ | Point forecast today. Gap for widening bands. |
| `power_forecast_warnings` Table 2 (incl. `STALE NWP`, `STALE POWER`) | 🚧 designed, **no issue** | The forecast-user channel. |
| `asset_health_history` Table 3 | 🚧 designed, **no issue** | User + provider channel. |
| NWP ingestion completeness checks ([#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161)) | 🚧 | Operator + provider. |
| Quantile pipeline ([#262](https://github.com/openclimatefix/nged-substation-forecast/issues/262) / [#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263) / [#264](https://github.com/openclimatefix/nged-substation-forecast/issues/264), v0.5) | 🚧 | Prerequisite for §3.3. |
| [#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423) tag assets R&D vs production | 🚧 | The asymmetry mechanism. |
| [#420](https://github.com/openclimatefix/nged-substation-forecast/issues/420) silence warnings for dead series | 🚧 | Warning fatigue. |

**Framed for the ops question — what would page a human?** Of the above, only a genuine upstream
schema change. That is the operability argument, and it is more persuasive as a table of failure
modes than as philosophy.

Confirmed implemented in `METRIC_NAMES`: `mae`, `nmae`, `rmse`, `mbe`, `crps`, `spread_skill_ratio`,
`pinball_loss`, `mean_pinball_loss`, `picp`, `interval_width`. So T1.3 needs no new metric — only a
scenario dimension to slice them by.

### 4.1 Audit: is the WARN/ERROR and Sentry discipline already in line with this plan?

Audited 2026-08-04. **Largely yes — and more carefully than expected.** Three divergences, all in
`live_forecasts`.

**In line:**

- **Both asset checks are WARN and non-blocking**, and **no ERROR-severity check exists anywhere in
  the repo.** Both carry inline comments giving the fail-open reasoning explicitly.
- **Sentry's log-to-event capture is deliberately disabled** — `LoggingIntegration(event_level=None)`
  in `_sentry.py`. This is subtle and exactly right: without it every `ERROR` log becomes an event,
  so a fail-open design would flood Sentry with events for conditions it deliberately tolerates.
- **The failure hook is attached to the three *scheduled* jobs only** (`defs/schedules.py`), not to
  manual, backfill, or experiment runs. The R&D/production asymmetry is therefore already
  implemented at the telemetry layer.
- **Three distinct Sentry channels** — exceptions (error), freshness (`capture_message`,
  `level="warning"`), heartbeat (absence) — mapping cleanly onto the three audiences in §3.5.
- **`report_power_freshness` never raises**, and its docstring gives the philosophy-correct reason:
  a telemetry failure inside a `blocking=False` check would trip the failure hook and fail the run.
  That is an explicit recognition that a bug in the *warning* path can silently convert fail-open
  into fail-closed.
- **The heartbeat is success-only and skipped on replay** — a replay backfill is not evidence that
  the service is alive now.
- **R&D fails fast.** All five `raise` sites in `cv_assets.py` are in the training/metrics path.

**Divergences, all in `live_forecasts`:**

1. **NWP more than ~15 days stale → hard failure.** A weeks-old NWP run no longer covers the
   forecast window, so its rows join to a null `ensemble_member`, are filtered out at
   `production_assets.py:280`, and the asset raises on `forecasts.height == 0` at `:286`. This is
   exactly rung 2 of the degradation ladder and today NGED gets nothing at all.
2. **Between 0 and ~15 days stale → silent degradation.** The forecast is built from an
   increasingly ancient run with `nwp_init_time` recorded on the row but **no warning anywhere** —
   no check, no Sentry event, no widened bands. Arguably worse than the hard failure, because it is
   undetectable from the consumer side without deriving staleness by hand.
3. **`select_nwp_init_time` raises when nothing qualifies** (`_production_helpers.py:71`). Narrower
   in practice — it fires only when the NWP table holds no run at or before the cutoff — but it is
   the same class, and it does bite `replay` backfills older than the retention window.

**Correctly fail-fast, keep as-is:** the raise on empty `trained_ids` (`production_assets.py:241`).
An empty model is a promotion bug, not a data outage. The codebase already distinguishes the two
cases — just not consistently.

**Ordering constraint this creates:** fixing divergence 1 *without* fixing 2 would make things
worse, converting a loud failure into a silent one. The `live_forecasts` check must land **with or
before** the degradation change. See issues 13 and 14 in §7.2.

---

## 5. Engineering hypotheses

### All three already exist as prose — they just aren't falsifiable

| Jack's | Already in `docs/background/requirements.md` |
|---|---|
| H1 | "Uptime: lenient by design" (L142) — "never a 2am page, and no on-call rota" |
| H2 | "ML experimentation at scale" (L76) — "on the order of hundreds of ML experiments per month" |
| H3 | "A short, safe path from R&D to production" (L94) — promotion as an audited config change |

So this is **extract, elevate, and make falsifiable** — not a page from nothing. The new page must
*link* to that prose rather than restate it, or we get two divergent copies.

**One gap the framing exposes.** "Uptime: lenient by design" is a *defensive* argument — outages
don't cause much damage, because forecasts run 14 days ahead and delivery is decoupled from compute.
H1 is a stronger *positive* claim: interventions will be rare. Those are different claims, and the
second is the one actually in dispute. A sceptic is not moved by "it's fine when it breaks"; they
may be moved by "here is our predicted intervention rate, here is how we're measuring it".

### Hypotheses, not aims — three reasons

1. **NIA funding is for transferable learning, including negative results.** Six report issues
   ([#128](https://github.com/openclimatefix/nged-substation-forecast/issues/128),
   [#130](https://github.com/openclimatefix/nged-substation-forecast/issues/130),
   [#131](https://github.com/openclimatefix/nged-substation-forecast/issues/131),
   [#132](https://github.com/openclimatefix/nged-substation-forecast/issues/132),
   [#135](https://github.com/openclimatefix/nged-substation-forecast/issues/135),
   [#156](https://github.com/openclimatefix/nged-substation-forecast/issues/156)) are natural
   consumers. "We predicted X, measured Y" is a far better report section than "we aimed for X".
2. **It converts a disagreement into something that resolves itself.** Jack and his co-founder hold
   opposing *predictions* about devops burden. A doc arguing one side is a worse outcome than a
   pre-registered measurement that settles it — and pre-registering a number signals confidence in
   a way prose cannot.
3. **It forces the measurement artifacts to exist in advance.** See the intervention log below.

**The commitment this entails:** a hypothesis without a number is an aim wearing a lab coat. Each H
needs a threshold and a window, and we must be willing to record a falsification. If H1 fails, that
is a finding worth publishing.

### The time-critical item: an intervention log

H1 is the **only** hypothesis that cannot be measured retrospectively. Experiment counts live in
MLflow forever; promotion steps can be counted any time. But "how many times did a human have to
intervene, and why?" is unrecoverable unless recorded as it happens — and the service is already
running on AWS, so the clock has started.

The artifact is cheap: an append-only log (markdown file or Delta table) with date, trigger, cause
category, human-minutes, and whether a runbook existed. The cause taxonomy is the point — H1
predicts that essentially all entries fall into "upstream format/contract change".

**This should ship first, ahead of the docs page.** Every week without it is a week of unrecoverable
evidence.

### Proposed operationalisation

H1 bundles three distinct claims. Keep the number (these will be cited from issues and reports —
**append, never renumber**) but give it three named tests.

| | Claim | Test | Threshold | Source | Resolvable |
|---|---|---|---|---|---|
| **H1** | Manual attention only for upstream format changes; graceful, legible degradation; faithful uncertainty | | | | |
| T1.1 | *Operability* | Interventions per quarter, classified by cause | ≥90% attributable to upstream format/contract change; zero out-of-hours | Intervention log | ~2 quarters of v1.0 |
| T1.2 | *Graceful degradation* | Forecast emitted for every series under every failure scenario, **and still beats `nged_incumbent`** | 100% emitted; beats incumbent at rungs 0–2 | Failure-scenario suite | v0.3, **after [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147)** |
| T1.3 | *Faithful uncertainty* | PICP + pinball computed **per degradation regime** | PICP within tolerance of nominal in every regime | Leaderboard, scenario dimension | v0.5 |
| **H2** | Hundreds of experiments per month | Registered leaderboard experiments/month, **and** median human-minutes each | ≥200/month; ≤5 human-min each | MLflow + timestamps | v0.5 |
| **H3** | One-click promotion of the winner | Commands from "leaderboard says X won" to "X is serving" — **and the same for rollback** | ≤1 each way | Runbook + `promoted_model` | v0.3 |

Notes:

- **T1.2 now has the incumbent as its acceptance criterion**, not merely "output exists". This is
  the direct consequence of Jack's insight, and it is strictly stronger. **Dependency:**
  `nged_incumbent` is *designed* (`docs/roadmap/metrics-and-leaderboard.md` → "The headline baseline")
  but **not yet implemented** — there is no Python match for it in the repo. It is
  [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147), also v0.3, so T1.2
  is blocked on that landing first.
- **T1.3 is the sharpest available test and needs no new metrics** — PICP, pinball and spread-skill
  are all ✅ from v0.3. Slicing them by degradation regime is the same "scenario dimension" that
  serves T1.2. One piece of machinery, three jobs.
- **H3 must include rollback.** One-click promotion without one-click demotion is not safe — it is
  the damping half of §1. `promotable_model_runs` + `promoted_model` give most of the forward path;
  the reverse path is worth naming in the hypothesis rather than discovering it missing during an
  incident.
- **H2's real claim is the second number.** A 200-config sweep trivially clears "hundreds per
  month". The transferable claim is throughput of *decision-grade* experiments with negligible human
  time. Worth pairing with cost-per-experiment, since `docs/architecture/aws-costs.md` already has
  the machinery and "hundreds of experiments/month for £N" is exactly what another DNO can reuse.

### Candidate additional hypotheses

Suggestions only. Stopping at six — hypothesis inflation makes the set unmemorable and each carries
a measurement cost.

- **H4 (cost)** — the service runs under £X/month at v1 scale and £Y at v2. A real number already
  exists: `docs/architecture/aws-costs.md` estimates **~£25–35/month for the whole v1 stack** at 32
  time series (not merely the control-plane box), with a v2 estimate on the same page. The most
  directly transferable NIA finding of the lot, and a second independent answer to the devops-team
  worry.
- **H5 (operability by a non-expert)** — an NGED operator can run the service from runbooks alone.
  Already designed as the operator contract in `docs/roadmap/handover.md`; framing it as a hypothesis
  turns the game days (handover §6) from a training exercise into a **measurement**.
- **H6 (scale without redesign)** — the architecture goes from 32 to ~2,500 series without
  structural change. The central engineering bet of the whole project, currently unstated anywhere.
  Only resolvable at v2, which is an argument for writing it down now.

---

## 6. Gaps found

1. **No GitHub issue for `power_forecast_warnings` or `asset_health_history`** — two of the five
   contractual v1.0 delivery tables. Fully designed in `docs/roadmap/delivery-tables.md`, listed
   under v1.0, tracked nowhere. `power_forecast_warnings` *is* the user-facing half of the
   philosophy.
2. **Bands don't widen with degradation.** #262–264 turn an *ensemble* into percentiles. Nothing
   makes the spread respond to stale or missing inputs.
3. **No failure-scenario evaluation** anywhere in docs or GitHub. Consequence: every v0.5 XGBoost
   experiment will pick a champion on clean-data skill alone.
4. **The physical envelope exists as a *feature*, but not as a *floor*.** Correcting an earlier
   reading of this gap: clear-sky irradiance is already a named deliverable of
   [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168) ("Linearize weather
   features for solar power and wind power"), designed in `docs/roadmap/xgboost-improvements.md` →
   "Linearised physics features for solar and wind", with substantial further physics in
   `docs/techniques/differentiable-physics.md` (POA transposition, Faiman thermal model, the planned
   `pvlib-pytorch`). What is missing is only the *framing*: nothing says clear-sky is what we fall
   back to when we have nothing. So this is an extension of #168, not new work.
5. **No weather-blind guarantee.** §3.2 — the "beats the incumbent with no NWP" claim has no
   mechanism behind it yet.
6. **No intervention log.** §5 — and this evidence is being lost daily.

---

## 7. The plan

### 7.1 Docs

**New page A: `docs/architecture/inherent-stability.md`**, nav directly after `Overview`.

`architecture/` because it is durable design rationale, and — for the future-Claude purpose —
CLAUDE.md's code-style rules permit code to link to `docs/architecture/`, so docstrings can point
at it without rotting.

Structure, in this order, serving the three purposes in sequence:

1. The principle in plain language + the incumbent-as-floor statement + the steering analogy.
   *(general reader; the part a sceptical co-founder actually reads)*
2. The **failure-mode table** — what breaks / what the system does / does a human get paged.
   Grounded in §4. *(the operability argument)*
3. The **rules**, numbered and imperative. *(instruction for future Claude sessions — prose is bad
   instruction, a numbered list is good)*
4. Mechanisms — §3, condensed.
5. What this deliberately is not — Postel's law; the R&D/production asymmetry.

**New page B: `docs/engineering-hypotheses.md`**, nav immediately after "Documentation Guide" —
above Background, because these are *our* claims, not NGED's context.

Three-page division of labour, no duplication:

| Page | Answers |
|---|---|
| `background/requirements.md` | *Why* we need this (NGED-derived). Existing prose stays put. |
| `engineering-hypotheses.md` | *What we claim*, how it is tested, what would falsify it. |
| `architecture/inherent-stability.md` | *How the design delivers H1.* |

**Pointers, without which the pages won't do their job:**

- `CLAUDE.md` — a short entry under Architecture. A docs page alone will not reliably reach Claude
  in future sessions; CLAUDE.md is what is always in context.
- `docs/roadmap/handover.md` §1 "The operator contract" — inbound link.
- `docs/index.md` — one line for each new page.
- `docs/background/nged-incumbent-forecast.md` — add a "Why it matters for us" note that the
  incumbent is our degradation floor.

**Edits to existing roadmap pages:**

| Page | Edit |
|---|---|
| `roadmap/delivery-tables.md` | Subsection on degradation-conditional band widening (Table 1). Add `warning_source` field to Table 2. |
| `roadmap/metrics-and-leaderboard.md` | New section: the failure-scenario suite, how it is scored, and the incumbent as acceptance criterion. |
| `roadmap/engineering-health.md` | Degradation smoke-tests in the scientific-rigor section. |
| `roadmap/xgboost-improvements.md` | The NaN default-direction caveat (§3.1). |
| `roadmap/index.md` | Milestone bullets for the new issues. |
| `background/requirements.md` | Cross-links to the hypotheses page. |
| `architecture/why-dagster-not-airflow.md` | Three surgical edits — see §7.5. |

### 7.2 GitHub issues

Sequencing argument: **the evaluation machinery must precede the model experiments it is meant to
judge.**

**"Milestone" below means the roadmap milestone / parent epic, not a GitHub milestone field.** The
repo has **zero** GitHub milestones defined (`gh api .../milestones` returns an empty list) — the
ordering lives in epic sub-issue lists and the OCF project board, per CLAUDE.md. So each row below
means "attach as a sub-issue of that epic, positioned by execution order".

| # | Issue | Milestone | Rationale |
|---|---|---|---|
| 0 | **Intervention log** — artifact, cause taxonomy, runbook line | **now, ahead of everything** | Evidence is being lost daily |
| 1 | Degradation smoke-tests: ablate input groups; assert output exists, stays in physical bounds, doesn't explode | **v0.2** | Cheap, CI-fast, no MLflow. Sibling of [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229). |
| 2 | Canonical failure-scenario suite — named, versioned degradation transforms over `AllFeatures` | **v0.3** | Shared by tests, leaderboard, and later training. Must exist before v0.5. |
| 3 | Score every leaderboard experiment under each failure scenario, **against `nged_incumbent`** | **v0.3** | Otherwise v0.5 picks a champion blind to degradation behaviour |
| 4 | `power_forecast_warnings` **Phase 1**: `STALE NWP` + `STALE POWER` only, with `warning_source` | **v0.3** | No dependency on v0.4/v0.6/v0.7. The user-facing half, buildable now. |
| 5 | Rollback path for `promoted_model` (H3's second half) | **v0.3** | Pairs with #4 |
| 6 | `power_forecast_warnings` **Phase 2**: meter-error types | **v0.4** | Depends on improved cleaning |
| 7 | `asset_health_history` table | **v0.4** | Same dependency |
| 8 | Degradation-conditional interval calibration (widening bands; conformal per regime) | **v0.5** | Directly after #263/#264 |
| 9 | Clear-sky as the zero-data **floor** — extend [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168), don't duplicate it | **v0.3** for the clear-sky primitive; the feature use stays with #168 in v0.5 | #168 already delivers clear-sky irradiance as a *feature*. Only the floor/ceiling framing is new. Pull the shared computation forward so the scenario suite has something to degrade *to* |
| 10 | Cost-per-experiment instrumentation (H2, H4) | **v0.5** | Piggybacks on aws-costs machinery |
| 11 | Weather-blind guarantee: outage-shaped training augmentation (§3.2 option A) | **v0.5**, was v0.9 | Promoted because "never worse than incumbent" now depends on it |
| 12 | Missingness contract on `BaseForecaster` — each family declares how it degrades | **v0.9**, note on [#362](https://github.com/openclimatefix/nged-substation-forecast/issues/362) | Forces the NN spike to answer the question rather than discover it in v2 |
| 13 | **Extend [#424](https://github.com/openclimatefix/nged-substation-forecast/issues/424)** — the `live_forecasts` asset check — to also report **NWP age at forecast time**, WARN severity, non-blocking | **v0.3** | Not a new issue: #424 already exists ("catch if the forecast asset succeeds but writes invalid forecasts or none at all"). It is the missing third check — every production asset has one except the one NGED consumes. Needs the degradation dimension added, a WARN/non-blocking severity decision, an epic, and project fields (it currently has none) |
| 14 | Make `live_forecasts` **degrade rather than raise** when NWP is absent or out of coverage — keep the `trained_ids` raise | **v0.3**, **after 13** | §4.1 divergences 1–3. Ordering is load-bearing: degrading before the check exists converts a loud failure into a silent one |

Plus: extend remaining v0.6/v0.7 warning types as sub-tasks of the existing epics rather than new
issues; add a note to [#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423)
that the R&D/production tag is the mechanism behind the fail-fast/fail-forward asymmetry.

**Existing issues this plan should attach to rather than duplicate** — checked 2026-08-04:

| Existing | Relationship |
|---|---|
| [#424](https://github.com/openclimatefix/nged-substation-forecast/issues/424) | *Is* plan item 13. Currently unlabelled beyond `enhancement`, no epic, no project fields |
| [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168) | Already delivers clear-sky irradiance; plan item 9 extends it |
| [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) | `nged_incumbent` baseline — **blocks T1.2** |
| [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161) | NWP ingestion validation checks — sibling of item 13 on the ingest side |
| [#420](https://github.com/openclimatefix/nged-substation-forecast/issues/420) | Silencing warnings for dead series — warning fatigue, directly downstream of items 4/6/7 |
| [#423](https://github.com/openclimatefix/nged-substation-forecast/issues/423) | R&D/production asset tags — the asymmetry's mechanism |
| [#374](https://github.com/openclimatefix/nged-substation-forecast/issues/374) | "Add more data-validation functions" — **empty body**; the natural home for the missing-vs-wrong distinction (§1) |

### 7.3 A cross-cutting label, not a new epic

This work deliberately spans v0.2–v0.9, and CLAUDE.md requires epics to map 1:1 to milestones. So:
an **`inherent-stability` label** across all the issues above. `gh issue list --label
inherent-stability` then gives a single legible workstream view without breaking the epic↔milestone
mapping. Cheap, and it is the artifact that makes the argument concrete for a sceptic.

### 7.4 Scope caution

Of these fifteen, items 4, 6 and 7 are on the critical path to a contractual v1.0 deliverable, and
items **13 and 14 are production-correctness fixes rather than quality work** — §4.1 divergence 1
is a live hard-failure mode that would cut NGED off entirely during an extended NWP outage. The
rest are quality and stability work.

So the scope widening is smaller than it first appears, and part of it is arguably overdue rather
than new. It **is** still a widening against the 2026-07-01 "live service first" reprioritisation,
and that call is Jack's — but items 13 and 14 sit *inside* "live service first" rather than against
it.

Mitigating this: per [§7.6](#76-effort-and-phasing--groundwork-now-or-the-whole-thing), the v0.3
tranche (0–5 and 9) is small — mostly pure functions over frames plus two schema decisions. The
genuinely large items (8, 11) are blocked on v0.5's quantile work regardless, so putting them later
costs nothing.

### 7.5 Does this change the Dagster-vs-Airflow assessment?

**Yes — three surgical edits, no change to the verdict.** Option C (stay on Dagster) still stands,
and nothing today weakens it. But today reweights one existing argument, adds a new cost to
Option B, and — most interestingly — makes the page's own central premise falsifiable.

**Edit 1 — reweight the asset-checks row.** The comparison table already carries:

> Asset checks — non-blocking WARN … | Partial — the capability exists as tasks; the non-blocking
> severity and check-status surface do not

That row is currently filed as an *observability* nicety. After today it is a **design-principle**
row. Non-blocking WARN checks are the mechanism by which the whole system stays fail-operational
while still telling the truth. In Airflow, data-quality checks are ordinary tasks and **blocking by
default** (as of 3.3.0): to get fail-open behaviour you must write every check as a task that
deliberately never fails and reports out-of-band. That makes inherent stability depend on developer
discipline *on every single check*, rather than on a first-class `severity` / `blocking` flag.

The edit: keep the row, but link it to `architecture/inherent-stability.md` and add a sentence
saying the gap is architectural rather than cosmetic. This is a genuine strengthening of the case
for Dagster that the page does not currently make.

**Edit 2 — a new cost for Option B.** Option B ports *only the live service* to Airflow. Today's
discussion shows that is precisely the half where the non-blocking check primitive matters most:
`power_data_is_fresh` and `nwp_has_no_unexpected_nulls` are both production-side, and the warning
channel to NGED originates there. So Option B would move the fail-open half of the system onto the
orchestrator that lacks a fail-open check primitive, while leaving the R&D half — which *wants*
fail-fast, and would be perfectly happy with blocking tasks — on Dagster. That is exactly backwards.

This belongs in Option B's "Against" list. It does not kill Option B (a handover signal from NGED
would still dominate), but it is a real cost that is currently unlisted.

**Edit 3 — a new trigger under "What would change this assessment".** The page's central argument
rests on experiment volume: *"At our experiment volume — hundreds of runs a month, with routine
re-runs of old experiments — small per-experiment friction compounds into a real tax."* Today we
turned that volume claim into **H2**, with a threshold and a measurement.

That cuts both ways, and honesty requires saying so: **if H2 is falsified — if we measure 20
experiments a month rather than 200 — the central argument for Dagster weakens materially.** A new
bullet should say so. This is the most interesting consequence of the hypotheses framing: it makes
an architectural decision re-testable rather than permanent, which is exactly what the page's
"documented seam" framing is for.

**Conditional fourth edit, depends on [decision 2](#8-decisions-needed-from-jack).** If the failure
scenarios become part of the *partition key* (`{experiment}__{fold}__{scenario}`) rather than a
column in the metrics table, the all-time partition-status catalog gets ~10× more cells — which
strengthens the existing "Airflow has the data model but no partition-status UI" gap considerably.
If scenarios stay a metrics *column* (the recommended option), partition counts are unchanged and no
edit is needed. Worth resolving decision 2 before touching this page.

### 7.6 Effort and phasing — groundwork now, or the whole thing?

**Answer: implement nearly all of it early. The split falls out of dependencies, not effort.**

The expensive items are not expensive-and-deferred by choice — they are *blocked* by the v0.5
quantile pipeline. And the items that must be early to be worth anything are the cheap ones. So
there is no real trade-off to make.

| Issue | Size | Why |
|---|---|---|
| 0. Intervention log | **XS** | A markdown file, a cause taxonomy, one runbook line |
| 1. Degradation smoke-tests | **S** | Trivial once #2 exists; pure functions, no MLflow, CI-fast |
| 2. Failure-scenario suite | **S** | Pure transforms over an `AllFeatures` frame — null NWP columns, age the lags, shift `nwp_init_time`. The **design** of the scenario vocabulary is the work; the code is small |
| 3. Score leaderboard under scenarios | **M** | Cost is N× *predict* + N× metrics, **not** N× train — you train once and predict per scenario. Tractable. Schema design is the real work |
| 4. `power_forecast_warnings` Phase 1 | **M** | New Delta table + writer, but only two warning types |
| 5. `promoted_model` rollback | **S** | The forward path already exists; this is mostly config + runbook |
| 9. Clear-sky envelope | **M** | Well-trodden (solar position + clear-sky model); also useful as a feature |
| 6, 7. Warning tables Phase 2 + health history | **M–L** | Contractual, and genuinely gated on v0.4 cleaning |
| 8. Degradation-conditional calibration | **L** | Real ML work. **Blocked** on quantile output (#263/#264, v0.5) |
| 11. Weather-blind training augmentation | **L** | Real ML work. **Blocked** on #2 and #3 existing to evaluate against |
| 12. `BaseForecaster` missingness contract | **S** | But only meaningful once a second model family exists (the v0.9 NN spike) |
| 13. `live_forecasts` check (#424) | **S** | Row-count and validity assertions plus an NWP-age metadata field. Reuses the `_to_asset_check_result` shape already established twice |
| 14. `live_forecasts` degrades not raises | **S–M** | Deleting two raises is trivial; deciding *what* a no-NWP forecast contains is the real work, and overlaps item 11 |

So: **items 0–5, 9, 13 and 14 can all land in v0.3, and together they are roughly one focused chunk
of work** — not a milestone-sized programme. Items 8 and 11 land in v0.5 because that is when their
prerequisite exists, and item 12 in v0.9 because that is when the NN spike happens.

**Within v0.3 the internal order is constrained**: 13 before 14 (§4.1), and
[#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) before 3 (T1.2 needs
the incumbent baseline to score against).

#### The two things to front-load, because retrofitting them is painful

This is the part worth getting right now even if everything else slips.

1. **Add the scenario dimension to the metrics schema immediately — populated with a single
   `"none"` value at first.** If the column is added later, every historical leaderboard metric
   either needs backfilling or the leaderboard carries a discontinuity. Adding a column now that
   holds one value costs almost nothing and removes a migration entirely.

2. **Design the scenario vocabulary carefully before writing any of it, and treat it as a contract.**
   It becomes an enum stamped into metrics rows, so changing it later invalidates historical
   comparisons — exactly the "re-runnability" property `requirements.md` says is load-bearing. Spend
   the effort on the vocabulary; the transforms themselves are easy to write and easy to change.

A third, smaller one: **no new column is needed on `PowerForecast`.** A consumer can derive the
degradation regime from `nwp_init_time` versus `power_fcst_init_time`, both of which already exist.
The in-band signal is the quantile spread (v0.5) plus this derivable provenance — the warnings table
carries the rest.

#### Revised recommendation

Do **not** split groundwork-now / implementation-later. Do items 0–5, 9, 13 and 14 in **v0.3**, with
the two schema decisions front-loaded, and accept that 8 and 11 arrive in v0.5 on their own
dependency schedule. That gets Jack's stated goal — *all our ML experiments validated against this*
— from v0.5 onward, which is when the experiments that matter actually start.

---

## 8. Decisions needed from Jack

1. **Is v0.3 the right home for the failure-scenario suite?** It is a real widening of v0.3
   (currently leaderboard + baselines + production monitoring), and it delays v0.5. The alternative
   is to let v0.5 run on clean-data skill and retrofit scenario scoring afterwards — cheaper now,
   but every v0.5 experiment must then be re-judged. *Recommendation: take the delay — and per
   [§7.6](#76-effort-and-phasing--groundwork-now-or-the-whole-thing) the delay is small, because
   the v0.3 tranche is mostly pure functions plus two schema decisions.*

2. **How do failure scenarios fit the evaluation model?** `EVALUATION_SCOPES` is currently
   `("leaderboard", "production_monitoring", "ad_hoc")`, with `EvalScopeType` a deliberately
   *narrower* `Literal["leaderboard", "ad_hoc"]` — the subset the asset handles today, documented to
   expand when Phase 8 lands. Any new scope must follow that established two-name pattern. Either a
   fourth scope, or a new **dimension** within `leaderboard` (every experiment scored *n* times,
   once per scenario).
   *Recommendation: the dimension, as a metrics **column**, not part of the partition key* — it
   makes degradation behaviour a first-class property of every experiment rather than an opt-in
   exercise, and it leaves partition counts unchanged. **This decision has a knock-on to
   [§7.5](#75-does-this-change-the-dagster-vs-airflow-assessment)**: putting scenarios in the
   partition key would ~10× the catalog and change the Airflow comparison. Resolve this before
   editing `why-dagster-not-airflow.md`.

3. **Does the inherent-stability page cover the whole system, or production only?** The co-founder
   audience cares about production, but omitting the R&D asymmetry makes the page read as "we never
   fail", which is the wrong instruction for future Claude. *Recommendation: one page,
   production-first in ordering, R&D as a clearly-marked contrasting section near the end.*

4. **§3.2 — do we commit to the weather-blind guarantee?** If "never worse than the incumbent" is to
   be a load-bearing claim rather than a hope, issue #11 moves into v0.5 and we accept the training
   cost. If it stays aspirational, it drops back to v0.9. *Recommendation: commit — it is the
   strongest sentence in the whole philosophy and it should not be unbacked.*

5. **Six hypotheses, or three?** H4–H6 are proposals only.
