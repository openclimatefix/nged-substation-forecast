# Handover to NGED

> **Status: 🚧 Planned.** This page is the design record for the post-NIA operating model NGED
> stated as their preference on **2026-07-14**: NGED running Flexpectation themselves, on
> NGED's own AWS account, after the NIA project ends. This is a preference, not yet a
> commitment — NGED's DSO, Cyber, and IT&D teams still need to sign off — but we design for it
> from now on. The requirement itself is recorded in
> [Requirements → Operating model & handover](../background/requirements.md#operating-model-handover);
> this page holds the engineering consequences and the handover workstreams. Epic:
> [#309](https://github.com/openclimatefix/nged-substation-forecast/issues/309).

## What this changes (and what it doesn't)

For the remainder of the NIA project, nothing about the milestone arc changes: OCF develops and
runs Flexpectation on OCF's AWS account, exactly as [the roadmap](index.md#milestones) already
plans. What changes is a **standing design constraint** on everything we build from now on:

**The service must be operable day to day by a non-expert at NGED.** The operator's scarce
skill is ops, not Python — if the service is designed well, the day-to-day maintainer should
never touch Python at all. Every routine action must reduce to "look at a dashboard, click a
button in the Dagster UI, or follow a runbook". Anything that can't be reduced to that is, by
definition, OCF's job, done on a scheduled cadence (e.g. quarterly maintenance windows) rather
than reactively.

Several decisions already made serve this constraint well, and this page makes that connection
explicit so we don't accidentally undo them:

- **The champion model is baked into the container image** with no MLflow (or any tracking
  server) on the production hot path — see
  [Production Deployment — Design](../architecture/production-deployment.md). Under the
  handover model this is a feature twice over: there are fewer runtime moving parts to break,
  and the model simply *freezes* between OCF's scheduled interventions.
- **Replay mode** means a missed slot is recovered by a one-click UI backfill, not by an
  engineer reconstructing state — see
  [Running live forecasts end-to-end](../live_service/dagster-workflow.md).
- **Promotion is rebuild + redeploy**, auditable via image tags — no live mutable model
  registry for an operator to mis-drive.
- **No static AWS keys** (IAM roles throughout) removes a whole class of credential-expiry
  incidents, and matches the constraints corporate AWS environments typically impose anyway.
- **The pipeline runs end-to-end on a laptop** — the live schedules were dress-rehearsed
  locally under `dg dev` before any AWS compute existed, and the standing preference is
  portable application logic over cloud-native glue (e.g. no EventBridge rules) wherever a
  portable option exists. Portability is what makes the service cheap to move into NGED's
  account — or anyone else's.
- **Lenient uptime requirements**: nothing very bad happens if the service misses a day,
  because NGED can always read the previous forecasts from S3 (forecasts extend 14 days
  ahead). Recovery can therefore always be "next business day, via runbook", never "2am page".

The honest caveat: the failures a non-expert can't handle mostly live at the *boundaries*, not
in our code — an NWP provider changing formats, NGED SCADA feed schema drift, credential and
certificate expiry, and AWS account plumbing. The workstreams below are aimed squarely at
those boundaries.

## Workstreams

### 1. The operator contract

Write an explicit, short **operator contract**: an enumeration of every action the NGED
operator is ever expected to take — acknowledge an alert, backfill a missed slot via replay
mode, restart the daemon, rebuild the control-plane box, escalate to OCF. Keep it to roughly
**ten items or fewer**, each one a documented button-press or a single command with a runbook
page in [`docs/live_service/`](../live_service/index.md).

Everything *not* on that list — model promotion, dependency upgrades, schema changes,
infrastructure changes — is OCF's job by definition, handled on a scheduled maintenance
cadence (and, post-NIA, under whatever support arrangement is agreed).

The existing `docs/live_service/` runbooks are the natural home for this material, but they
are currently written for *us* (Python-literate researchers). Before handover they need an
editing pass with the NGED operator as the audience, plus a top-level "operator contract" page
that indexes them.

### 2. Alert on absence, not just failure

Per-task failure alerts miss whole
classes of silent failure: a hung daemon, a full disk, an expired credential, a schedule that
simply stopped firing. The fix is a **dead-man's switch**: an alarm that fires when *no
successful forecast has landed in N hours* (e.g. 8 hours, i.e. one missed 6-hourly slot plus
margin), regardless of why. The planned mechanism is **Sentry cron monitoring**
([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)): each successful
run checks in with Sentry, and Sentry alerts on a missed check-in — Sentry sits outside the
service being watched (a dead daemon simply stops checking in), and check-in pings are plain
portable code. Details:
[the dead-man's switch](live-service.md#alert-on-absence-the-dead-mans-switch). The handover
consideration: the Sentry account is OCF's today, so at handover the alert routing (and
possibly the account itself) moves to NGED.

The [production monitoring plan](live-service.md#production-monitoring) already sketches a
"no fresh forecast" staleness alarm; this workstream promotes it from a nice-to-have to the
**primary** alert, because it is the one alert whose false-negative rate a non-expert operator
cannot compensate for.

Every alert — dead-man's switch and per-task alike — must link directly to a runbook that ends
in either a specific operator action or "escalate to OCF". An alert without a runbook is a bug
in the operator contract.

### 3. De-pet the control-plane box

The always-on EC2 control-plane box ([Option B](live-service.md#aws-architecture)) is the
riskiest element under a non-expert operating model: an unattended pet VM accumulates entropy —
disks fill, instances get retirement notices, OS patches drift. Mitigations, roughly in build
order:

- EC2 auto-recovery alarm and instance status-check alarm (some of this is already sketched in
  the Option B plan).
- Log rotation and scheduled disk-cleanup jobs on the box.
- Most importantly: a **tested, unattended rebuild-from-scratch script**, so the runbook
  answer to "the box is sick" is *destroy and recreate*, never *diagnose*. This is the point
  at which infrastructure-as-code stops being premature complexity and becomes the thing that
  lets a non-expert redeploy safely.

### 4. Infrastructure-as-code, portable to NGED's account

The live-service plan already defers infra-as-code to
[Access-phasing Stage 2](live-service.md#access-phasing) — that sequencing stands. What the
handover requirement adds:

- **By handover time, IaC is mandatory, not optional.** The rebuild-from-scratch runbook
  (workstream 3) and the deployment into NGED's account (workstream 5) both depend on it.
- **The IaC must be account-portable**: no OCF-specific resource names, account IDs, or
  network assumptions baked in. Deploying into a second AWS account should be "set variables,
  apply".
- The open [Terraform-vs-CDK question](live-service.md#deployment-workstream-3-aws-infrastructure)
  gains a new input: what NGED's infrastructure teams already know and are allowed to run
  matters as much as what suits OCF. Ask them before deciding.

The possible **hybrid model** (see
[Requirements](../background/requirements.md#operating-model-handover)) — NGED running the
production instance while OCF runs a second instance for development, other DNOs, or
commercial products — makes account-portability doubly valuable: the second deployment of the
same IaC is OCF's own.

### 5. Probe NGED's AWS landing zone early

NGED's corporate AWS environment is the biggest unknown, and the one item on this page that
should start **well before** the final months of the project. Corporate DNO cloud environments
commonly impose service control policies, mandatory patching/security agents, restricted
egress, and bans on long-lived credentials — and **Tailscale specifically may not survive
NGED's security review**. That matters because in the current design
[the network layer *is* the auth layer](live-service.md#access-phasing): none of the web UIs
(Dagster, MLflow, Marimo) has built-in authentication, so if Tailscale is prohibited, the
access design needs an NGED-compatible replacement (e.g. their VPN + private subnets, or an
SSO-fronted proxy), not just a substitution.

Concrete steps:

- Raise landing-zone constraints with NGED early: what can and can't run in their account,
  what network ingress/egress is permitted, how their teams authenticate to internal web UIs.
  These conversations overlap with the internal sign-off NGED needs anyway — their DSO,
  Cyber, and IT&D teams must approve the operating model before NGED can commit to it — so
  they double as progress on turning the stated preference into a concrete answer.
- Clarify **who the operator actually is**. NGED has IT/infrastructure teams; the realistic
  split may be a domain person doing the Dagster-level operating while their infrastructure
  team owns OS- and AWS-level issues. That split changes what the runbooks need to cover (and
  who game-day training targets).
- Stand up a **staging copy in NGED's account well before handover** — discovering in the
  final months that our networking approach is prohibited would be painful.

### 6. Game days

Before handover, run deliberate failure exercises with the actual NGED operator, using only
the runbooks: break the NWP feed, fill the disk, kill the daemon, expire a credential, let a
forecast slot get missed. The operator recovers each one unaided (or the runbook gets fixed).
Game days find documentation gaps faster than any amount of review, and they double as
operator training.

### 7. Organisational prerequisites (recorded so they're not forgotten)

These are not engineering workstreams, but NIA projects most often fail at the transition to
business-as-usual for exactly these reasons, so they are recorded here alongside the technical
work:

- A **named owner at NGED** with allocated time to operate the service. Our current NGED
  contacts are very engaged, and if they stay the owner question answers itself — the real
  risk is staff turnover. The mitigation is to institutionalise rather than rely on
  individuals: everything needed to operate lives in the written runbooks (no tribal
  knowledge), and the game days train more than one person.
- A **budget line** at NGED for the AWS spend and for any OCF support retainer.
- A written support agreement defining what OCF does post-NIA (scheduled maintenance,
  emergency fixes, model updates) — the details are TBD, but "in writing" is the requirement.
  This also hedges the staff-turnover risk above: an agreement survives the departure of the
  individuals who championed it.

## Timing and decision gates

- **Now → late NIA project**: OCF develops and runs the service on OCF's AWS account. The
  operator-contract constraint applies to new design work from today; workstream 5
  (landing-zone probing) starts early; the rest of the workstreams land alongside the v1/v2
  milestones they depend on.
- **Scale gate**: we will not know whether the service is truly hand-over-able until OCF has
  run the full v2 service (~2,500 time series) for a few months. NGED has accepted this.
- **Last few months of the NIA project**: progressively hand control to NGED, support them as
  they get up to speed, run the game days. NGED then decides whether to run it themselves.
- **Post-NIA**: OCF is no longer on call; NGED handles day-to-day operations. OCF may continue
  developing the software and models (the hybrid model), possibly under a retainer — all TBD.
