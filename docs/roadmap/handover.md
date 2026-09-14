# Handover to NGED

> **Status: 🚧 Planned.** The working assumption is that, after the Network Innovation Allowance
> (NIA) project, NGED runs the Flexpectation service on its own AWS account, subject to NGED's
> normal governance. This page is the design record for that operating model, and we design for
> that operating model from now on. The requirement itself is recorded in
> [Requirements → Operating model & handover](../background/requirements.md#operating-model-handover);
> this page holds the engineering consequences and the handover workstreams. Epic:
> [#309](https://github.com/openclimatefix/nged-substation-forecast/issues/309).

## What this changes (and what it doesn't)

For the remainder of the NIA project, nothing about the milestone arc changes — see
[the roadmap](index.md#milestones) for the v1/v2 plan and [Requirements → Operating model &
handover](../background/requirements.md#operating-model-handover) for the planned phasing.
What changes is a **standing design constraint** on everything we build from now on:

**NGED staff who did not develop the code must be able to run the service day to day, working
from the runbooks.** The day-to-day skill the service calls for is operations, not Python. If the
service is designed well, the day-to-day operator never needs to touch Python at all. Every
routine action must reduce to "look at a dashboard, click a button in the Dagster UI, or follow a
runbook". During the NIA project, any action that can't be reduced to those three actions is
OCF's job, done on a scheduled cadence (e.g. quarterly maintenance windows) rather than
reactively.

Several decisions already made serve this constraint well, and this page makes that connection
explicit so we don't accidentally undo them:

- **The champion model is baked into the container image** with no MLflow (or any tracking
  server) on the production hot path — see
  [Production Deployment — Design](../architecture/production-deployment.md). Under the
  handover model this is a feature twice over: there are fewer runtime moving parts to break,
  and the model simply *freezes* between model updates.
- **Replay mode** means a missed slot is recovered by a one-click UI backfill, not by an
  engineer reconstructing state — see
  [Operating the live service](../live_service/operations.md).
- **Promotion is rebuild + redeploy**, auditable via image tags — no live mutable model
  registry for an operator to mis-drive.
- **No static AWS keys for OCF's own resources** (IAM roles throughout) removes a whole class
  of credential-expiry incidents. The two static credentials that remain sit at the boundary with
  NGED — the NGED read-access user and the source-bucket credentials.
  [Workstream 5](#5-confirm-ngeds-cloud-and-security-standards-early) checks whether NGED's
  standards permit them.
- **The pipeline runs end-to-end on a laptop** — the live schedules were dress-rehearsed
  locally under `dg dev` before any AWS compute existed. And the standing preference is
  portable application logic over cloud-native glue (e.g. no EventBridge rules) wherever a
  portable option exists. Portability is what makes the service cheap to move into NGED's
  account — or anyone else's.
- **Lenient uptime requirements** mean recovery can always be "next business day, via
  runbook", never "2am page" — see
  [Requirements → Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design)
  for why an outage costs so little (the 14-day forecast horizon, S3 delivery decoupled from
  compute, and NGED's existing forecasting tools).

**The failures the runbooks cannot resolve mostly live at the *boundaries*, not in our code.**
Those boundaries are an NWP provider changing formats, changes to the format of upstream data
feeds, credential and certificate expiry, and AWS account plumbing. We expect the most common
failure mode to be input data that arrives incomplete or wrong, whose fix lives upstream of the
forecasting service rather than in the service itself. The workstreams below are aimed squarely
at those boundaries.

## Workstreams

### 1. The operator contract

**Write a short, explicit operator contract that lists every action the NGED operator is ever
expected to take.**

- Acknowledge an alert.
- Backfill a missed slot via replay mode.
- Restart the daemon.
- Rebuild the control-plane box.
- Escalate to OCF.

Keep the contract to roughly **ten items or fewer**, each one a documented button-press or a single
command with a runbook page in [`docs/live_service/`](../live_service/index.md).

Everything *not* on that list — model promotion, dependency upgrades, schema changes, infrastructure
changes — is OCF's job during the NIA project, handled on a scheduled maintenance cadence. After
the NIA project, the [written support agreement](#7-organisational-prerequisites) sets who does
that work. Model re-training sits in this category too: the re-training pipeline is automated
end-to-end, so "re-training" means triggering and reviewing an automated run on a regular cadence.

One **optional tier** sits between "follow a runbook" and "escalate to OCF", for the more
mysterious bugs that fall outside the contract but that NGED may want to try fixing themselves:

1. Reproduce the issue locally — the pipeline runs end-to-end on a laptop, precisely to make
   this possible.
2. Propose a fix — using AI coding tools (e.g. Claude Code) where NGED's policies allow them.
3. Run the full test suite, then put the fix through NGED's change-control process. Where the
   support agreement covers code review, OCF reviews the fix before it reaches production.
4. If any of that fails, escalate to OCF.

"Escalate to OCF" is always an acceptable answer. The bug-fixing tier is nonetheless where
most of the in-person handover training (workstream 6) is expected to focus, because routine
operation should need very little training if the operator contract is doing its job.

The existing [`docs/live_service/`](../live_service/index.md) runbooks are the natural home for this
material, but the docs are currently written for *OCF* (Python-literate researchers). Before
handover they need an editing pass with the NGED operator as the audience, plus a top-level
"operator contract" page that indexes them.

### 2. Alert on absence, not just failure

**Per-task failure alerts miss whole classes of silent failure, so the service also needs a
missed-check-in alarm.** A hung daemon, a full disk, an expired credential, or a schedule that
simply stopped firing raises no per-task alert. A missed-check-in alarm (Sentry's cron-monitoring
terminology) fires when *no successful forecast has landed in N hours* (e.g. 8 hours, i.e. one
missed 6-hourly slot plus margin), regardless of why. That alarm is
built and running on **Sentry cron monitoring**: each successful `live_forecasts` run checks in with
Sentry, and Sentry alerts on a missed check-in — Sentry sits outside the service being watched (a
dead daemon simply stops checking in), and check-in pings are plain portable code. The as-built
mechanism is [Send telemetry to
Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence).
What handover still has to settle: the Sentry account is OCF's today, so the alert routing (and
possibly the account itself) moves to NGED.

The [production monitoring plan](live-service.md#production-monitoring) already sketches a "no fresh
forecast" staleness alarm; this workstream promotes it from a nice-to-have to the **primary** alert,
because it is the one alert whose false negatives an operator working from the runbooks has no
other way to notice.

Every alert — missed-check-in and per-task alike — must link directly to a runbook that ends in
either a specific operator action or "escalate to OCF". An alert without a runbook is a bug in the
operator contract.

### 3. Make the control-plane box rebuildable from scratch

**The always-on EC2 control-plane box ([the accepted option](live-service.md#aws-architecture)) is
the riskiest element when the operator works from the runbooks rather than from knowledge of the
box's internals.** An unattended, long-lived virtual machine accumulates faults: disks fill,
instances get retirement notices, and operating-system patches drift. Mitigations, roughly in
build order:

- EC2 auto-recovery alarm and instance status-check alarm (some of this is already sketched in
  the accepted-option plan).
- Log rotation and scheduled disk-cleanup jobs on the box.
- Most importantly: a **tested, unattended rebuild-from-scratch script**, so the runbook
  answer to "the box is sick" is *destroy and recreate*, never *diagnose*. A tested rebuild
  script is the point at which infrastructure-as-code stops being premature complexity and
  becomes the mechanism that lets an operator working from the runbooks redeploy safely.

### 4. Infrastructure-as-code, portable to NGED's account

The live-service plan already defers infra-as-code to
[Access-phasing Stage 2](live-service.md#access-phasing) — that sequencing stands, and the work
is tracked as [#326](https://github.com/openclimatefix/nged-substation-forecast/issues/326).
What the handover requirement adds:

- **By handover time, IaC is mandatory, not optional.** The rebuild-from-scratch runbook
  (workstream 3) and the deployment into NGED's account (workstream 5) both depend on it.
- **The IaC must be account-portable**: no OCF-specific resource names, account IDs, or
  network assumptions baked in. Deploying into a second AWS account should be "set variables,
  apply".
- The open [Terraform-vs-CDK question](live-service.md#deployment-workstream-3-aws-infrastructure)
  gains a new input: the tools NGED's engineers already use, and the tools NGED's standards
  permit, matter as much as what suits OCF. Agree the choice with NGED before making it.

The possible **hybrid model** (see
[Requirements](../background/requirements.md#operating-model-handover)) — NGED running the
production instance while OCF runs a second instance for development or other distribution
network operators — makes account-portability doubly valuable: the second deployment of the
same IaC is OCF's own.

### 5. Confirm NGED's cloud and security standards early

**OCF's access design will need to fit NGED's cloud and security standards, so OCF needs to
confirm those standards with NGED early** — well before the final months of the project.
Corporate cloud environments commonly impose service control policies, mandatory patching,
mandatory security agents, restricted egress, and bans on long-lived credentials. The access
design is the part of the service those standards bear on most, because in the current
design [the network layer *is* the authentication layer](live-service.md#access-phasing).
None of the web UIs (Dagster, MLflow, and Marimo) has built-in authentication, and Tailscale
is what restricts who can reach them. If NGED's standards call for a different network
layer, the access design needs an NGED-compatible replacement as a whole (e.g. NGED's
virtual private network plus private subnets, or a proxy fronted by single sign-on).

Concrete steps:

- Confirm with NGED what can run in NGED's AWS account, which network ingress and egress are
  permitted, and how NGED staff authenticate to internal web UIs.
- Confirm whether NGED's standards permit long-lived access keys, because the NGED read-access
  user and the source-bucket credentials both depend on them (see
  [Setting up the live service on AWS](../live_service/aws.md)).
- Agree with NGED which teams own which parts of operation: the Dagster level, the operating
  system, and the AWS account. That split changes what the runbooks need to cover, and whom the
  game days train.
- Stand up a **staging copy in NGED's account well before handover**, so that any mismatch
  between our networking approach and NGED's standards surfaces early.

### 6. Game days and in-person training

**Before handover, run deliberate failure exercises with the NGED staff who will operate the
service, using only the runbooks.**

- Break the NWP feed.
- Fill the disk.
- Kill the daemon.
- Expire a credential.
- Let a forecast slot get missed.

The operator recovers from each failure unaided, or the runbook gets fixed.
Game days find documentation gaps faster than any amount of review, and they double as
operator training. The game days train more than one person, and everything needed to operate the
service lives in the written runbooks rather than in any one person's head.

Game days sit alongside an **in-person training visit**: OCF spending up to a week on-site with
the NGED team around handover time. Most of that training is expected to cover the bug-fixing
escalation tier described in workstream 1 rather than routine operation — the service should be
simple enough to run that routine operation needs far less than a week.

### 7. Organisational prerequisites

**The transition to business-as-usual needs planning of its own.** The three prerequisites below
are not engineering workstreams, but the handover depends on them, so they are recorded here
alongside the technical work:

- A **named service owner at NGED** with allocated time to operate the service.
- **Funding for running costs and support**: the AWS spend and whatever support the written
  agreement below covers.
- A **written support agreement** setting out who does scheduled maintenance, emergency fixes,
  and model updates after the NIA project.

## Timing and decision gates

- **The planned phasing is recorded in [Requirements → Operating model &
  handover](../background/requirements.md#operating-model-handover)**: OCF running the service
  through the NIA project, a scale gate at v2, and progressive handover in the final months.
  The phasing rests on the working assumption that NGED runs the service on its own AWS
  account after the NIA project. Workstream 5 (confirming NGED's cloud and security
  standards) is the one workstream that should start well before that final-months handover.
  The rest land alongside the v1/v2 milestones they depend on.
- **The operating model after the NIA project remains NGED's decision.**
