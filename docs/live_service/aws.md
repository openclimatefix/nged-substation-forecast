# Setting up the live service on AWS

Every step to stand up the v0.1 live service on AWS, in order, ending with the full stack
running unattended: data tables on S3 (Simple Storage Service, AWS's object store), the champion
model baked into a container image, an always-on control-plane box running Dagster behind
Tailscale, and 6-hourly forecasts executing on ephemeral Fargate (AWS's serverless container
compute) tasks. When you finish this page, you open the Dagster UI from your laptop over
Tailscale and watch forecasts land.

You can point a coding agent (e.g. Claude Code) at this page and have it drive the console steps
alongside you. It's also worth asking the agent to check your work as you go: after completing a
step (or a run of steps), ask it to verify what you just did using the `aws` CLI — e.g. "check
that I've completed all steps up to and including step 7" — rather than trusting a screenshot or
your own memory of which button you clicked.

This is one-time setup (per AWS environment). Day-to-day driving of the running service —
promoting a champion, backfilling a missed slot, inspecting a forecast — lives in
[Operating the live service](operations.md). What each `Settings` field means (the three storage
roots, the derive-from-root convention) lives in the
[Configuration reference](setup.md). Design rationale for *why* the deployment looks like this —
bake the model in at build time, an always-on control plane rather than EventBridge — lives in
[Production Deployment — Design](../architecture/production-deployment.md).

> **Scope: everything here is done by hand** — AWS console plus SSH; no infrastructure-as-code
> (Terraform, or CDK — AWS's Cloud Development Kit) yet.
> That's deliberate: this is Stage 1 ("solo, Tailscale only") of the
> [access-phasing plan](../roadmap/live-service.md#access-phasing), and infrastructure-as-code
> ([#326](https://github.com/openclimatefix/nged-substation-forecast/issues/326)) is scheduled
> to start at Stage 2, when team access adds enough moving parts to justify it.
> Sentry error telemetry and the missed-check-in alarm are wired (set the `SENTRY_*` vars in
> [Step 14](#step-14-configure-dagster-on-the-box)); only per-task failure emails (SNS) are still
> to come — see [the roadmap](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm).

## Step 1 — Create the S3 buckets

We split storage across **two** buckets so the five tables that form NGED's stable delivery
contract are physically separate from OCF's own working data, which may change shape at any time
with no notice. See [Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it)
for why this split exists, and [Delivery tables](https://openclimatefix.github.io/nged-substation-forecast/roadmap/delivery-tables/)
for exactly which five tables count as "delivery."

In the AWS console → **S3** → **Create bucket**, twice:

- **Region** `eu-west-2` (London) for both — keep every resource in one region so S3 ↔ compute
  transfer stays free. It isn't the cheapest region available (`eu-west-1` runs meaningfully
  cheaper for Fargate); see [Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it)
  for the price comparison and why `eu-west-2` is picked anyway, provisionally, pending NGED
  confirmation.
- **Names**: `nged-forecast-delivery` (the 5 NGED-facing tables) and `nged-forecast-internal`
  (NWP, raw power telemetry, forecast metrics, and everything else OCF's pipeline needs but
  hasn't promised to keep stable). Bucket names are globally unique; pick your own if these are
  taken.
- **Every other setting on the create-bucket form can stay at its console default:**
    - **Bucket namespace** — leave at **Global namespace**; do not switch to "Account Regional
      namespace", even though the console marks it "(recommended)". That option changes the
      bucket's ARN (Amazon Resource Name — AWS's canonical resource identifier) shape and scopes
      name-uniqueness to your account/region instead of globally, which the IAM (Identity and
      Access Management — AWS's permissions system) policies below (Step 2) and the
      `deltalake`/`object_store` machinery all assume it isn't.
    - **Object Ownership** (ACLs disabled) — access is controlled entirely by IAM/bucket policy;
      ACLs (legacy per-object access-control lists) stay off.
    - **Block Public Access** (all four boxes on) — nothing here is public.
    - **Bucket Versioning** (disabled) — not required by the app.
    - **Default encryption** (SSE-S3 — server-side encryption with S3-managed keys) — the app
      needs no more than this; **Bucket Key** only affects SSE-KMS (encryption via AWS KMS, the
      Key Management Service), so it's irrelevant here either way.
    - **Object Lock**, **Tags** — unused by the app.

No DynamoDB lock table is needed for either bucket. The `deltalake` version we use commits via
S3's native conditional-put, so concurrent-safe Delta writes work on plain S3 with no lock table
and no `AWS_S3_ALLOW_UNSAFE_RENAME` flag.

## Step 2 — Grant data access with IAM

OCF's own pipeline needs to read and write **both** buckets (internal tables during ingestion and
training; delivery tables at forecast time), so one IAM policy covers both:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::nged-forecast-delivery",
        "arn:aws:s3:::nged-forecast-delivery/*",
        "arn:aws:s3:::nged-forecast-internal",
        "arn:aws:s3:::nged-forecast-internal/*"
      ]
    }
  ]
}
```

**Create it once**, in the AWS console → **IAM** → **Policies** → **Create policy** → switch to the
**JSON** editor → paste the JSON above → **Next** → name it (e.g. `nged-forecast-read-and-write`) →
**Create policy**. IAM is global, so there's no region selector to worry about.

Then attach it to **whichever identity runs the code**:

- **Compute running on AWS** → attach the policy to that resource's **IAM role**. This page does
  that twice: the Fargate **task role** ([Step 7](#step-7-iam-roles-for-the-fargate-task)) and the
  control-plane box's **instance role** ([Step 11](#step-11-launch-the-control-plane-box)). Nothing
  else is needed: `object_store` (used by `delta-rs`) auto-discovers the role's temporary
  credentials and region at runtime, so all four `DATA_STORE_*` settings stay **empty** (see the
  [Configuration reference](setup.md#the-configuration-model)).
- **Your laptop, running the pipeline** — **not set up yet, deliberately**: for now, only AWS
  compute gets write access, so there's exactly one writer touching the buckets at a time. Running
  Dagster from both a laptop and AWS against the same tables risks two instances racing on the
  same Delta commits. If a one-off laptop write is ever needed (e.g. hand-patching bad data), it
  can reuse the same read/write policy above via its own **IAM user**, created the same way as the
  dashboard user below — just with `nged-forecast-read-and-write` attached instead of
  `nged-forecast-read-only`.
- **Your laptop, running only the dashboard** (the marimo apps at `packages/dashboard/`, e.g.
  `view_forecasts.py`) → it only ever
  reads, so give it a separate, read-only **IAM user** instead of reusing the read/write
  credential:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["s3:GetObject", "s3:ListBucket"],
          "Resource": [
            "arn:aws:s3:::nged-forecast-delivery",
            "arn:aws:s3:::nged-forecast-delivery/*",
            "arn:aws:s3:::nged-forecast-internal",
            "arn:aws:s3:::nged-forecast-internal/*"
          ]
        }
      ]
    }
    ```

    Same console flow as the read/write policy above for the policy: **IAM** → **Policies** →
    **Create policy** → **JSON** editor → paste the JSON above → **Next** → name it (e.g.
    `nged-forecast-read-only`) → **Create policy**. Then create the user: **IAM** → **Users** →
    **Create user** → name it (e.g. `nged-forecast-dashboard`) → leave **Provide user access to the
    AWS Management Console** unchecked (programmatic-only) → **Next** → **Attach policies
    directly** → select the policy → **Next** → **Create user**. Then its access key: open the
    user → **Security credentials** tab → **Access keys** → **Create access key** → use case
    **Application running outside AWS** → acknowledge the warning → **Create access key** → copy
    the **Access key ID** and **Secret access key** immediately (shown once) into the laptop's
    `.env` as `DATA_STORE_*` values (see the
    [at-a-glance table](setup.md#at-a-glance-which-settings-for-which-environment) for the exact
    shape). Losing this key can't corrupt any table, only leak read access to it.

## Step 3 — Pick and promote a champion model

Follow [Operating the live service: Step 1](operations.md#step-1-pick-a-champion-model)
and [Step 2](operations.md#step-2-materialise-promoted_model) to materialise
`promoted_model`. Promotion always happens **on your laptop** — the candidate models live in the
laptop's local MLflow file store — and populates `data/production_model/` on disk, the same
directory the image build below `COPY`s from.

## Step 4 — Build and verify the image

With a champion model on disk (Step 3), one script builds the image and smoke-tests it with
**zero network access** — the test that matters, since the entire point of baking the model in is
that production inference has no MLflow dependency at runtime:

```bash
scripts/build_and_verify_image.sh    # no arguments — everything is derived
```

The script always builds for **linux/arm64**, whatever the host is, because the deployment is
ARM end-to-end: the Fargate task definition declares ARM64
([Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition)) and the control-plane box
is a Graviton instance ([Step 11](#step-11-launch-the-control-plane-box)), so an amd64 image
would push fine but then fail every task launch with `image Manifest does not contain
descriptor matching platform 'linux/arm64 v8'`. On an x86 laptop the build and smoke test run
under QEMU emulation — noticeably slower than native, but correct. The script checks the
emulator is registered and prints the one-time fix
(`docker run --privileged --rm tonistiigi/binfmt --install arm64`) if it isn't.

> Note that, during the smoke test, **Dagster is EXPECTED halt with the exception**
> `_internal.TableNotFoundError: Local path
"/app/.venv/data/NWP" does not exist or you don't have access!`. That is correct behaviour.

The script builds the image tagged with the promoted model's run id, runs it offline, and prints
the container log with a pass/fail summary. (The smoke test's one-shot run uses a hard-coded,
arbitrary partition key — the offline run fails at the NWP lookup long before the slot's
validity could matter, so there is no key worth choosing.)
It hard-fails only if the runtime touches MLflow — the hermeticity guarantee worth automating —
and otherwise asks you to confirm by eye that the run loaded the model and failed *only* on
missing NWP data (expected: no data tables are mounted for this isolated test). The script header
documents every choice it makes and is the source of truth for the mechanics.

## Step 5 — Create the ECR repository

ECR (Elastic Container Registry) is AWS's private Docker-image store — where the image just built
gets pushed so AWS compute can pull it. In the [AWS
console](https://eu-west-2.console.aws.amazon.com) →
**[ECR](https://eu-west-2.console.aws.amazon.com/ecr)** → **Create repository** → private (same
`eu-west-2` region as the S3 buckets in [Step 1](#step-1-create-the-s3-buckets)):

- **Repository name**: `nged-forecast`, with **no namespace prefix**. The scripts hard-code this
  exact flat name — `scripts/push_and_deploy_image.sh` derives the remote URI as
  `<account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<tag>` — and it matches the local
  image tag `nged-forecast:<tag>` from [Step 4](#step-4-build-and-verify-the-image). A namespace
  is only an optional `prefix/` for grouping many repositories; adding one would break that
  derived URI.
- **Every other setting on the create-repository form can stay at its console default:**
    - **Image tag mutability** (Mutable) and **Mutable tag exclusions** (empty) — image tags here
      are already unique per promoted model (the run id's short prefix from
      [Step 4](#step-4-build-and-verify-the-image)), so nothing relies on retagging — but nothing
      needs immutability enforced, either.
    - **Encryption settings** (AES-256) — the console warns this can't be changed after creation,
      which is fine: the app has no KMS requirement, for the same reason the S3 buckets in
      [Step 1](#step-1-create-the-s3-buckets) stay on SSE-S3.
    - **Scan on push** (off) — the console marks this per-repository setting deprecated; scanning
      is now configured once at the *registry* level, which is done right after creating the
      repository (next bullet).
- Click **Create**, then turn on registry-level scanning so every push still gets free vulnerability
  scanning: **ECR** → **Private registry** → **Features & Settings** → [**Scanning** →
  **configure**](https://eu-west-2.console.aws.amazon.com/ecr/private-registry/edit-scanning) → keep
  the scan type at **Basic** ("basic" is free; Enhanced hands scanning to Amazon Inspector, which
  costs money) → either check the **Scan on push all repositories** check box _or_ add a filter of
  `nged-forecast`. This is a one-time registry setting, so it also covers any repository created
  later under a matching filter.

## Step 6 — Push the image to ECR

One script pushes the image built in [Step 4](#step-4-build-and-verify-the-image) and — on
later redeploys — points the ECS task definition at it:

```bash
scripts/push_and_deploy_image.sh    # no arguments — everything is derived
```

It takes no arguments by design, so nothing can be mistyped: the tag is derived from
`data/production_model/promotion.json` exactly as Step 4's build script derives it (so only an
image that was built and verified can be pushed), and the AWS account id comes from
`aws sts get-caller-identity`. The script logs Docker into ECR, tags, and pushes; then, if the
`nged-forecast` task-definition family already exists
([Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition)), it registers a new
revision pointing at the new image. If this is the first pass through the runbook then the family
doesn't exist yet — the script says so and exits cleanly, and the push is all this step needs. The
script header documents every choice it makes and is the source of truth for the mechanics.

## Step 7 — IAM roles for the Fargate task

Fargate tasks run inside ECS (Elastic Container Service — the AWS orchestrator that launches and
supervises containers), and they need **two** separate IAM roles, not one — they serve different
principals. Both roles are created with the same console flow — [**IAM** → **Roles** → **Create
role**](https://us-east-1.console.aws.amazon.com/iam/home?region=eu-west-2#/roles/create) → trusted
entity type **AWS service** → under **Use case**, search for and pick **Elastic Container Service**,
then select the **Elastic Container Service Task** radio button (the *Task* variant is what puts
`ecs-tasks.amazonaws.com` in the role's trust policy, so that ECS tasks can assume the role) →
**Next**. As with [Step 2](#step-2-grant-data-access-with-iam), IAM is global, so there's no region
selector to worry about. From the **Add permissions** page onwards the two roles diverge:

- **Task execution role**, `nged-forecast-task-execution-role` — used by the *ECS agent* itself,
  before your code ever runs: pulling the image from ECR, shipping container output to
  CloudWatch Logs, and injecting the secrets from
  [Step 8](#step-8-store-secrets-in-parameter-store).

    On the **Add permissions** page, search for and tick the AWS-managed
    **`AmazonECSTaskExecutionRolePolicy`** (covers the ECR pull + CloudWatch Logs) → **Next** →
    **Role name** `nged-forecast-task-execution-role` → **Create role**.

    Then add one small inline policy for the secrets — it's the *execution* role that reads
    them, not the task role, because the ECS agent resolves secrets before the container
    starts. Open the role you just created → **Permissions** tab → **Add permissions** →
    **Create inline policy** → switch to the **JSON** editor → paste the JSON below, replacing
    `<account-id>` with your 12-digit AWS account id (shown in the account menu at the top right of
    the console, or run this locally:
    `aws sts get-caller-identity --query Account --output text`) → **Next** →
    name it (e.g. `nged-forecast-read-ssm-parameters`) → **Create policy**:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": "ssm:GetParameters",
          "Resource": "arn:aws:ssm:eu-west-2:<account-id>:parameter/nged-forecast/*"
        }
      ]
    }
    ```

    (No KMS statement needed as long as the parameters use the default `aws/ssm` key.)

- **Task role**, `nged-forecast-task-role` — used by *your code* once it's running: this is what
  lets the container read and write **both** data buckets (delivery and internal). Run the whole
  create-role flow a second time (trusted entity **AWS service** → **Elastic Container Service
  Task**), but on the **Add permissions** page reuse the same S3 policy from
  [Step 2](#step-2-grant-data-access-with-iam): search for and tick the customer-managed
  `nged-forecast-read-and-write` policy → **Next** → **Role name** `nged-forecast-task-role` →
  **Create role**. Nothing else is needed — with this role attached, delta-rs' `object_store`
  auto-discovers temporary credentials at runtime, so `DATA_STORE_*` stays unset.

No static AWS keys anywhere in either role — the same IAM-role auto-discovery
[Step 2](#step-2-grant-data-access-with-iam) relies on for all compute running on AWS.

## Step 8 — Store secrets in Parameter Store

The container can't start without `NGED_S3_BUCKET_URL`, `NGED_S3_BUCKET_ACCESS_KEY`, and
`NGED_S3_BUCKET_SECRET` (`Settings` requires them at import — see the smoke test in
[Step 4](#step-4-build-and-verify-the-image)), and the deployed service genuinely uses them: the
hourly `power_time_series_and_metadata` schedule pulls fresh telemetry from NGED's bucket.

They are also the one credential in this deployment that can't come from an IAM role: NGED's
bucket lives in NGED's AWS account, so these are unavoidably static third-party keys. Don't
paste them into the task definition as plain-text environment values — anyone with ECS
describe access could read them there. Store them in **SSM Parameter Store** (SSM is AWS
Systems Manager; Parameter Store is its encrypted key-value configuration service) as
SecureStrings and let ECS inject them at container start:

In the AWS console → [**Systems
Manager**](https://eu-west-2.console.aws.amazon.com/systems-manager/home?region=eu-west-2) →
**Application Tools** → [**Parameter
Store**](https://eu-west-2.console.aws.amazon.com/systems-manager/parameters?region=eu-west-2) →
**Create parameter**, four times, in `eu-west-2` (same region as everything else):

- **Name** — one parameter each. The shared `/nged-forecast/` prefix is exactly what
  [Step 7](#step-7-iam-roles-for-the-fargate-task)'s execution-role policy grants access to,
  so a new secret added under the same prefix later needs no IAM change.
    - `/nged-forecast/nged-s3-bucket-url` — the URL of NGED's source S3 bucket, which the
      hourly `power_time_series_and_metadata` schedule pulls raw telemetry from (the
      `nged_s3_bucket_url` field of `Settings`).
    - `/nged-forecast/nged-s3-bucket-access-key` — the access-key ID for NGED's bucket
      (`nged_s3_bucket_access_key`) - a static credential issued by NGED, since the bucket
      lives in NGED's AWS account.
    - `/nged-forecast/nged-s3-bucket-secret` — the secret access key paired with NGED's
      access-key ID (`nged_s3_bucket_secret`).
    - `/nged-forecast/dagster-pg-password` — the password for the control-plane box's
      Postgres ([Step 14](#step-14-configure-dagster-on-the-box)). It's stored here because
      every launched run connects back to that Postgres to record its events, so the Fargate
      containers need it injected exactly like the NGED credentials.
- **Tier**: Standard (free; these values are tiny, nowhere near the 4 KB limit).
- **Type**: **SecureString**, with the default `aws/ssm` KMS key — using the default key is
  what lets Step 7's inline policy skip a `kms:Decrypt` statement.
- **KMS key source** and **KMS Key ID** — these two fields appear once **SecureString** is
  selected; leave both at their defaults: **My current account** and `alias/aws/ssm`.
  (`alias/aws/ssm` is the console's name for the default `aws/ssm` key the Type bullet just
  referred to.) The console shows a blue notice that the default AWS-managed key "cannot be
  shared with other AWS accounts, and all users in this AWS account and Region have access to
  the key" — both limitations are fine here: nothing outside this account ever reads these
  parameters, and having access to the *key* doesn't grant access to the *parameters* — reading
  them still requires the `ssm:GetParameters` permission that
  [Step 7](#step-7-iam-roles-for-the-fargate-task)'s policy grants only to the execution role.
- **Value**:
    - The three `nged-s3-bucket-*` values are copied from the matching
      `NGED_S3_BUCKET_*` lines of your local `.env`.
    - `dagster-pg-password` can't be copied from anywhere, because it doesn't exist yet:
      Postgres isn't installed until [Step 14](#step-14-configure-dagster-on-the-box), and
      when it is, *you* choose its password rather than receiving one. So mint it now: run
      `openssl rand -hex 24` in a terminal on your laptop (any machine with `openssl` works —
      the command just prints 48 random hex characters and touches nothing), and paste the
      output straight into this parameter's **Value** field. That's the whole minting step:
      this parameter *is* the authoritative copy, and there is nowhere else to record the
      password now — no need to keep it in a password manager or a local file. It gets read
      back in two places later: [Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition)
      injects it into every Fargate run as the `DAGSTER_PG_PASSWORD` environment variable,
      and in Step 14 you copy the same value (view it again in the Parameter Store console
      via **Show decrypted value**) into `~/nged-forecast/.env` on the box — whose
      `docker-compose.yml` hands it to Postgres as `POSTGRES_PASSWORD` on first start, which
      is the moment the password actually gets *set* on a real database.

There is no wiring to do in this step — that happens in the task definition
([Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition)), where each parameter is
referenced as a **ValueFrom** environment variable. The ECS agent, authenticated as the
*execution* role, fetches and decrypts the values just before the container starts; they never
appear in the task definition, the ECS console, or CloudWatch.

Parameter Store is picked over Secrets Manager deliberately: Standard parameters are free
(Secrets Manager is $0.40/secret/month), and these credentials don't need Secrets Manager's
flagship feature, automatic rotation — NGED's keys rotate on NGED's schedule, not ours. When
NGED does rotate them, just update the parameter values (and your local `.env`); running tasks
keep the old values until they next start, since injection happens once per container launch.

## Step 9 — Create the ECS cluster and Fargate task definition

First, a reminder of the deployment's shape, because from this step onwards it matters which half
of it you're building. The stack runs **two different kinds of compute**, on two different AWS
technologies:

- The **always-on control plane** — the Dagster daemon, webserver, code-location server, and
  Postgres — runs on a plain **EC2 virtual machine** (EC2 is Elastic Compute Cloud), launched by
  hand in [Step 11](#step-11-launch-the-control-plane-box) and managed with Docker Compose. ECS
  and Fargate play no part in running it. Its schedules are responsible for **every recurring
  job**, not just the forecasts: pulling fresh telemetry from NGED's bucket hourly, downloading
  the daily ECMWF ensemble NWP from Dynamical.org, and issuing each 6-hourly forecast — though
  the runs themselves execute on Fargate (next bullet). This is the **only compute in the deployment whose
  operating system we install and maintain ourselves**: [Step 11](#step-11-launch-the-control-plane-box)
  chooses its OS image (an Ubuntu Server AMI), and from then on the OS is ours to look after —
  installing software on it ([Step 12](#step-12-join-the-tailnet) and
  [Step 13](#step-13-install-docker-and-pull-the-image)) and keeping it patched (Ubuntu's
  `unattended-upgrades`, checked in [Optional hardening](#optional-hardening)). (Running the
  control plane on Fargate too would have removed even that OS burden — why we don't is covered
  in [Considered but rejected designs](../architecture/production-deployment.md#an-always-on-fargate-service-for-the-control-plane).)
  That maintenance never needs a scheduled outage agreed with NGED: forecasts are produced only
  every 6 hours, and NGED reads published forecasts straight from S3, so the gap between one
  forecast run and the next is a built-in maintenance window in which the box can be stopped,
  patched, and rebooted — see
  [Requirements → Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design).
- The **ephemeral workers** — each scheduled run: the hourly NGED telemetry ingest, the daily
  Dynamical.org NWP download, and the 6-hourly forecast — run as **ECS tasks on Fargate**: a
  container is created for one run and destroyed when it exits. The control plane *dispatches*
  these tasks but never executes one itself. Here there is **no
  operating system for us to install or maintain**: AWS owns and patches the machines Fargate
  tasks run on, and everything we are responsible for travels inside the container image from
  [Step 6](#step-6-push-the-image-to-ecr).

This step builds the scaffolding for the ephemeral half: the cluster its tasks launch into, and
the task definition describing how to run them. Why the compute is split this way is the
[orchestration design](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm).

Two AWS terms do a lot of work in this step, so it's worth being precise about how they relate.
**ECS** (Elastic Container Service) is AWS's container orchestrator: it takes a **task definition**
— a recipe naming the container image to run plus the CPU, memory, IAM roles, environment variables,
and secrets to run it with — and launches and supervises containers from that recipe. **Fargate** is
one of ECS's two "launch types" — the two ways of providing the compute the containers run on. The
other launch type (named "EC2") has ECS place containers onto EC2 instances that you provision and
patch yourself. In contrast, with Fargate, AWS conjures right-sized compute for each task when it
launches and tears it down when the task exits, billed by the second. We use ECS on Fargate so that
every scheduled run — data ingest and forecast alike — gets a fresh, ephemeral machine and
nothing sits idle between runs.

**Why create a "cluster" when nothing here auto-scales?** On the EC2 launch type, a cluster
really is a fleet — the pool of instances that tasks get placed onto. On Fargate there are no
instances, so the cluster is purely a **logical namespace for running tasks**, and we need one
only because every task has to launch *into* a cluster: it's the `--cluster` that
[Step 10](#step-10-verify-run-a-forecast-task-manually)'s manual `run-task` call targets and the
`cluster:` the `EcsRunLauncher` config in
[Step 14](#step-14-configure-dagster-on-the-box) names, and it's where the console groups the
running tasks you'll watch. An empty cluster manages no capacity and costs nothing, so a single
`nged-forecast` cluster is all this deployment ever needs.

1. [**ECS**](https://eu-west-2.console.aws.amazon.com/ecs?region=eu-west-2) → **Clusters** →
   **Create cluster** → **Cluster name** `nged-forecast` → under **Infrastructure**, keep
   **Fargate only**, the pre-selected default (no EC2 instances to manage, per the explanation
   above). **Every other section on the create-cluster form can stay at its console default:**
    - **Service Connect defaults** (unset) — Service Connect wires long-running ECS *services*
      to each other; this deployment runs only standalone tasks, so it has nothing to connect.
    - **Monitoring → Container Insights** (turned off) — the default CloudWatch metrics are
      enough at this scale, and both Container Insights options bill for the extra metrics they
      ingest, which is why the console's "Recommended" tag on the enhanced option is ignored
      here.
    - **ECS Exec encryption and logging** (no KMS key; logging **Default**) — nothing in this
      deployment uses ECS Exec (interactive shells into running containers).
    - **Encryption** (both KMS fields empty) — no KMS requirement, for the same reason the S3
      buckets in [Step 1](#step-1-create-the-s3-buckets) stay on SSE-S3.
    - **Tags** (none) — unused, as everywhere else on this page.

    Then click **Create**.

2. **ECS** → [**Task definitions**](https://eu-west-2.console.aws.amazon.com/ecs/v2/task-definitions?region=eu-west-2) → **Create new task definition**.
   Work down the form; anything not named below stays at its console default:
    - **Task definition family**: `nged-forecast` — not a free choice: it's the
      `--task-definition` in [Step 10](#step-10-verify-run-a-forecast-task-manually)'s manual
      run, the `task_definition:` in [Step 14](#step-14-configure-dagster-on-the-box)'s launcher
      config, and the family `scripts/push_and_deploy_image.sh` registers new revisions into.
    - **Launch type**: keep **AWS Fargate**, the pre-ticked default.
    - **Operating system/Architecture**: **Linux/ARM64** — the console defaults to
      `Linux/X86_64`, but [Step 4](#step-4-build-and-verify-the-image)'s script always builds
      the image for ARM, and ARM Fargate is also ~20% cheaper than x86 for the same task size.
      (**Network mode** is greyed out at `awsvpc` — the only mode Fargate supports.)
    - **Task size**: **4 vCPU** / **16 GB** — comfortably above the measured inference peak
      (~9 GB).
    - **Task role**: the task role from [Step 7](#step-7-iam-roles-for-the-fargate-task), e.g.
      `nged-forecast-task-role`.
    - **Task execution role**: the execution role from
      [Step 7](#step-7-iam-roles-for-the-fargate-task), e.g. `nged-forecast-task-execution-role`.
      The field defaults to **Create default role** — don't leave it there: the generic role the
      console would mint has no Parameter Store access, so the secrets injection below would
      fail at container start.
    - **Container – 1 → Name**: `nged-forecast` — [Step 14](#step-14-configure-dagster-on-the-box)'s
      run-launcher config names this container, so keep it predictable. Leave **Essential
      container** at **Yes**.
    - **Image URI**: `<account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<tag>` from
      [Step 6](#step-6-push-the-image-to-ecr) — `<tag>` is the first 12 characters of the
      promoted model's MLflow run id. Step 6's script prints the full URI as it pushes; to
      recover it later, copy it from the pushed image in the ECR console, or run
      `jq -r '.mlflow_run_id[:12]' data/production_model/promotion.json` on the machine that
      built the image.
    - **Port mappings**: **Remove** the pre-filled port-80/HTTP row — every task here is a batch
      run that listens on nothing; port mappings are for services that accept traffic.
    - **Environment variables**: `DATA_PATH_INTERNAL=s3://nged-forecast-internal/data` and
      `DATA_PATH_DELIVERY=s3://nged-forecast-delivery/data` — both are needed, since the delivery
      tables live in a separate bucket from everything else (see the
      [Configuration reference](setup.md#the-configuration-model)). Leave
      `DATA_STORE_*` unset, since the task role supplies credentials. These are plain
      environment values (**Value type** = **Value**), not secrets — bucket URIs are safe in
      clear text.
    - **Secrets — the four Parameter Store entries.** In the same **Environment variables**
      section, add each parameter from [Step 8](#step-8-store-secrets-in-parameter-store) as a
      variable whose **Value type** is **ValueFrom** (the console's secrets mechanism). The
      **Key** is the env-var name exactly as the code expects; the **Value** is the matching
      Parameter Store name:
        - **Key** `NGED_S3_BUCKET_URL` — **Value** `/nged-forecast/nged-s3-bucket-url`
        - **Key** `NGED_S3_BUCKET_ACCESS_KEY` — **Value**
          `/nged-forecast/nged-s3-bucket-access-key`
        - **Key** `NGED_S3_BUCKET_SECRET` — **Value** `/nged-forecast/nged-s3-bucket-secret`
        - **Key** `DAGSTER_PG_PASSWORD` — **Value** `/nged-forecast/dagster-pg-password`
    - **Logging**: keep **Use log collection** ticked with destination **Amazon CloudWatch** and
      the pre-filled options, setting **awslogs-group** to `/ecs/nged-forecast`. (The blue
      "sidecar" notice concerns the *other* destinations, which route logs through an extra
      container; the CloudWatch option is a plain Docker log driver — no sidecar.) One catch:
      the pre-filled `awslogs-create-group: true` asks the *ECS agent* to create the log group
      at first task start, which needs `logs:CreateLogGroup` — a permission neither the
      AWS-managed `AmazonECSTaskExecutionRolePolicy` nor
      [Step 7](#step-7-iam-roles-for-the-fargate-task)'s inline policy grants. Rather than widen
      the role, create the log group yourself now: **CloudWatch** → **Logs** → [**Log
      Management**](https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#logsV2:log-groups) (the console's name for the log-groups list) → **Create log group** → name
      `/ecs/nged-forecast`, region `eu-west-2` — or in one CLI call:
      `aws logs create-log-group --log-group-name /ecs/nged-forecast --region eu-west-2`. With
      the group already in place the agent never attempts creation, so the missing permission is
      never exercised.
    - **Docker configuration** (a collapsed section near the bottom of the container panel) →
      **Entry point**: `/usr/bin/env`; leave **Command** and **Working directory** empty. This
      **overrides the image's own `ENTRYPOINT ["dagster"]`**, and matters more than it looks.
      ECS concatenates entry point + command into one argv, and both the manual verification in
      [Step 10](#step-10-verify-run-a-forecast-task-manually) and the `EcsRunLauncher` in
      [Step 14](#step-14-configure-dagster-on-the-box) supply a *full* command that already
      starts with `dagster` (the launcher generates `dagster api execute_run …` and can't be
      told otherwise). With the image's default entry point, the argv would come out as
      `dagster dagster api execute_run …` and fail; `/usr/bin/env dagster …` simply resolves
      `dagster` from `PATH` and runs it. The image's own entry point remains convenient for
      local `docker run` smoke tests, which is why it isn't changed in the Dockerfile itself.

    Everything else — **Task placement**, **Fault injection**, the container's **Restart
    policy** / **HealthCheck** / timeouts, **Storage** (the 20 GiB ephemeral default is
    plenty), **Monitoring**, **Tags** — stays at its default. Then click **Create**.

Creating the task definition in the console is one-time. Later image changes never repeat it —
`scripts/push_and_deploy_image.sh` registers new revisions of this family automatically (see
[Redeploying a new champion model](#redeploying-a-new-champion-model)).

## Step 10 — Verify: run a forecast task manually

Before building the control plane, trigger one task directly and confirm the compute path
end-to-end — this manual `RunTask` also remains the fallback verification path once the
schedules exist.

**This command is a template — it will not run as pasted.** It contains three placeholders you
must replace first: `<subnet-id>`, `<sg-id>`, and the partition key `<key>`. How to fill in each
one is explained below the command.

```bash
aws ecs run-task \
  --region eu-west-2 \
  --cluster nged-forecast \
  --task-definition nged-forecast \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides": [{"name": "nged-forecast", "command": [
    "dagster", "job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job",
    "--tags", "{\"dagster/partition\": \"<key>\"}"
  ]}]}'
```

Note the command starts with `dagster` — Step 9's `/usr/bin/env` entry point means every command
spells out the full argv.

Fill in the three placeholders:

- **`<subnet-id>`** — a **public** subnet in the VPC the task will run in. Unless you have built
  custom networking in this account, that VPC is the default VPC, and every subnet in a default
  VPC is public (it routes to the internet gateway and auto-assigns public IPs). List them in
  the console under **VPC → Subnets**, or with:

    ```bash
    aws ec2 describe-subnets --region eu-west-2 \
      --filters Name=default-for-az,Values=true \
      --query 'Subnets[].[SubnetId,AvailabilityZone]' --output table
    ```

    Any one of them will do.

- **`<sg-id>`** — the security group the Fargate tasks run with. Create a dedicated one now:
  **EC2 → Security Groups → Create security group** → name `nged-forecast-task-sg`, any
  description, and the same VPC as the subnet above → leave **Inbound rules** empty (the task
  only dials *out*: ECR, S3, CloudWatch, and later Postgres on the control-plane box) → keep
  the default allow-all **Outbound rules** → **Create**. The id (`sg-…`) is shown on the
  group's detail page. Don't reuse the VPC's `default` group:
  [Step 14](#step-14-configure-dagster-on-the-box) opens the control-plane box's Postgres port
  to exactly this group, so a dedicated group keeps that inbound rule tightly scoped.

- **`<key>`** — the `live_forecasts` partition to run, formatted `YYYY-MM-DD-HH:MM` with the
  time at a 6-hourly boundary (e.g. `2026-07-04-00:00`). A key names the *start* of its 6-hour
  window, and the forecast init time is that window's *end* — read the
  [partition-semantics note](operations.md#step-3-let-the-schedule-run-or-materialise-live_forecasts-by-hand)
  before picking one; the most recent *completed* window is the natural choice for this
  verification run.

> **Region matters here.** Unlike [Step 6](#step-6-push-the-image-to-ecr)'s script (which
> passes `--region` explicitly on every call), a hand-typed AWS CLI command falls back to your
> configured default region if `--region` is omitted — check yours with
> `aws configure get region` and don't assume it's `eu-west-2`.

`assignPublicIp=ENABLED` (with a public subnet) is the simplest way to give the task internet
egress for its ECR pull and S3/CloudWatch calls without a NAT gateway. Keep this `<subnet-id>`
and `<sg-id>` to hand — the run launcher in
[Step 14](#step-14-configure-dagster-on-the-box) launches tasks with exactly the same network
configuration.

Follow the run in **ECS** → the cluster → **Tasks**, then **CloudWatch Logs** (log group
`/ecs/nged-forecast`) for the container's output. What counts as a pass depends on whether any
data has been ingested yet:

- **On the first pass through this runbook, the buckets are still empty** — nothing ingests
  data until [Step 16](#step-16-turn-on-the-schedules-and-verify-end-to-end)'s schedules turn
  on — so the run *cannot* produce a forecast. The pass here is the cloud twin of
  [Step 4](#step-4-build-and-verify-the-image)'s offline smoke test: the task starts (proving
  the image pulled and all four secrets resolved), its logs stream to CloudWatch, and the run
  fails *only* at the NWP-availability lookup — a traceback ending in `TableNotFoundError`,
  raised from `_available_nwp_init_times`. Loading the model writes no log line, so that
  specific failure *is* the proof the model loaded: the `live_forecasts` asset loads the model
  first and checks NWP availability second, so dying at the lookup means
  `load_forecaster_from_dir` already succeeded. This exercises every AWS-side link in the
  chain — cluster, task definition, IAM roles, secrets, networking, logging — which is exactly
  what this step exists to verify.

- **Once data exists** (after [Step 16](#step-16-turn-on-the-schedules-and-verify-end-to-end),
  when this command is the fallback verification path), additionally confirm the run succeeds
  and a new forecast lands under `s3://nged-forecast-delivery/data/power_forecasts/…` — the
  delivery bucket, not the internal one, since `power_forecasts` is one of the five NGED-facing
  tables.

## Step 11 — Launch the control-plane box

We will now set up one small always-on EC2 instance to run the Dagster daemon (schedules, sensors,
run dispatch), the Dagster webserver (the UI), the code-location server, and Postgres
(run/event/schedule history) — everything except the actual forecast compute and data fetching jobs,
which stay on ephemeral Fargate. (To read about why we're using an always-on box, and why this
shape, see the [orchestration
decision](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm);
the sizing and cost are the roadmap's [accepted
option](../roadmap/live-service.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month).)

First create its IAM role: **IAM** →
[**Roles**](https://us-east-1.console.aws.amazon.com/iam/home?region=eu-west-2#/roles) → **Create
role** → trusted entity type **AWS service** → under **Use case**, pick **EC2** from the **Service
or use case** dropdown, then select the **EC2** radio button (the plain *EC2* variant, not
*EC2 - Spot Instances* or the other sub-options — it's what puts `ec2.amazonaws.com` in the role's
trust policy, so the instance can assume the role) → **Next** → attach `nged-forecast-read-and-write`
([Step 2](#step-2-grant-data-access-with-iam)) and the AWS-managed
`AmazonEC2ContainerRegistryReadOnly` (pull the image) → **Next**. On the final **Name, review, and
create** page, set **Role name** `nged-forecast-ctrl-role` and a **Description** such as *"Control-plane
box for the NGED forecast service: runs the Dagster daemon/webserver, reads and writes the forecast S3
buckets, pulls the image from ECR, and dispatches forecast runs to Fargate."* Leave the auto-generated
**Trust policy** (the `ec2.amazonaws.com` principal) exactly as shown — it's correct and never
hand-edited here. Then **Create role**.

Now add one inline policy (e.g. `nged-forecast-launch-runs`) so the daemon can dispatch runs to
Fargate — this is a *permissions* policy, separate from the trust policy above, and it's added after
the role exists, exactly as in [Step 7](#step-7-iam-roles-for-the-fargate-task). Open the role you
just created → **Permissions** tab → **Add permissions** → **Create inline policy** → switch to the
**JSON** editor → paste the JSON below, replacing `<account-id>` with your 12-digit AWS account id
(the account menu at the top right of the console, or
`aws sts get-caller-identity --query Account --output text`) → **Next** → name it → **Create policy**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:RunTask",
        "ecs:StopTask",
        "ecs:DescribeTasks",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecs:TagResource",
        "ecs:ListTagsForResource"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::<account-id>:role/nged-forecast-task-role",
        "arn:aws:iam::<account-id>:role/nged-forecast-task-execution-role"
      ],
      "Condition": {
        "StringEquals": { "iam:PassedToService": "ecs-tasks.amazonaws.com" }
      }
    }
  ]
}
```

(`iam:PassRole` is what lets the daemon hand [Step 7](#step-7-iam-roles-for-the-fargate-task)'s
roles to the tasks it launches; the condition stops the box passing them to anything but ECS.
`ecs:RegisterTaskDefinition` is included because `EcsRunLauncher` may register a derived task
definition depending on config — harmless to grant, confusing to debug when missing.)

Leave the role's **Maximum session duration** at its 1-hour default — it's irrelevant to how this
role is used. `MaxSessionDuration` only bounds sessions created by an explicit `sts:AssumeRole` call
that requests a duration (a human assuming a role, or role chaining). An **EC2 instance profile**
takes neither path: the Instance Metadata Service auto-issues and auto-rotates the role's temporary
credentials on the instance's behalf, refreshing them before expiry for as long as the box runs, so
this field is never consulted and the box's credentials never lapse.

Then **EC2** → **Instances** → **Launch instance**. Nearly every default the wizard pre-fills —
Amazon Linux, 64-bit x86, `t3.micro`, an 8 GiB volume, and a security group that allows SSH from
anywhere — is *not* what we want, so work through each section and change it:

- **Name and tags** → **Name**: `nged-forecast-ctrl`.
- **Application and OS Images (Amazon Machine Image)**: click the **Ubuntu** quick-start tile,
  set the **Architecture** dropdown to **64-bit (Arm)** *first* (it defaults to x86, and the AMI
  ID and the entire OS image change when you switch it), then in the **Amazon Machine Image
  (AMI)** dropdown choose **Ubuntu Server 26.04 LTS**. Confirm the panel then shows
  **Username: ubuntu** — that is the login name [Step 12](#step-12-join-the-tailnet)'s SSH uses.
- **Instance type**: `t4g.medium` (2 vCPU / 4 GiB — comfortable for daemon + webserver + code
  server + Postgres; the costed sizing is in the roadmap link above). The `t4g` family is
  Graviton (Arm), which is why the Arm AMI above is required — an x86 AMI won't offer these types.
- **Key pair (login)** → **Create new key pair**: name it `nged-forecast-ctrl`, **Key pair
  type** **ED25519**, **Private key file format** **.pem**, then **Create key pair** and the
  browser downloads `nged-forecast-ctrl.pem`. ED25519 over RSA: shorter keys, faster handshakes,
  and every client in this runbook supports it. Move the file somewhere durable and lock it down
  — `mv ~/Downloads/nged-forecast-ctrl.pem ~/.ssh/ && chmod 400 ~/.ssh/nged-forecast-ctrl.pem`
  (`ssh` refuses a private key with group/other-readable permissions). It's only needed for the
  first login below — Tailscale SSH takes over in [Step 12](#step-12-join-the-tailnet).
- **Network settings** → **Edit** (the collapsed summary can't set the subnet or edit the
  security-group rules, so you must expand it):
    - **VPC**: the same one (the default `vpc-…`) as
      [Step 10](#step-10-verify-run-a-forecast-task-manually)'s task.
    - **Subnet**: change it from **No preference** to a specific **public subnet** in that VPC —
      the same subnet Step 10's task ran in is the natural pick (every default-VPC subnet is
      public, so any of them works).
    - **Auto-assign public IP**: **Enable** (internet egress for apt/ECR/S3/Tailscale without a
      NAT gateway).
    - **Firewall (security groups)**: keep **Create security group** and set its **Security
      group name** to `nged-forecast-ctrl-sg`. The wizard pre-adds one inbound rule — **Allow SSH
      traffic from Anywhere** — so **delete it**, leaving **no inbound rules at all**. Tailscale
      needs none (it dials out), and "no public inbound ports" is a load-bearing security
      decision, since the Dagster UI has no authentication of its own. Leave the default
      allow-all **outbound** rule untouched. One inbound rule is added later, for Postgres
      ([Step 14](#step-14-configure-dagster-on-the-box)).

- **Configure storage**: change the single **Root volume** from the default 8 GiB to **20 GiB**,
  volume type **gp3** (general-purpose SSD). Leave **File systems** at **None**.
- **Advanced details** (expand this section near the bottom of the form):
    - **IAM instance profile**: `nged-forecast-ctrl-role`.
    - **Metadata version**: **V2 only (token required)**, and — easy to miss, breaks everything
      quietly if skipped — **Metadata response hop limit: 2**. The Dagster containers fetch the
      instance role's credentials from the instance metadata service, and Docker's bridge adds a
      network hop; with the default hop limit of 1, boto3 and `object_store` inside the
      containers silently find no credentials.

Check the **Summary** panel on the right (it should read Ubuntu 26.04, `t4g.medium`, your new
`nged-forecast-ctrl-sg`, 20 GiB gp3), then **Launch instance**.

## Step 12 — Join the tailnet

[Step 11](#step-11-launch-the-control-plane-box)'s security group has **no inbound rules**, so
there is no way in yet: a plain `ssh` from your laptop *and* the console's browser **EC2 Instance
Connect** both fail with *"Port 22 (SSH) is not authorized"*. Open **one temporary inbound rule**
for this single bootstrap login, then delete it the moment Tailscale is up (from then on Tailscale
dials out and the box needs no inbound SSH ever again).

Add the rule scoped to your laptop's current public IP — **EC2** → **Security Groups** →
`nged-forecast-ctrl-sg` → **Inbound rules** → **Edit inbound rules** → **Add rule**: **Type**
`SSH`, **Source** **My IP** (the console fills in your laptop's `/32`) → **Save rules**.

Now find the address to SSH *to*: the **control-plane box's own public IPv4 address** — the one
AWS auto-assigned the instance at launch (Step 11's *Auto-assign public IP*), **not** your laptop's
IP from the rule you just added. Read it off **EC2** → **Instances** → select `nged-forecast-ctrl`
→ the **Details** tab → **Public IPv4 address**, or from the CLI:

```bash
aws ec2 describe-instances --region eu-west-2 \
  --filters Name=tag:Name,Values=nged-forecast-ctrl Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text
```

A stop/start reassigns this address, but you only need it for this one login — Tailscale's stable
MagicDNS name takes over afterwards. SSH in with
[Step 11](#step-11-launch-the-control-plane-box)'s key pair, substituting that address for
`<public-ip>`:

```bash
ssh -i ~/.ssh/nged-forecast-ctrl.pem ubuntu@<public-ip>
```

**First, bring the box fully up to date.** Now — before anything runs on it — is the clean moment,
because an `apt upgrade` can pull a new kernel that only takes effect on reboot, and it's far
nicer to bounce an empty box than one running Dagster and Postgres. A fresh cloud image runs
`cloud-init` and `unattended-upgrades` at first boot, which hold the dpkg lock, so wait for those
to finish first:

```bash
sudo cloud-init status --wait          # returns once first-boot automation releases the dpkg lock
sudo apt update && sudo apt upgrade -y
```

If that upgraded the kernel or libc, `sudo reboot` and reconnect (`ssh -i
~/.ssh/nged-forecast-ctrl.pem ubuntu@<public-ip>` again) before continuing.

**Then install Tailscale and join the tailnet.** Use Tailscale's install script rather than
`apt install tailscale`: the script adds Tailscale's *own* APT repository and installs from it, so
the box tracks Tailscale's current stable release and keeps getting it through `apt upgrade`;
Ubuntu's `universe` package is frozen at the release's snapshot and lags. (If you'd rather not pipe
a script to a shell, Tailscale's [manual APT repo
steps](https://tailscale.com/kb/1476/install-ubuntu-2404) do exactly what the script automates.)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh --hostname=nged-forecast-ctrl
```

Open the authentication URL that `tailscale up` prints and sign in **with the OCF Google Workspace
account (`…@openclimatefix.org`)**, so the box joins the shared OCF org tailnet rather than a
personal one. `--ssh` enables Tailscale SSH, so from now on any device on the tailnet can
`ssh ubuntu@nged-forecast-ctrl` with no key management; `--hostname` gives the box a stable MagicDNS
name. Confirm from your laptop:

```bash
tailscale ping nged-forecast-ctrl
ssh ubuntu@nged-forecast-ctrl
```

**Disable key expiry for this node.** A device authenticated as a *user* inherits Tailscale's
default node-key expiry (~180 days); when it lapses, an always-on box silently drops off the
tailnet and you lose SSH and UI access until someone re-authenticates it. In the
[Tailscale admin console](https://login.tailscale.com/admin/machines), open the
`nged-forecast-ctrl` machine → **Disable key expiry**. (The tidier long-term option is to re-auth
the box with a *tagged* auth key, e.g. `tag:nged-forecast`, which makes it org-owned and
non-expiring in one step; disabling expiry on the user-owned node is the quick fix.)

Because the tailnet is the *only* way in and the Dagster UI has no authentication of its own,
anyone who can reach this box over the OCF tailnet gets its UI and — via `--ssh` — a shell as
`ubuntu`, governed by the tailnet's ACLs. That is intended here (OCF-wide access is fine for this
box); tighten it later with Tailscale ACLs or tags if that ever changes.

Now **delete the temporary inbound rule** (**Edit inbound rules** → **Delete** → **Save rules**),
returning `nged-forecast-ctrl-sg` to zero inbound rules — Tailscale establishes its connections
outbound, so nothing else needs the port open. The only inbound rule the box ever keeps is the
Postgres one added in [Step 14](#step-14-configure-dagster-on-the-box).

> **Prefer never opening a public port, even briefly?** Two alternatives keep the group at zero
> inbound rules throughout, at the cost of more setup: create an **EC2 Instance Connect Endpoint**
> in the subnet and connect through it, or attach the AWS-managed `AmazonSSMManagedInstanceCore`
> policy to `nged-forecast-ctrl-role` and use **SSM Session Manager** — a browser shell needing no
> keys and no inbound ports (the Ubuntu AMI ships the SSM agent by default). For a single
> bootstrap login the temporary rule above is the least work.

## Step 13 — Install Docker and pull the image

On the box:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu    # then log out and back in for it to take effect

sudo snap install aws-cli --classic
REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-2.amazonaws.com
aws ecr get-login-password --region eu-west-2 \
  | docker login --username AWS --password-stdin "$REGISTRY"
docker pull "$REGISTRY/nged-forecast:<tag>"
```

`REGISTRY` builds itself from the box's own 12-digit account id, which `aws sts get-caller-identity`
reads from the instance role ([Step 11](#step-11-launch-the-control-plane-box)) — so there is
nothing to paste for the account id. **You do still have to fill in `<tag>`**: it is the first 12
characters of the promoted model's MLflow run id, naming exactly which image to deploy. Step 6's
script prints the full URI as it pushes; to recover it later, copy it from the pushed image in the
ECR console, or run `jq -r '.mlflow_run_id[:12]' data/production_model/promotion.json` on the machine
that built the image.

No `aws configure` is needed — the instance role from
[Step 11](#step-11-launch-the-control-plane-box) supplies credentials. The ECR login token
expires after 12 hours, which is fine: image pulls here are a manual, per-deploy event, so just
re-run the `docker login` line whenever you pull a new tag.

## Step 14 — Configure Dagster on the box

Everything lives in one directory:

```text
~/nged-forecast/
├── docker-compose.yml
├── .env                     # chmod 600
└── dagster_home/
    ├── dagster.yaml
    └── workspace.yaml
```

### One image, four roles

The deployment runs two kinds of job — the always-on Dagster control plane on this box, and the
6-hourly forecast run on ephemeral Fargate — yet the repository builds a **single Dockerfile**.
There is no contradiction: a container image is not a program, it's a packaged **filesystem** —
here, the Python environment (with every dependency installed), the `nged_substation_forecast`
code, and the baked-in champion model. *Which program runs* is chosen per container, by the
command each one is started with. The same image therefore plays four roles:

| Role | Runs where | Started as | What it uses from the image |
|---|---|---|---|
| Code-location server | this box (compose) | `dagster code-server start …` | the Python env + project code, served to the other services over gRPC |
| Webserver (the UI) | this box (compose) | `dagster-webserver …` | just the installed `dagster-webserver` binary — it asks the code server for definitions |
| Daemon | this box (compose) | `dagster-daemon run …` | the `dagster-daemon` binary, plus `dagster-aws` for the run launcher |
| Forecast run | ephemeral Fargate task | `dagster api execute_run …` (generated by the launcher) | everything: env, code, **and the champion model** — the only role that loads it |

One image rather than a slim control-plane image plus a fat run image is deliberate:

- **No version skew.** The code location that evaluates schedules and the Fargate worker that
  executes the run see byte-identical code and dependencies, because they are literally the
  same image tag. With two images, a partial deploy could leave the control plane and the run
  worker disagreeing about what `live_forecasts` *is* — a failure mode this design cannot have.
- **One pipeline.** A single build/verify/push ([Steps 4](#step-4-build-and-verify-the-image)
  [–6](#step-6-push-the-image-to-ecr)) and a single tag to reason about, instead of two builds
  to keep in lockstep.
- The only cost is that the three control-plane containers carry the champion model as dead
  weight — a few megabytes of XGBoost files they never load. Cheap, and harmless.

The one place the single-image design needs care is **entrypoints**, because the image can only
declare one default (`ENTRYPOINT ["dagster"]`) while the four roles start three different
executables: `dagster code-server start` is an ordinary `dagster` subcommand, so the image's
entrypoint stands for it; `dagster-webserver` and `dagster-daemon` are **separate
console-script binaries**, not `dagster` subcommands, so those two compose services carry
`entrypoint:` overrides; and the Fargate run container neutralises the entrypoint with
`/usr/bin/env` because the launcher's generated command already starts with `dagster` (the
[Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition) gotcha).

`~/nged-forecast/.env` (readable by you alone — `chmod 600 .env`):

```dotenv
IMAGE=<account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<tag>
DATA_PATH_INTERNAL=s3://nged-forecast-internal/data
DATA_PATH_DELIVERY=s3://nged-forecast-delivery/data
NGED_S3_BUCKET_URL=<same value as the Parameter Store entry>
NGED_S3_BUCKET_ACCESS_KEY=<same value as the Parameter Store entry>
NGED_S3_BUCKET_SECRET=<same value as the Parameter Store entry>
DAGSTER_PG_PASSWORD=<same value as /nged-forecast/dagster-pg-password from Step 8>
SENTRY_DSN=<OCF Sentry project DSN>
SENTRY_ENVIRONMENT=production
SENTRY_MONITOR_FORECASTS=true
```

The three `SENTRY_*` lines turn on error telemetry and the missed-check-in alarm (see
[Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence)).
`SENTRY_ENVIRONMENT=production` is what the alarm's alert rule is scoped to; a developer testing
from a laptop instead sets `SENTRY_ENVIRONMENT=<name>-laptop` and leaves `SENTRY_MONITOR_FORECASTS`
unset (so a laptop never registers a stale check-in on the production monitor). An empty/absent
`SENTRY_DSN` disables Sentry entirely, so this is safe to omit while bringing the box up.

(The Fargate containers get these injected from Parameter Store; the box's containers read this
file instead — Stage-1 simplicity, one hand-managed box.)

`~/nged-forecast/docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:17
    restart: always
    environment:
      POSTGRES_USER: dagster
      POSTGRES_PASSWORD: ${DAGSTER_PG_PASSWORD}
      POSTGRES_DB: dagster
    ports:
      - "5432:5432"   # reachable by the Fargate run workers; see the security-group rule below
    volumes:
      - pgdata:/var/lib/postgresql/data

  code-server:
    image: ${IMAGE}
    restart: always
    # `code-server start` is a `dagster` subcommand, so the image's ENTRYPOINT ["dagster"] stands
    command:
      ["code-server", "start", "--host", "0.0.0.0", "--port", "4266",
       "-m", "nged_substation_forecast.definitions"]
    env_file: .env
    environment:
      DAGSTER_HOME: /opt/dagster/dagster_home
      AWS_DEFAULT_REGION: eu-west-2   # boto3 has no other region source on the box; see note below
    volumes:
      - ./dagster_home:/opt/dagster/dagster_home
    depends_on: [postgres]

  webserver:
    image: ${IMAGE}
    restart: always
    entrypoint: ["dagster-webserver"]   # separate binary, NOT a `dagster` subcommand
    # --db-pool-max-overflow lifts the Postgres pool ceiling above its default of 20; see note below
    command:
      ["-h", "0.0.0.0", "-p", "3000", "-w", "workspace.yaml", "--db-pool-max-overflow", "40"]
    working_dir: /opt/dagster/dagster_home
    ports:
      - "3000:3000"
    env_file: .env
    environment:
      DAGSTER_HOME: /opt/dagster/dagster_home
      AWS_DEFAULT_REGION: eu-west-2   # boto3 has no other region source on the box; see note below
    volumes:
      - ./dagster_home:/opt/dagster/dagster_home
    depends_on: [postgres, code-server]

  daemon:
    image: ${IMAGE}
    restart: always
    entrypoint: ["dagster-daemon"]      # separate binary, NOT a `dagster` subcommand
    command: ["run", "-w", "workspace.yaml"]
    working_dir: /opt/dagster/dagster_home
    env_file: .env
    environment:
      DAGSTER_HOME: /opt/dagster/dagster_home
      AWS_DEFAULT_REGION: eu-west-2   # boto3 has no other region source on the box; see note below
    volumes:
      - ./dagster_home:/opt/dagster/dagster_home
    depends_on: [postgres, code-server]

volumes:
  pgdata:
```

**`AWS_DEFAULT_REGION` is set on every box container** because boto3 does not fall back to
instance metadata for its *region* the way it does for credentials. The instance role
([Step 11](#step-11-launch-the-control-plane-box)) supplies credentials over IMDS, but region has
no such fallback — so without this the daemon's `EcsRunLauncher` fails the moment it tries to launch
a run, with `botocore.exceptions.NoRegionError: You must specify a region`, and the box's other
boto3 calls (S3) would fail the same way. The Fargate run workers don't need it: a task reads its
region from the ECS task metadata automatically, so this gap is box-only.

**`--db-pool-max-overflow 40` on the webserver** raises its Postgres connection-pool ceiling. Every
`dagster-postgres` storage hard-codes `pool_size: 1`, and the webserver's overflow defaults to 20 —
a 21-connection ceiling. The UI's live asset-status poll (`AssetGraphLiveQuery`) fans out one
connection per asset node across a threadpool, and on this small box those queries hold their
connections long enough (Postgres shares 2 vCPU / 4 GiB with the daemon, webserver, and code-server)
that concurrent polling drains the pool. When it does, connection checkouts wait 30 s, time out, and
the GraphQL query dies with `too many retries for DB connection` /
`QueuePool limit of size 1 overflow 20 reached`. Lifting the overflow to 40 (41 connections) clears
it with headroom, still well under `postgres:17`'s default `max_connections` of 100 — leaving room
for the daemon, code-server, and the Fargate run workers that also connect back. Only the webserver
takes this flag; the daemon and code-server don't run the live query, so they don't need it.

`~/nged-forecast/dagster_home/workspace.yaml` — tells the webserver and daemon where user code
lives (only ever read on the box, so the compose-network hostname is fine here):

```yaml
load_from:
  - grpc_server:
      host: code-server
      port: 4266
      location_name: nged_substation_forecast
```

`~/nged-forecast/dagster_home/dagster.yaml` — the instance config. The `concurrency`,
`run_monitoring`, and `python_logs` blocks are the same ones the local `dagster.yaml` in the
[repository README](https://github.com/openclimatefix/nged-substation-forecast#setup) uses, and
for the same reasons; what's new is Postgres storage and the run launcher:

```yaml
storage:
  postgres:
    postgres_db:
      hostname: <box-private-ip>   # NOT "postgres" — see note below
      port: 5432
      username: dagster
      password:
        env: DAGSTER_PG_PASSWORD
      db_name: dagster

run_launcher:
  module: dagster_aws.ecs
  class: EcsRunLauncher
  config:
    task_definition: nged-forecast    # the family from Step 9; latest revision is used
    container_name: nged-forecast     # must match Step 9's container name
    use_current_ecs_task_config: false   # the daemon runs on EC2, not inside an ECS task
    secrets_tag: null                 # we use Parameter Store, not Secrets Manager; see note below
    run_task_kwargs:                  # boto3 RunTask kwargs, hence the camelCase
      cluster: nged-forecast
      launchType: FARGATE
      networkConfiguration:
        awsvpcConfiguration:
          subnets: ["<subnet-id>"]          # same as Step 10's manual run-task
          securityGroups: ["<sg-id>"]       # same as Step 10's manual run-task
          assignPublicIp: ENABLED

concurrency:
  pools:
    default_limit: 1  # Used to limit concurrency of the ecmwf_ens asset.

run_monitoring:
  # Without this, a crashed/killed run can leak its concurrency-pool slot (e.g. the pool
  # above) forever, since nothing else frees a slot held by a run that never reached a
  # normal finally-block exit. This lets the daemon self-heal: any run finished (in any
  # terminal status) for longer than the threshold has its slots freed automatically.
  enabled: true
  free_slots_after_run_end_seconds: 300

python_logs:
  managed_python_loggers:
    - nged_data
  python_log_level: DEBUG
```

Four things in `dagster.yaml` deserve explanation:

- **`secrets_tag: null` disables Secrets Manager enumeration.** `EcsRunLauncher.secrets_tag`
  defaults to `"dagster"`, which makes the launcher call `secretsmanager:ListSecrets` (filtered by
  that tag) on every run launch to inject matching secrets as env vars. This stack uses Parameter
  Store, not Secrets Manager ([Step 8](#step-8-store-secrets-in-parameter-store)), so that
  call finds nothing useful and the instance role deliberately doesn't grant the permission —
  leaving the default in place fails run launch with `AccessDeniedException ... secretsmanager:ListSecrets`.
  Setting it to `null` skips the lookup entirely.

- **The Postgres hostname is the box's VPC private IP** — the `172.31.x.x` address on its primary
  network interface (it matches the box's `ip-172-31-…` hostname), *not* the compose service name
  `postgres`, and *not* the Tailscale (`100.x` / `fd7a:…`) or Docker-bridge (`172.17.x`) addresses
  that `hostname -I` also lists now that Tailscale and Docker are installed. To read just that one
  address, ask the instance metadata service (an IMDSv2 token is required, because
  [Step 11](#step-11-launch-the-control-plane-box) set token-required):

    ```bash
    TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/local-ipv4
    ```

    This instance config is not only read on the box: the daemon serializes it into every run it
    launches, and the **Fargate run worker connects back to this same Postgres** to record its
    events and heartbeats. The compose-network alias only resolves on the box, whereas the private
    IP is reachable both from Fargate (same VPC) and from the box's own containers (via the
    published port). A private IPv4 persists across stop/start — it only changes if the instance is
    terminated and replaced, at which point this file needs the new IP.

- **`<subnet-id>` and `<sg-id>` in the run launcher** are the public subnet and security group the
  Fargate run workers launch into — the same pair
  [Step 10](#step-10-verify-run-a-forecast-task-manually)'s manual `run-task` used: the
  `nged-forecast-task-sg` group, and any public subnet in the default VPC. Read them off the
  console (**EC2 → Security Groups →** `nged-forecast-task-sg` for the group id; **VPC → Subnets**
  for a subnet id), or from the CLI:

    ```bash
    # <sg-id> — the Fargate task security group
    aws ec2 describe-security-groups --region eu-west-2 \
      --filters Name=group-name,Values=nged-forecast-task-sg \
      --query 'SecurityGroups[0].GroupId' --output text

    # <subnet-id> — any public subnet in the default VPC (all default-VPC subnets are public)
    aws ec2 describe-subnets --region eu-west-2 \
      --filters Name=default-for-az,Values=true \
      --query 'Subnets[].[SubnetId,AvailabilityZone]' --output table
    ```

    The same `<sg-id>` is used twice: here, and as the source of the Postgres inbound rule below.

- Because the run workers connect in, **add the one inbound security-group rule now**. It lives on
  the *security group's* own page, not the instance's **Networking** tab: **EC2 → Security Groups**
  (left sidebar, under *Network & Security*) → open `nged-forecast-ctrl-sg` → the **Inbound rules**
  tab → **Edit inbound rules** → **Add rule**: **Type** `PostgreSQL` (fills in TCP 5432), **Source**
  `Custom` → the Fargate task security group `nged-forecast-task-sg` (`<sg-id>` from
  [Step 10](#step-10-verify-run-a-forecast-task-manually)) → **Save rules**. (From the instance's
  **Security** tab you can click the `nged-forecast-ctrl-sg` link to jump straight to that page.)
  Scoped to that security group — the box still has no publicly-reachable ports. This is also why `DAGSTER_PG_PASSWORD` was
  added to the task definition's secrets in
  [Step 9](#step-9-create-the-ecs-cluster-and-fargate-task-definition): `dagster.yaml`
  references it as `env:`, so the run worker resolves it at startup from its own environment.

The authoritative schema for the `EcsRunLauncher` block is
[Dagster's ECS deployment docs](https://docs.dagster.io/deployment/oss/deployment-options/aws) —
if the daemon fails to start with a config-validation error after a Dagster upgrade, check there
first.

## Step 15 — Start the stack and connect over Tailscale

```bash
cd ~/nged-forecast
docker compose up -d
docker compose ps            # all four services should be Up
docker compose logs -f daemon    # watch it start its schedule/sensor loops
```

> **Expected on a cold start: one `Could not reach user code server` warning.** The first daemon
> log lines often include a `UserWarning: Error loading repository location
> nged_substation_forecast` with a gRPC `UNAVAILABLE` / `Connection refused` traceback pointing at
> the code-server's port 4266. This is a harmless startup race, not a failure. The daemon and
> webserver `depends_on` the code-server, but plain Compose `depends_on` waits only for the
> code-server *container to start*, not for its gRPC server to be *ready* — and the code-server
> has to import the whole project (all code plus the baked-in champion model) before it binds
> 4266, which takes several seconds. The daemon starts sooner, polls once too early, logs the
> warning, and retries. You can confirm it self-healed a few seconds later: a
> `Received LocationStateChangeEventType.LOCATION_UPDATED event for location
> nged_substation_forecast, refreshing` line means the location loaded successfully. Only treat it
> as a real problem if the UI (checked below) shows a *persistent* load error on the code
> location, or the daemon keeps logging the failure with no `LOCATION_UPDATED` recovery.

From your laptop, open **`http://nged-forecast-ctrl:3000`** (Tailscale MagicDNS; the raw
Tailscale IP works too). This is safe without any login because of the network design: the
webserver's port 3000 is published on the box's interfaces, but the security group allows no
inbound traffic — the only way in is through the Tailscale tunnel, and tailnet membership *is*
the authentication (the open-source Dagster webserver has none of its own — see
[the access-phasing plan](../roadmap/live-service.md#access-phasing)).

In the UI, confirm the box is healthy: **Deployment** should show the
`nged_substation_forecast` code location loaded, and **Deployment → Daemons** should show the
scheduler and run-monitoring daemons with green heartbeats.

## Step 16 — Turn on the schedules and verify end-to-end

1. **UI → Automation**: switch on `power_time_series_and_metadata_schedule`,
   `ecmwf_ens_schedule`, and `live_forecasts_schedule`. Schedule state lives in Postgres, so
   this is a one-time action — it survives restarts and reboots.
2. **First time only — materialise the upstream assets once so `live_forecasts` has something
   to read.** A Dagster `deps=[...]` declaration records lineage; it does *not* make
   materialising `live_forecasts` reach back and build its parents first. On a brand-new box the
   Delta tables are empty, so the very first `live_forecasts` run has no NWP, no telemetry, and no
   grid weights to consume. Materialise these once, from the UI, in order (each is its own asset
   because they run on different cadences and, for `ecmwf_ens` vs `live_forecasts`, different
   partition definitions — so there is no single run that can build the whole chain):

    1. `h3_grid_weights` (unpartitioned) — `ecmwf_ens` depends on it, so do this first.
    2. the latest `ecmwf_ens` partition — gives `live_forecasts` an NWP run to forecast from.
    3. `power_time_series_and_metadata` (unpartitioned) — the power spine and substation metadata.

    (The production model is not in this list. It is deliberately *not* a Dagster dependency of
    `live_forecasts`: its artifacts were promoted back in
    [Step 3](#step-3-pick-and-promote-a-champion-model) and baked into the image, so there is
    nothing to materialise for it here — do not materialise `promoted_model` on the box, which has
    no MLflow.) Once
    the schedules from step 1 are on, each of these upstream assets is kept fresh by its own
    schedule; this manual pass is only to seed the empty tables for the first tick.

3. **Kick a run now rather than waiting for a tick**: materialise the latest `live_forecasts`
   partition from the UI (see
   [Operating the live service: Step 3](operations.md#step-3-let-the-schedule-run-or-materialise-live_forecasts-by-hand)).
   Watch the run get dispatched by the launcher: it appears in the Dagster UI, a Fargate task
   spins up in the ECS console, its logs stream to CloudWatch, and forecast rows land under
   `s3://nged-forecast-delivery/data/power_forecasts/…`.
4. **Reboot test**: `sudo reboot` on the box. Docker's systemd unit plus `restart: always`
   must bring all four services back unattended; the UI comes back over Tailscale with run
   history intact.
5. **Leave it running for several days**: a forecast appears after every 6-hourly slot and a
   fresh NWP ingest after each daily 00Z publication; check Cost Explorer against the
   [roadmap's cost model](../roadmap/live-service.md#cost-summary).

If a slot gets missed (box down, failed run), backfill it from the same UI —
[Operating the live service: Backfilling a missed slot](operations.md#backfilling-a-missed-slot).

### Optional hardening

- **Nightly Postgres dump to S3** — Dagster's run history is rebuildable in principle but nice
  to keep. On the box, `crontab -e`:

    ```cron
    15 2 * * * docker compose -f /home/ubuntu/nged-forecast/docker-compose.yml exec -T postgres pg_dump -U dagster dagster | gzip | aws s3 cp - s3://nged-forecast-internal/backups/dagster-$(date +\%F).sql.gz
    ```

- **Security patches**: Ubuntu Server installs security updates automatically via
  `unattended-upgrades`; confirm it's active with `systemctl status unattended-upgrades`.

The missed-check-in alarm that catches a silently-dead daemon is wired via Sentry cron monitoring
(the `SENTRY_*` vars in [Step 14](#step-14-configure-dagster-on-the-box)); the one console step it
needs is to **scope its alert rule to `environment:production`** in Sentry, so a developer testing
from a laptop never trips it. The one remaining operational safety net is per-task failure alerts
(email via SNS, the Simple Notification Service) — still to come; see
[the roadmap](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm) and
[Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence).

## Redeploying a new champion model

Once the service is live, shipping a better model is a repeat of a slice of this page:

1. Promote the new champion locally —
   [Operating the live service: Steps 1–2](operations.md#step-1-pick-a-champion-model).
2. Build and verify the new image ([Step 4](#step-4-build-and-verify-the-image)) — it gets a
   new tag (the new model's run id).
3. `scripts/push_and_deploy_image.sh` ([Step 6](#step-6-push-the-image-to-ecr)) — one command
   that pushes the new tag and registers a new task-definition revision pointing at it. The
   run launcher resolves the family's latest revision at launch time, so the next scheduled
   run picks it up with no restart on the box.
4. The control-plane containers can keep running the old image — the baked-in model is dead
   weight to them, so a *model* change doesn't affect them. When the *code* changes (assets,
   schedules, a Dagster upgrade), also update the box: `docker login` + `docker pull` the new
   tag ([Step 13](#step-13-install-docker-and-pull-the-image)), update `IMAGE` in
   `~/nged-forecast/.env`, and `docker compose up -d`.

## Granting NGED read access (recommended; confirm before doing this)

> This step hasn't been agreed as final yet — recorded here as the current recommendation so it
> isn't lost, but check before actually creating anything.

[Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it) already
assumes **a single authenticated AWS user** for NGED, with no per-user entitlement matrix needed
— NGED is the only consumer. That points at a dedicated **IAM user** (not a cross-account role):
Excel and Power BI are named as expected client tools in that same page, and neither supports
AWS role-assumption — they need a plain access key and secret, the same shape of credential an
IAM user provides.

Recommended shape: **one** IAM user (not one per bucket — the stability signal comes from the
bucket split itself, not from access segmentation), with a read-only policy across both bucket
ARNs (`s3:GetObject`, `s3:ListBucket` — no `PutObject`/`DeleteObject`), and an access key handed
to NGED the same way [Step 2](#step-2-grant-data-access-with-iam)'s dashboard credentials are
configured. Rotate the key periodically once this is live.

## See also

- [Operating the live service](operations.md) — driving the deployed service day to day:
  promotion, the 6-hourly schedule, inspecting forecasts, backfilling missed slots.
- [Running the whole stack locally](local.md) — the same service end-to-end on a laptop, no
  AWS involved.
- [Configuration reference](setup.md) — what the storage roots and `DATA_STORE_*`/credential
  settings mean, and which combination each environment uses.
- [Production Deployment — Design](../architecture/production-deployment.md) — why the image is
  built this way, and why the control plane is an always-on box rather than EventBridge.
- [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) — the costed
  options behind this design, access phasing (team read-only access, the public dashboard), and
  the still-to-come alerting and infra-as-code work.
