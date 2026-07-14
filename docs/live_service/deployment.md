# Deploying a new production image

How to get a promoted champion model into a runnable production image, and run it on AWS. Design
rationale for why the container is built this way — bake the model in at build time, no MLflow
at runtime — lives in
[Production Deployment — Design](../architecture/production-deployment.md); this page is the
mechanical recipe.

> **Scope: manual, point-and-click AWS compute — not yet the unattended deployment.** Steps 3–8
> below get a verified image onto ECR and running as a one-off Fargate task, which is enough to
> confirm the whole path end-to-end. The always-on control-plane box (Dagster daemon, Postgres,
> `EcsRunLauncher`, the 6-hourly schedule) and infra-as-code are still not built — see
> [AWS architecture](../roadmap/live-service.md#aws-architecture) for that plan. Everything here
> is done by hand in the AWS console; no Terraform/CDK yet.

## Step 1 — Pick and promote a champion model

Follow [Running live forecasts end-to-end: Step 1](dagster-workflow.md#step-1-pick-a-champion-model)
and [Step 2](dagster-workflow.md#step-2-materialise-promoted_model) to materialise
`promoted_model`. This populates `data/production_model/` on disk — the same directory the
image build below `COPY`s from.

## Step 2 — Build and verify the image

With a champion model on disk (Step 1), one script builds the image and smoke-tests it with
**zero network access** — the test that matters, since the entire point of baking the model in is
that production inference has no MLflow dependency at runtime:

```bash
scripts/build_and_verify_image.sh <partition-key>   # e.g. 2026-07-04-00:00
```

The `<partition-key>` only has to be **well-formed** (`YYYY-MM-DD-HH:MM`, e.g. `2026-07-04-00:00`);
it need not name a partition that already exists. The smoke test fails at the NWP lookup long
before the slot's validity could matter, so any correctly-formatted key works — off-cadence or
out-of-range is fine, and only a malformed key fails (fast, with a clear `time data … does not
match format` error). If you would rather pass a genuine slot: real slots are the 6-hourly
boundaries at 00:00/06:00/12:00/18:00 UTC from the partition's start date onward; browse the exact
list under the `live_forecasts` asset in the Dagster UI (see also
[the partition-semantics note](dagster-workflow.md#step-3-let-the-schedule-run-or-materialise-live_forecasts-by-hand)).
The script builds the image tagged with the promoted model's run id, runs it offline, and prints
the container log with a pass/fail summary.
It hard-fails only if the runtime touches MLflow — the hermeticity guarantee worth automating —
and otherwise asks you to confirm by eye that the run loaded the model and failed *only* on
missing NWP data (expected: no data tables are mounted for this isolated test). The script header
documents every choice it makes and is the source of truth for the mechanics; a full end-to-end
run instead needs real data-table roots per [Environment & storage setup](setup.md).

## Step 3 — Create the ECR repository

In the AWS console → **ECR** → **Create repository** (same `eu-west-2` region as the S3 bucket
in [Environment & storage setup: Step 1](setup.md#step-1-create-the-s3-buckets)):

1. **Visibility** Private.
2. **Name** `nged-forecast` (matches the local image tag `nged-forecast:<tag>` from
   [Step 2](#step-2-build-and-verify-the-image), keeping it and the ECR URI consistent).
3. **Scan on push** on — free vulnerability scanning, no reason not to.
4. Leave tag immutability off; image tags here are already unique per promoted model (the
   run id's short prefix from [Step 2](#step-2-build-and-verify-the-image)), so nothing relies
   on retagging.

## Step 4 — Push the image to ECR

Authenticate Docker against the new repository, then tag and push the image built in
[Step 2](#step-2-build-and-verify-the-image):

```bash
aws ecr get-login-password --region eu-west-2 \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-2.amazonaws.com

docker tag nged-forecast:<run-id-short> \
  <account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<run-id-short>

docker push <account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<run-id-short>
```

`<account-id>` is the 12-digit AWS account ID (visible in the console's top-right account menu,
or `aws sts get-caller-identity --query Account --output text`).

## Step 5 — IAM roles for the task

Fargate tasks need **two** separate roles, not one — they serve different principals:

- **Task execution role**, `nged-forecast-task-execution-role` — used by the *ECS agent* itself,
  before your code ever runs: pulling the image from ECR, shipping container output to
  CloudWatch Logs, and injecting the NGED credential secrets from
  [Step 6](#step-6-store-the-nged-source-credentials-in-parameter-store). Create a role for the
  `ecs-tasks.amazonaws.com` service and attach the AWS-managed
  `AmazonECSTaskExecutionRolePolicy` (covers ECR + CloudWatch), plus one small inline policy for
  the secrets — it's the *execution* role that reads them, not the task role, because the ECS
  agent resolves secrets before the container starts:

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
  lets the container read and write **both** data buckets (delivery and internal). Reuse the same
  S3 policy from
  [Environment & storage setup: Step 2](setup.md#step-2-grant-access-with-iam), attached to a
  second role (also trusted by `ecs-tasks.amazonaws.com`). Nothing else is needed — with this
  role attached, delta-rs' `object_store` auto-discovers temporary credentials at runtime, so
  `DATA_STORE_*` stays unset (see [setup.md's IAM-role row](setup.md#step-3-point-settings-at-the-buckets)).

No static AWS keys anywhere in either role — this is the same IAM-role auto-discovery setup.md
already relies on for compute running on AWS.

## Step 6 — Store the NGED source credentials in Parameter Store

The container can't start without `NGED_S3_BUCKET_URL`, `NGED_S3_BUCKET_ACCESS_KEY`, and
`NGED_S3_BUCKET_SECRET` (`Settings` requires them at import — see the smoke test in
[Step 2](#step-2-build-and-verify-the-image)), and the deployed service genuinely uses them: the
hourly `power_time_series_and_metadata` schedule pulls fresh telemetry from NGED's bucket.

They are also the one credential in this deployment that can't come from an IAM role: NGED's
bucket lives in NGED's AWS account, so these are unavoidably static third-party keys. Don't
paste them into the task definition as plain-text environment values — anyone with ECS
describe access could read them there. Store them in **SSM Parameter Store** as SecureStrings
and let ECS inject them at container start:

In the AWS console → **Systems Manager** → **Parameter Store** → **Create parameter**, three
times, in `eu-west-2` (same region as everything else):

- **Name**: `/nged-forecast/nged-s3-bucket-url`, `/nged-forecast/nged-s3-bucket-access-key`,
  and `/nged-forecast/nged-s3-bucket-secret` respectively. The shared `/nged-forecast/` prefix
  is exactly what [Step 5](#step-5-iam-roles-for-the-task)'s execution-role policy grants
  access to, so a new secret added under the same prefix later needs no IAM change.
- **Tier**: Standard (free; these values are tiny, nowhere near the 4 KB limit).
- **Type**: **SecureString**, with the default `aws/ssm` KMS key — using the default key is
  what lets Step 5's inline policy skip a `kms:Decrypt` statement.
- **Value**: copied from the matching line of your local `.env`.

There is no wiring to do in this step — that happens in the task definition
([Step 7](#step-7-create-the-ecs-cluster-and-fargate-task-definition)), where each parameter is
referenced as a **ValueFrom** environment variable. The ECS agent, authenticated as the
*execution* role, fetches and decrypts the values just before the container starts; they never
appear in the task definition, the ECS console, or CloudWatch.

Parameter Store is picked over Secrets Manager deliberately: Standard parameters are free
(Secrets Manager is $0.40/secret/month), and these credentials don't need Secrets Manager's
flagship feature, automatic rotation — NGED's keys rotate on NGED's schedule, not ours. When
NGED does rotate them, just update the parameter values (and your local `.env`); running tasks
keep the old values until they next start, since injection happens once per container launch.

## Step 7 — Create the ECS cluster and Fargate task definition

1. **ECS** → **Clusters** → **Create cluster** → *Networking only* (Fargate; no EC2 instances to
   manage) — a bare cluster is just a namespace, so a single `nged-forecast` cluster is enough
   for now.
2. **ECS** → **Task definitions** → **Create new task definition** → *Fargate*:
    - **Task role**: the task role from [Step 5](#step-5-iam-roles-for-the-task).
    - **Task execution role**: the execution role from [Step 5](#step-5-iam-roles-for-the-task).
    - **Task size**: 4 vCPU / 16 GB, **ARM64** (`linux/arm64`) — matches the measured inference
      peak (~9 GB) and the image's own build target (see the Dockerfile's ARM build note); ARM
      Fargate is also ~20% cheaper than x86 for the same size.
    - **Container**: image URI `<account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<tag>`
      from [Step 4](#step-4-push-the-image-to-ecr); log configuration → **awslogs**, a new
      CloudWatch log group (e.g. `/ecs/nged-forecast`), region `eu-west-2`.
    - **Environment variables**: `DATA_PATH_INTERNAL=s3://nged-forecast-internal/data` and
      `DATA_PATH_DELIVERY=s3://nged-forecast-delivery/data` — both are needed, since the delivery
      tables live in a separate bucket from everything else (see
      [setup.md's on-AWS settings](setup.md#step-3-point-settings-at-the-buckets)). Leave
      `DATA_STORE_*` unset, since the task role supplies credentials.
    - **Secrets — the three NGED source-bucket credentials.** Add each Parameter Store entry
      from [Step 6](#step-6-store-the-nged-source-credentials-in-parameter-store) as an
      environment variable of type **ValueFrom** (the console's "Secrets" mechanism): the
      env-var name exactly as `Settings` expects (`NGED_S3_BUCKET_URL`,
      `NGED_S3_BUCKET_ACCESS_KEY`, `NGED_S3_BUCKET_SECRET`), the value the matching parameter
      name (e.g. `/nged-forecast/nged-s3-bucket-url`).

## Step 8 — Verify: run the task manually

There's no schedule yet, so trigger one task directly and confirm the whole path end-to-end —
this mirrors the "manual `RunTask`" verification the roadmap's
[AWS architecture](../roadmap/live-service.md#aws-architecture) plan already anticipates as the
fallback verification path regardless of the eventual always-on architecture:

```bash
aws ecs run-task \
  --region eu-west-2 \
  --cluster nged-forecast \
  --task-definition nged-forecast \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides": [{"name": "<container-name>", "command": [
    "job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job",
    "--tags", "{\"dagster/partition\": \"<key>\"}"
  ]}]}'
```

> **Region matters here.** Unlike the `docker` commands above (region is baked into the ECR
> URI), the AWS CLI falls back to your configured default region if `--region` is omitted —
> check yours with `aws configure get region` and don't assume it's `eu-west-2`.

`assignPublicIp=ENABLED` (with a public subnet) is the simplest way to give the task internet
egress for its ECR pull and S3/CloudWatch calls without a NAT gateway; revisit once the
always-on control-plane box (and its VPC design) exists. Follow the run in **ECS** → the
cluster → **Tasks**, then **CloudWatch Logs** for the container's output — confirm it reaches
`load_forecaster_from_dir` and a new forecast lands under
`s3://nged-forecast-delivery/data/power_forecasts/…` — the delivery bucket, not the internal one,
since `power_forecasts` is one of the five NGED-facing tables (see
[setup.md: Step 3](setup.md#step-3-point-settings-at-the-buckets)).

## See also

- [Production Deployment — Design](../architecture/production-deployment.md) — why the image is
  built this way, and why the control plane that will schedule it is an always-on box rather
  than EventBridge.
- [Running live forecasts end-to-end](dagster-workflow.md) — driving the promoted model day to
  day once it's live, and backfilling missed slots.
- [Environment & storage setup](setup.md) — where data tables and local artifacts live, and the
  S3 bucket + IAM policy Steps 3–8 above build on.
- [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) — the always-on
  control-plane box, scheduling, and infra-as-code, still to come.
