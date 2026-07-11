# Deploying a new production image

How to get a promoted champion model into a runnable production image, and run it on AWS. Design
rationale for why the container is built this way — bake the model in at build time, no MLflow
at runtime — lives in
[Production Deployment — Design](../architecture/production-deployment.md); this page is the
mechanical recipe.

> **Scope: manual, point-and-click AWS compute — not yet the unattended deployment.** Steps 4–8
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

## Step 2 — Build the image

The build never contacts MLflow — it only `COPY`s the directory Step 1 just populated, so it
stays hermetic:

```bash
docker build \
  --build-arg MODEL_RUN_ID=<run-id> \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t nged-forecast:<run-id-short> .
```

`MODEL_RUN_ID` and `GIT_SHA` are stamped as OCI labels (and `GIT_SHA` also as a runtime env
var) purely for traceability — confirm with `docker inspect nged-forecast:<tag>`.

## Step 3 — Verify the build locally

Before trusting a new image, confirm it actually runs with **zero network access** — this is
the test that matters, since the entire point of baking the model in is that production
inference has no MLflow dependency at runtime:

```bash
docker run --network=none nged-forecast:<tag> \
  job execute -m nged_substation_forecast.definitions -j live_forecasts_job \
  --tags '{"dagster/partition": "<key>"}'
```

Partition selection uses `--tags`, not `--select`/`--partition`: `dagster job execute` has no
`--partition` flag at all, and `dagster asset materialize --select <asset>` (which does) hits a
pre-existing, unrelated `antlr4-python3-runtime`/Python 3.14 incompatibility in Dagster's own
asset-selection-string parser — reproduces identically outside Docker, on a plain `dg dev`
checkout, so it's an upstream/environment issue, not something introduced by this image.
Selecting by job name (`-j`) skips that parser entirely.

**Verified 2026-07-10** against a real promoted model: the run reaches
`load_forecaster_from_dir` and loads the model successfully (confirmed via the step ordering in
`production_assets.py` — model load happens before the NWP-availability lookup) with zero
`mlflow` mentions anywhere in the log, then fails only on missing NWP data access (expected —
no `DATA_PATH` was mounted for this isolated test). A full end-to-end run needs a real
`DATA_PATH` (local mount or S3 credentials) supplied via environment variables, per
[Environment & storage setup](setup.md).

## Step 4 — Create the ECR repository

In the AWS console → **ECR** → **Create repository** (same `eu-west-2` region as the S3 bucket
in [Environment & storage setup: Step 1](setup.md#step-1-create-the-s3-bucket)):

1. **Visibility** Private.
2. **Name** `nged-forecast` (matches the local image tag used in [Step 2](#step-2-build-the-image)
   above — keeps `docker build -t nged-forecast:<tag>` and the ECR URI consistent).
3. **Scan on push** on — free vulnerability scanning, no reason not to.
4. Leave tag immutability off; image tags here are already unique per `git rev-parse HEAD`
   short-SHA, so nothing relies on retagging.

## Step 5 — Push the image to ECR

Authenticate Docker against the new repository, then tag and push the image built in
[Step 2](#step-2-build-the-image):

```bash
aws ecr get-login-password --region eu-west-2 \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-2.amazonaws.com

docker tag nged-forecast:<run-id-short> \
  <account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<run-id-short>

docker push <account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<run-id-short>
```

`<account-id>` is the 12-digit AWS account ID (visible in the console's top-right account menu,
or `aws sts get-caller-identity --query Account --output text`).

## Step 6 — IAM roles for the task

Fargate tasks need **two** separate roles, not one — they serve different principals:

- **Task execution role** — used by the *ECS agent* itself, before your code ever runs: pulling
  the image from ECR and shipping container output to CloudWatch Logs. Create a role for the
  `ecs-tasks.amazonaws.com` service and attach the AWS-managed
  `AmazonECSTaskExecutionRolePolicy` — it already covers exactly these two things, so no custom
  policy is needed.
- **Task role** — used by *your code* once it's running: this is what lets the container read
  and write the data bucket. Reuse the same S3 policy from
  [Environment & storage setup: Step 2](setup.md#step-2-grant-access-with-iam), attached to a
  second role (also trusted by `ecs-tasks.amazonaws.com`). Nothing else is needed — with this
  role attached, delta-rs' `object_store` auto-discovers temporary credentials at runtime, so
  `DATA_STORE_*` stays unset (see [setup.md's IAM-role row](setup.md#step-3-point-settings-at-the-bucket)).

No static AWS keys anywhere in either role — this is the same IAM-role auto-discovery setup.md
already relies on for compute running on AWS.

## Step 7 — Create the ECS cluster and Fargate task definition

1. **ECS** → **Clusters** → **Create cluster** → *Networking only* (Fargate; no EC2 instances to
   manage) — a bare cluster is just a namespace, so a single `nged-forecast` cluster is enough
   for now.
2. **ECS** → **Task definitions** → **Create new task definition** → *Fargate*:
   - **Task role**: the task role from [Step 6](#step-6-iam-roles-for-the-task).
   - **Task execution role**: the execution role from [Step 6](#step-6-iam-roles-for-the-task).
   - **Task size**: 4 vCPU / 16 GB, **ARM64** (`linux/arm64`) — matches the measured inference
     peak (~9 GB) and the image's own build target (see the Dockerfile's ARM build note); ARM
     Fargate is also ~20% cheaper than x86 for the same size.
   - **Container**: image URI `<account-id>.dkr.ecr.eu-west-2.amazonaws.com/nged-forecast:<tag>`
     from [Step 5](#step-5-push-the-image-to-ecr); log configuration → **awslogs**, a new
     CloudWatch log group (e.g. `/ecs/nged-forecast`), region `eu-west-2`.
   - **Environment variables**: at minimum `DATA_PATH=s3://<your-bucket>/data` (see
     [setup.md's on-AWS settings](setup.md#step-3-point-settings-at-the-bucket)) — leave
     `DATA_STORE_*` unset, since the task role supplies credentials.

## Step 8 — Verify: run the task manually

There's no schedule yet, so trigger one task directly and confirm the whole path end-to-end —
this mirrors the "manual `RunTask`" verification the roadmap's
[AWS architecture](../roadmap/live-service.md#aws-architecture) plan already anticipates as the
fallback verification path regardless of the eventual always-on architecture:

```bash
aws ecs run-task \
  --cluster nged-forecast \
  --task-definition nged-forecast \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides": [{"name": "<container-name>", "command": [
    "job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job",
    "--tags", "{\"dagster/partition\": \"<key>\"}"
  ]}]}'
```

`assignPublicIp=ENABLED` (with a public subnet) is the simplest way to give the task internet
egress for its ECR pull and S3/CloudWatch calls without a NAT gateway; revisit once the
always-on control-plane box (and its VPC design) exists. Follow the run in **ECS** → the
cluster → **Tasks**, then **CloudWatch Logs** for the container's output — confirm it reaches
`load_forecaster_from_dir` and a new forecast lands under `s3://<your-bucket>/data/power_forecasts/…`.

## See also

- [Production Deployment — Design](../architecture/production-deployment.md) — why the image is
  built this way.
- [Running live forecasts end-to-end](dagster-workflow.md) — driving the promoted model day to
  day once it's live, and backfilling missed slots.
- [Environment & storage setup](setup.md) — where data tables and local artifacts live, and the
  S3 bucket + IAM policy Steps 4–8 above build on.
- [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) — the always-on
  control-plane box, scheduling, and infra-as-code, still to come.
