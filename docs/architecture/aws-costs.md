# AWS Running Costs

What the live service costs to run on AWS: the costed estimate for the v1 deployment as built
(32 time series), and a projected estimate at v2 scale (~2,500 time series). The architecture
these costs price — one small always-on control-plane VM dispatching every run to an ephemeral
Fargate task — is described in [Production Deployment — Design](production-deployment.md); the
alternatives it was costed against, with their per-option prices, are recorded in
[Live service: AWS architecture](../roadmap/live-service.md#aws-architecture).

## Pricing basis

All prices are for **eu-west-2 (London)** — the nearest AWS region to the UK — taken from
the AWS price-list API (2026-07-03). AWS bills in USD; figures here are converted at
**\$1 = £0.75** (ECB rate, 2026-07-03). Why we deploy in eu-west-2 rather than the slightly
cheaper eu-west-1 is a data-transfer question, not a compute-price question — the reasoning
and the region price comparison are in
[Forecast Delivery: Securing it](forecast-delivery.md#securing-it).

The Fargate rates used throughout: x86 £0.0349/vCPU-hr + £0.0038/GB-hr; **ARM 20% cheaper**
(£0.0279 + £0.0031), and polars/XGBoost run fine on arm64, so every task below is ARM.

**These estimates exclude ML experimentation but include routine re-training.** For v1 we
expect most — maybe all — experimental training to run on our own laptops; the only
experimentation AWS sees is an optional UI-launched backtest, priced per run below. (At v2
scale that assumption weakens — see
[Backtests and training at v2](#backtests-and-training-scale-with-forecast-rows-too).)
Re-training the *deployed* model is different: we plan to re-train it roughly **once a week**,
probably on a GPU, and both estimates below carry a line for it. Fargate offers no GPUs, so
the weekly re-train is priced as an ephemeral EC2 GPU instance (start, train, promote the new
model, terminate); the GPU rates are from the price-list API on 2026-07-23, converted at the
same \$1 = £0.75.

## v1 (32 time series): ~£25–35/month

Estimated total for the deployed architecture: **~£25–35/month** (nudging ~£40 only if the
polling-and-ingest line lands at its pessimistic top), made up of:

| Component | £/month |
|---|---|
| Always-on control plane (EC2 `t4g.medium` + 20 GB EBS) | 15–22 (1-yr reserved vs on-demand) |
| Live Fargate inference (4 runs/day) + hourly polling & ingest | 5–12 |
| S3 storage (~100 GB) + requests | 2–4 |
| Data transfer (ingress + egress) | ≈0 |
| Weekly re-training (planned; ephemeral GPU instance) | ~£1 |
| Backtests | ~£0.65 per run, as needed |

The weekly re-train row is a *planned* addition rather than an as-built cost: v0.1 bakes a
frozen champion model into the image, and any re-training still happens on a laptop. Once the
weekly re-train moves to AWS, the arithmetic stays small: the v1 model trains in ~10 minutes
on a laptop, so budgeting ~30 minutes of `g4dn.xlarge` (1× T4 GPU, 4 vCPU / 16 GB;
\$0.615/hr → £0.46/hr on-demand) per weekly run — instance provisioning and data reads
included — is ~£0.25/run, about **£1/month**.

The roadmap's [access phasing](../roadmap/live-service.md#access-phasing) (team read-only
Dagster access, then a public dashboard) adds negligible spend on top of this estimate:
Stage 2's Caddy and oauth2-proxy, and Stage 3's wake-proxy, all run as plain processes on the
existing control-plane box — nothing new billable beyond a DNS domain and hosted zone
(pennies per month). The only material new billable resource across all three stages is the
second Fargate task/service for Stage 3's public Marimo instance, roughly priced by the
same-shaped workload in
[Option D](../roadmap/live-service.md#option-d-serverless-control-plane-no-pets-4145month)
(~£6.20/month for Marimo as its own tiny Fargate service).

### Workload model

Live cadence: `ecmwf_ens` 1/day (daily 00Z partition), `power_time_series_and_metadata` 4/day,
`live_forecasts` 4/day (6-hourly partitions), the monitoring `metrics(production_monitoring)`
step ~4/day (planned —
[#224](https://github.com/openclimatefix/nged-substation-forecast/issues/224)) →
**~13 materialisations/day ≈ 395/month**. This cadence ingests only ECMWF ENS and
the NGED power feed today; a near-real-time ERA5/ERA5T ingest would join it *only if* live capacity
estimation is made to depend on ERA5 — a new external dependency we may prefer to avoid by
[keeping ERA5 offline](../roadmap/capacity-estimation.md#irradiance-inputs). A backtest
experiment today is ~4–6 materialisations (`eligible_time_series` + `trained_cv_model` +
`cv_power_forecasts` per fold, plus `metrics`; single leaderboard fold), rising to ~15–20 under
the future multi-yearly-fold epoch.

Fargate compute: a right-sized 4 vCPU / 16 GB ARM task (measured inference peak ~9 GB) is
£0.16/hr → 4 × 15-min live runs/day ≈ **£5/month**; a 2-hour 8 vCPU / 32 GB ARM backtest ≈
**£0.65/run**. The hourly polling wake-ups add ~£2/month at their billed floor of about one
minute each — but every wake-up also bills image pull and worker startup, and the real ingest
runs share the same oversized task definition, so
[Production Deployment — Design](production-deployment.md#running-the-data-ingest-runs-on-the-control-plane-vm)
bounds the whole hourly-task workload at ~\$5–10/month (£4–7.50). The runbook's closing
Cost Explorer check is what settles this line.

### Storage & data transfer

These costs are the same under every architecture option the roadmap decision considered, and
at v1 scale come to **~£2–4/month in total**:

- **S3 Standard storage — ~£2/month.** The working set is ~100 GB, dominated by NWP — the
  entire v1 `power_forecasts` table, ~407 M rows of it, packs into under 1 GB (see
  [How big is Flexpectation's power forecast data?](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data))
  — at £0.018/GB-month → ~£1.80/month, plus headroom for Delta version history between
  vacuums. Grows ~40 GB/year as daily NWP partitions accumulate.
- **S3 requests — ~£0.50–1.50/month.** Delta Lake is request-heavy (transaction-log JSON
  reads, checkpoints, many small parquet GETs per scan), but ~13 materialisations/day is
  tiny volume: a generous 1–2 M GET (£0.00031/1k) + 100–200 k PUT/COPY/POST/LIST
  (£0.0040/1k) per month lands well under £1.50.
- **Data transfer — ≈£0/month.** Ingress is free (the daily NWP download from Dynamical
  costs nothing on the AWS side); S3 ↔ Fargate/EC2 traffic within eu-west-2 is free;
  internet egress (the Tailscale-tunnelled Dagster UI and Marimo dashboard) is a few
  GB/month, inside AWS's account-wide 100 GB/month free egress allowance (£0.067/GB
  beyond).
- **Everything else — pennies.** ECR image storage and CloudWatch Logs ingestion for ~700
  task runs/month are each well under £0.50/month.

## v2 scale (~2,500 time series): projected ~£70–140/month

> **Status: projection, not a measurement.** The v1 figures above are anchored in measured
> task sizes and observed run times; nothing has yet run at v2 scale. The figures below scale
> the measured v1 workload by the components that grow with the series count and hold fixed
> the components that don't, with each assumption stated inline. The inference task size and
> per-run wall time are the projections to re-measure first when v2-scale runs begin.

Scaling from 32 to ~2,500 time series multiplies the series count by ~78×, but most of the
bill does not follow it:

- **NWP is unchanged.** The ECMWF ENS download covers the whole GB grid regardless of how many
  time series consume it — the same daily download, the same ingest compute, and the same
  ~40 GB/year of Delta growth.
- **The control plane is unchanged.** The box coordinates runs but does no per-series work —
  keeping pipeline compute off the box was a design requirement precisely so that v2 scaling
  would be
  [a task-size change, not a box resize](production-deployment.md#running-the-data-ingest-runs-on-the-control-plane-vm).
- **Power-telemetry ingest grows 78× from a tiny base.** 2,500 series at half-hourly
  resolution is ~120k rows/day — trivial for both compute and storage, so the hourly polling
  cost barely moves.
- **What genuinely scales is the forecast itself.** Each 6-hourly run grows from ~1.1 M rows
  (32 series × 51 ensemble members × 672 half-hours) to **~86 M rows** — the row arithmetic is
  in
  [How big is Flexpectation's power forecast data?](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data)
  — dragging inference compute and forecast storage with it.

Projected total: **~£70–140/month** — roughly 3–4× the v1 bill, dominated by inference
compute:

| Component | £/month |
|---|---|
| Always-on control plane (unchanged) | 15–22 |
| Live Fargate inference (4 runs/day, larger task) | 45–90 |
| Hourly polling wake-ups (unchanged) | ~2 |
| S3 storage + requests (first v2 year) | 5–10, growing ~£5/month with each further year of history |
| Data transfer | ≈0 (assuming NGED read via eu-west-2 compute) |
| Weekly re-training (heavier model, ephemeral GPU instance) | ~2–15 |
| Backtests | very roughly £10–50 per run, as needed |

### Inference compute — the dominant uncertainty

A v1 live run takes ~15 min on a 4 vCPU / 16 GB ARM task (~£0.04/run, ~£5/month). Much of
that quarter-hour is work that does *not* scale with the series count — task provisioning,
reading the day's NWP partitions, the GB-wide spatial join — while the row-proportional part
(feature engineering and per-series prediction over the ~1.1 M-row `AllFeatures` frame) is a
small slice of it that multiplies by ~78×.

Bracketing that split on a 16 vCPU / 96 GB ARM task (£0.74/hour at the
[rates above](#pricing-basis)): if ~1 min of the v1 run is row-proportional, a v2 run is
~30 min (£0.37/run → **~£45/month**); if ~2.5 min is, a v2 run is ~60 min (£0.74/run →
**~£90/month**). Either way the run fits comfortably inside the 6-hourly cadence.

Memory is the other axis: 86 M rows of `AllFeatures` is roughly 17 GB for the feature columns
alone, several times that with join intermediates. If a single frame doesn't fit a 96 GB task,
the fallback is chunking inference into per-series batches — Fargate's ARM pricing is linear
in vCPU and GB, so a smaller task running proportionally longer costs about the same, and the
choice can be made empirically once v2-scale data exists.

### Storage grows linearly with accumulated history

Live v2 forecasts add ~86 M rows/run × 4 runs/day ≈ **125 billion rows/year**, which at the
measured ~1.8 bytes/row is **~225 GB/year** of Delta growth — each year of accumulated v2
history adds ~£4/month at £0.018/GB-month, and NWP accumulation adds a further ~40 GB/year
(~£0.75/month per year). Starting from the ~100 GB v1 working set, the first v2 year
therefore averages ~£4/month of storage and ends at ~£7/month, plus £1–3/month of
requests (commit cadence is unchanged, but each forecast commit writes more parquet files).
Left unpruned this grows without bound; if it ever matters, the lever is lifecycle policy on
old forecast history (colder S3 tiers or deletion), not a format change.

### Data transfer depends on how NGED read

S3 → AWS compute in the same region is free even across accounts, so NGED reading the
delivery bucket via their own eu-west-2 compute (Athena, Glue, EC2) costs nothing — that is
the read path v2 scale points towards anyway, and a load-bearing part of
[why we deploy in eu-west-2](forecast-delivery.md#securing-it). Reads over the public
internet instead cost £0.067/GB beyond the account-wide 100 GB/month free allowance — a full
pull of one year of v2 forecast history (~225 GB) would be ~£8–15 depending on how much of
that month's free allowance is left, so routine bulk reads that way would make this line item
real rather than ≈0.

### Backtests and training scale with forecast rows too

A v1 backtest is ~£0.65/run (2 hours, 8 vCPU / 32 GB). Its compute is dominated by the same
row-proportional work as inference, so a v2-scale backtest lands **very roughly at £10–50 per
run** (strict 78× linear scaling gives ~£50; the fixed NWP-reading share pulls it below
that). At those prices a heavy experimentation month becomes a real line item — and training
2,500 per-series models may also outgrow the laptop assumption in the
[pricing basis](#pricing-basis) — so re-cost this properly before planning sustained v2
experimentation.

The weekly re-train scales the same way. We may also adopt somewhat heavier models at v2, so
the ~10-minute v1 laptop train (~£1/month as an AWS GPU run — see the
[v1 estimate](#v1-32-time-series-2535month)) grows on both axes at once. Bracketing one to
four hours per weekly run on a single-GPU instance — `g4dn.xlarge` (1× T4, £0.46/hr) up to
`g6.xlarge` (1× L4, \$1.02/hr → £0.77/hr) — gives ~£0.50–3/run, or **~£2–15/month**. That
range is wide because both the model family and its training time are still open; like the
inference figures above, it is a bracket to re-measure, not a commitment.

## See also

- [Production Deployment — Design](production-deployment.md) — the architecture these costs
  price, and why it was chosen.
- [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) — the costed
  decision record: the accepted option and the four rejected alternatives, with per-option
  prices.
- [Forecast Delivery: Securing it](forecast-delivery.md#securing-it) — the eu-west-1 vs
  eu-west-2 price comparison and the data-transfer reasoning behind the region choice.
- [Setting up the live service on AWS](../live_service/aws.md) — the bring-up runbook whose
  final verification step checks Cost Explorer against this page.
