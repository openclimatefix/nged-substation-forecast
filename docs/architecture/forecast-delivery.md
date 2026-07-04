# Forecast Delivery: Delta Lake on S3

> **Status: durable architecture explainer.** This page explains *why* OCF delivers forecasts
> to NGED as Delta Lake tables on S3 rather than through a custom REST API — where that choice
> came from, what it gives NGED, and how a REST API could still fit in later. The concrete
> table schemas and their implementation status live in the
> [delivery tables](../roadmap/delivery-tables.md) page.

OCF has real experience with both delivery styles: our national solar forecast is served
through a custom REST API ([quartz-api](https://github.com/openclimatefix/quartz-api)), and
that's the right tool for that product. So when NGED Flexpectation delivers files on object
storage instead, it's natural to ask why — "just files on S3" can sound like a shortcut we'd
eventually replace with a real API. This page walks through the reasoning: why the two products
suit different mechanisms, what Delta Lake on S3 actually gives NGED (more than it might first
appear), and when a REST API *would* earn its keep here.

## Two products, two shapes of problem

A REST API is the right answer for OCF's commercial solar forecast because of the shape of that
problem:

- **Many consumers.** Multiple paying customers, each needing their own credentials,
  entitlements, and usage tracking — fine-grained permissions are a core requirement.
- **Small payloads.** The typical query is "give me the latest national forecast": a few
  kilobytes of JSON per request.
- **URL-shaped consumers.** Customers integrate the forecast into their own applications and
  want an HTTP endpoint, not a database.

NGED Flexpectation turns out to look quite different on each of those axes:

- **One user.** NGED is the only consumer. There is no multi-tenant permission problem to
  solve — a single authenticated principal covers the entire requirement.
- **Power users.** NGED's analysts are comfortable in Python and want the full firehose: not
  just the latest run, but routine access to the *entire history* of forecasts and backtests,
  for their own evaluation and downstream analysis.
- **Much more data** — quantified below. Per-ensemble-member probabilistic forecasts across
  thousands of time series are simply a different order of magnitude from a national
  point forecast.
- **Novel concepts.** The [delivery tables](../roadmap/delivery-tables.md) carry information
  that has no analogue in OCF's existing products: per-ensemble-member forecasts,
  [forecast warnings](../roadmap/delivery-tables.md#table-2-power_forecast_warnings),
  time-varying [effective capacity](../roadmap/delivery-tables.md#table-4-effective_capacity),
  [asset-health history](../roadmap/delivery-tables.md#table-3-asset_health_history), and
  [substation switching](../roadmap/delivery-tables.md#table-5-substation_switching). An API
  surface for all of this would have to be designed from scratch.
- **Evolving requirements.** NGED are refreshingly open that they don't yet know exactly which
  views of the data they will find most useful. We need headroom to iterate table schemas
  rapidly. Delta Lake supports schema evolution directly; a REST API would add a
  versioning-and-deprecation cycle on top of every schema change.

### How big is the data?

Each 6-hourly forecast run produces one row per time series, ensemble member, and half-hour of
the 14-day horizon. For the V1 trial area that is 32 series × 51 members × ~672 half-hours ≈
**1.1 million rows per run**. Our development `power_forecasts` table already holds **~404
million rows (~6 GB compressed)** from 417 backtest init times — for just 32 time series.

V2 scales to ~2,500 time series: ~78× more, or roughly **86 million rows per run** and on the
order of **100 billion rows — more than a terabyte — per year of history**. NGED wants routine
access to all of it. That volume is an awkward fit for JSON request/response cycles; in
practice, REST designs for workloads like this tend to grow a "bulk export" endpoint that hands
back files — at which point the files are doing the real work, and the API has become a
wrapper around them.

## A very short history of REST

REST was named and formalised in Roy Fielding's
[2000 PhD dissertation](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm) as
a retrospective description of *why the web worked*: many independent clients, small
hypermedia resources, stateless request/response over HTTP, caches in the middle. It is a
superb architecture for exactly that — operational queries, multi-tenant products, and
integrations where each interaction moves a small resource.

What it wasn't designed for is bulk analytical access to terabytes of tabular history. Teams
that press it into that role tend to find themselves re-implementing, one endpoint at a time,
the things analytical storage formats already provide: pagination (chunked reads),
retry-and-resume (transactional snapshots), server-side filtering (predicate pushdown), column
selection (columnar layout), and compression. Each of those becomes a bespoke design decision
to make, document, and maintain — and a bespoke client for NGED to write against.

## "An API" is not the same thing as "a REST API"

It's easy to hear this design as "we're not building an API, we're just putting files on S3" —
but that framing undersells what's actually being delivered. An API is, at heart, a *contract*:
a precisely specified interface that lets independent programs interoperate. Delta Lake is
exactly that — an
[open, versioned protocol specification](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)
with mature, independently developed client implementations: Polars, pandas, DuckDB, Spark,
Power BI, Rust, and more.

So we *are* delivering through an API. The difference is that the protocol, the client
libraries, the authentication layer (S3/IAM), and the server (S3 itself) are all off-the-shelf,
open components with far more engineering investment behind them than any service we could
build. Our part of the contract shrinks to the piece that is genuinely ours to define: the
table schemas, which are pinned down in code by the
[`contracts` package](../api/contracts/index.md).

## …and it is a database: ACID on object storage

Delta Lake is Parquet files plus a transaction log, and that log is what elevates "files on S3"
into a database with full **ACID** guarantees:

- **Atomicity** — each 6-hourly publish either becomes fully visible or not at all. NGED can
  never read a half-written forecast.
- **Consistency** — every committed version of the table satisfies its schema; readers never
  see mixed or partial states.
- **Isolation** — readers are never blocked by, or corrupted by, a concurrent write. A query
  that starts mid-publish sees the previous complete version.
- **Durability** — committed data lives on S3, with its
  [eleven-nines object durability](https://aws.amazon.com/s3/faqs/#Durability_.26_Data_Protection).

On top of ACID, the transaction log gives **time-travel** (read the table exactly as it was at
any past version — useful for auditing what NGED saw at a given moment) and **schema
evolution** (add columns without breaking existing readers).

## Lazy reads: query it, don't download it

Delivering the full history does *not* mean NGED downloads the full history. Reading the table
is one line:

```python
forecasts = polars.scan_delta("s3://<bucket>/power_forecast")
```

`scan_delta` is lazy: nothing is fetched until a query runs, and then only the bytes that query
touches cross the wire. Three mechanisms make that efficient:

- **Partition pruning** — filters on partition columns skip whole directories of files.
- **Row-group skipping** — Parquet stores min/max statistics per row group, so filters skip
  chunks within files.
- **Column pruning** — Parquet is columnar, so unselected columns are never read at all.

The practical effect: fetching a single forecast run from a multi-billion-row table takes a
fraction of a second, because only a few megabytes actually move. NGED writes **zero custom
data-reading code** — they point existing tools (Polars, pandas, DuckDB, Power BI, Excel) at
the bucket and query it like a database. The same mechanism powers our own pipeline: the
[lazy evaluation strategy](overview.md#lazy-evaluation-strategy) that keeps our training memory
bounded is exactly what keeps NGED's reads cheap.

## An established industry pattern

We're also in good company. "Analytical data as cloud-optimised files on object storage,
queried in place" has quietly become one of the standard ways to ship large datasets:

- [Dynamical.org](https://dynamical.org) publishes global weather datasets as Zarr on object
  storage — and this project *consumes* its ECMWF ensemble NWP exactly that way, every day.
  Our delivery to NGED mirrors how we ourselves receive hundreds of gigabytes of upstream data.
- [Earthmover](https://earthmover.io) built a commercial data platform around the same
  principle for scientific data.
- [Source Cooperative](https://source.coop) and the
  [AWS Open Data programme](https://registry.opendata.aws/) distribute petabytes of public
  data as files on S3, not behind bespoke APIs.

Internally, Delta Lake on object storage is already our
[storage layer](overview.md#core-components) for power telemetry and NWP data. Delivery adds no
new technology to build, learn, or operate — the delivery tables are produced by the same
mechanism as everything else in the pipeline.

## What we don't have to build or run

Choosing Delta Lake over a bespoke REST API removes an entire service from the project:

- no API server to design, write, deploy, monitor, scale, and keep patched;
- no endpoint schema to version and document alongside the table schemas;
- no authentication microservice or token lifecycle to manage;
- no client SDK for NGED to install and for us to maintain;
- availability is S3's SLA, not something we are on call for.

For a small team whose mission is forecast quality, that's a lot of engineering effort we get
to spend on the forecasts themselves instead.

## Securing it

The bucket holds data about NGED's customers, so it is not public: it is protected with
standard S3/IAM authentication. With a single consumer this is straightforward — one
authenticated principal, no entitlement matrix — and it mirrors exactly how NGED protect their
own time-series JSON bucket, which we read the same way. Every tool named above supports
authenticated S3 access natively.

## When would a REST API earn its keep?

REST APIs have their place, and there are futures in which this project grows one:

- **Many consumers** needing per-customer permissions — e.g. if the forecasts were productised
  beyond NGED.
- **Small operational queries from non-Python systems** — e.g. a control-room application that
  just wants "the latest forecast for substation X" as JSON over HTTP.
- **Browser-based consumers** that can't speak S3 directly.

The reassuring part is that adding one later is **purely additive**: a thin, stateless service
that reads from the same Delta tables and serves slices of them over HTTP. Nothing about the
Delta-first design forecloses it — which is why a REST API sits comfortably as a
[v2 stretch goal](../roadmap/index.md#v20-scale-up-future-research) rather than a v1
requirement. The Delta tables remain the system of record either way; the API would be a
convenience layer on top, added if and when a consumer appears whose needs it fits.
