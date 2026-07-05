# Forecast Delivery: Delta Lake on S3

> **Status: durable architecture explainer.** This page explains *why* OCF delivers forecasts
> to NGED as Delta Lake tables on S3 rather than through a custom REST API — where that choice
> came from, what it gives NGED, and how a REST API could still fit in later. The concrete
> table schemas and their implementation status live in the
> [delivery tables](../roadmap/delivery-tables.md) page.

OCF's national solar forecast is served through a custom REST API
([quartz-api](https://github.com/openclimatefix/quartz-api)), and that's the right tool for that
product. So when we talk about NGED Flexpectation delivering files on object storage instead, it's
natural to ask why — "just files on S3" can sound like a shortcut we'd eventually replace with a
"real API". This page walks through the reasoning: what Delta Lake actually is, why the two
products suit different mechanisms, what Delta Lake on S3 gives NGED (more than it might first
appear), and when a REST API *would* earn its keep here.

## What Delta Lake is (and who else uses it)

It's worth being concrete about what Delta Lake actually *is*, because "Parquet files
on S3" can conjure an image of our code carefully hand-crafting every file and hoping nothing
goes wrong mid-write. That's not what happens. [Delta Lake](https://delta.io) is an open-source
table format created at Databricks and donated to the Linux Foundation in 2019, and — just like
a traditional database — all the fiddly bookkeeping is the library's job, not ours.

Writing a forecast run is one function call:

```python
from deltalake import write_deltalake

write_deltalake("s3://<bucket>/power_forecast", forecasts, mode="append")
```

The library ([delta-rs](https://github.com/delta-io/delta-rs), written in Rust) decides how to
split rows across files, names the files, lays out the partition directories, records per-column
statistics, and — the crucial part — atomically commits the new files to the transaction log.
(Our [`delta_store`](../api/delta_store/index.md) package layers our own storage policy on top —
row sort order, per-column parquet encodings, precision rounding — but that too is just
configuration passed to the same library call.)

On disk, the result looks like this (our development `power_forecasts` table):

```text
power_forecasts/
├── _delta_log/                    ← the transaction log
│   ├── 00000000000000000000.json
│   ├── 00000000000000000001.json
│   └── …
├── experiment_name=xgboost_cv_0001/
│   └── fold_id=mid_2025_to_mid_2026/
│       ├── part-00000-cbb0236f-….zstd.parquet
│       ├── part-00000-ef8aad4e-….zstd.parquet
│       └── …
└── …
```

Readers never touch those Parquet files directly. Opening the table is a single call —
`DeltaTable("s3://<bucket>/power_forecast")` — and the first thing that client does is read
`_delta_log`. The log tells it exactly which Parquet files make up the current version of the
table and, for each file, which partition values and value ranges it holds — so a query can
skip irrelevant files without downloading them. A file the log doesn't mention — half-uploaded,
or left behind by a failed write — simply doesn't exist as far as any reader is concerned. That
log is where the atomic-publish guarantee described
[below](#and-it-is-a-database-acid-on-object-storage) physically lives. And every tool that
reads Delta Lake starts the same way, including Polars' `scan_delta`
([below](#lazy-reads-query-it-dont-download-it)).

If this still feels unusual, remember that *every* database is ultimately files on disk.
Postgres — OCF's workhorse — stores each table as files of 8 kB pages in a binary format that
only Postgres can read, and you reach them through the Postgres server process. Delta Lake is
the same idea with the layers rearranged: the on-disk format is open and columnar, so dozens of
independently developed tools read it directly, and the "server" role is played by object
storage. In neither case does application code manage the files by hand.

And this is thoroughly mainstream technology, not an exotic bet. Databricks built its entire
platform on Delta Lake; Apple, Comcast, Adobe, and Salesforce all run it at massive scale; and —
closest to home — [NESO](https://www.neso.energy/) uses Delta Lake too. More broadly, the
open-table-format family it belongs to (Delta Lake; [Apache Iceberg](https://iceberg.apache.org/),
created at Netflix; [Apache Hudi](https://hudi.apache.org/), created at Uber) is now the standard
way large companies store and share analytical data. In other words: we picked the boring,
battle-tested option.

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

- **Novel concepts.** The [delivery tables](../roadmap/delivery-tables.md) carry information that
  has no analogue in OCF's existing products. An API surface for all of this would have to be
  designed from scratch:
    - time-varying [effective capacity](../roadmap/delivery-tables.md#table-4-effective_capacity),
    - [asset-health history](../roadmap/delivery-tables.md#table-3-asset_health_history),
    - [substation switching](../roadmap/delivery-tables.md#table-5-substation_switching),
    - [per-ensemble-member forecasts](../roadmap/delivery-tables.md#representation-1-ensemble-of-deterministic-forecasts)
      (including full backtest history & live forecasts in the same table, for multiple ML models
      run in parallel, and multiple ways of representing uncertainty),
    - and [forecast warnings](../roadmap/delivery-tables.md#table-2-power_forecast_warnings).
- **Much more data** — quantified [below](#how-big-is-flexpectations-power-forecast-data).
  Per-ensemble-member probabilistic forecasts across thousands of time series are simply a different
  order of magnitude compared to OCF's national solar forecast.
- **One user.** NGED is the only consumer. There is no multi-tenant permission problem to solve — a
  single authenticated user covers the entire requirement. But this is not a ceiling on the
  architecture: S3 can grant read access to further users via bucket policies or [S3 Access
  Points](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-points.html), so having
  multiple consumers wouldn't force us off Delta Lake on S3. A small, fixed number of views can
  even be split across separate tables — e.g. keeping forecasts for NGED's generation customers
  private while openly publishing the substation forecasts — with bucket-policy permissions set
  per table. What doesn't fit that pattern is *dynamic, per-customer* entitlements decided at
  query time rather than at data-layout time — many customers each needing their own filtered
  slice of the same table — which is where a REST API would earn its keep, as covered
  [below](#when-would-a-rest-api-earn-its-keep).
- **Power users.** NGED's analysts are comfortable in Python and want the access to the full
  firehose of data: not just the latest run, but routine access to the *entire history* of
  forecasts, backtests, and our automated analysis of NGED's power data, all for their own
  evaluation and downstream analysis.

## Evolving requirements

NGED are refreshingly open that they don't yet know exactly which views of the data they will find
most useful. We need headroom to iterate table schemas rapidly. Delta Lake supports schema evolution
directly; a REST API would add a versioning-and-deprecation cycle on top of every schema change.

And, crucially, NGED are _already_ finding uses for our "firehose of data" that we had never
considered. These are exactly the sort of unforeseen use-cases that simply wouldn't have occurred to
anyone if we only provided a minimal API to NGED. In fact, these "unforeseen use-cases" might end up
being just as important to NGED as the power forecasts.

## How big is Flexpectation's power forecast data?

Each 6-hourly forecast run produces one row per time series, ensemble member, and half-hour of the
14-day horizon. For the V1 trial area that is 32 series × 51 members × 672 half-hours ≈ **1.1
million rows per run**. Our development `power_forecasts` table already holds **~404 million rows**
from 417 backtest init times — for just 32 time series. Our
[`delta_store`](../api/delta_store/index.md) storage format packs that into **0.73 GB, ~1.8 bytes
per row**.

V2 scales to ~2,500 time series: ~78× more, or roughly **86 million rows per run** and on the
order of **100 billion rows per year of history** — a couple of hundred gigabytes at the
measured bytes-per-row, and several *terabytes* uncompressed. Serialised as uncompressed JSON
(measured on real forecast rows: ~356 bytes each), the same year of history would be roughly
**36 terabytes on the wire — about 200× the Delta footprint**. NGED wants routine access to all
of it!

That volume is an awkward fit for JSON request/response cycles (to say the least!); in practice,
REST designs for workloads like this tend to grow a "bulk export" endpoint that hands back files —
at which point the files are doing the real work, and the REST API has become a wrapper around them.

## What REST was designed for

REST was named and formalised in Roy Fielding's
[2000 PhD dissertation](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm) as
a retrospective description of *why the web worked*: many independent clients, small
hypermedia resources, stateless request/response over HTTP, caches in the middle. It is a
superb architecture for exactly that — operational queries, multi-tenant products, and
integrations where each interaction moves a small resource.

What REST wasn't designed for is bulk analytical access to terabytes of tabular history. Teams
that press it into that role tend to find themselves re-implementing, one endpoint at a time,
the things analytical storage formats already provide: pagination (chunked reads),
retry-and-resume (transactional snapshots), server-side filtering (predicate pushdown), column
selection (columnar layout), and compression. Each of those becomes a bespoke design decision
to make, document, and maintain — and a bespoke client for NGED to write against.

## We _are_ delivering data over an API (but not a _REST_ API)

It's easy to hear this design as "we're not building an API, we're just putting files on S3" — but
that framing undersells what's actually being delivered. An application programming interface (API)
is, at heart, a *contract*: a precisely specified interface that lets independent programs
interoperate. Delta Lake is exactly that — an [open, versioned protocol
specification](https://github.com/delta-io/delta/blob/master/PROTOCOL.md) with mature, independently
developed client implementations: Polars, pandas, DuckDB, Spark, Power BI, Rust, and more.

So we *are* delivering through an API. The difference is that the protocol, the client
libraries, the authentication layer (S3/IAM), and the server (S3 itself) are all off-the-shelf,
open components with far more engineering investment behind them than any service we could
build. Our part of the contract shrinks to the piece that is genuinely ours to define: the
table schemas, which are pinned down in code by the
[`contracts` package](../api/contracts/index.md).

## …and it is a database: ACID on object storage

As the [opening section](#what-delta-lake-is-and-who-else-uses-it) showed, Delta Lake is
Parquet files plus a transaction log — and that log is what elevates "files on S3" into a
database with full **ACID** guarantees:

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
  storage — and Flexpectation *consumes* its ECMWF ensemble NWP exactly that way, every day.
  Our delivery to NGED mirrors how we ourselves receive hundreds of gigabytes of upstream data.
- [Earthmover](https://earthmover.io) built a commercial data platform around the same
  principle for scientific data.
- [Source Cooperative](https://source.coop) and the
  [AWS Open Data programme](https://registry.opendata.aws/) distribute petabytes of public
  data as files on S3, not behind bespoke APIs.

Inside the Flexpectation codebase, Delta Lake on object storage is already our
[storage layer](overview.md#core-components) for power telemetry and NWP data. Delivery adds no
new technology to build, learn, or operate — the delivery tables are produced by the same
mechanism as everything else in the pipeline.

## What OCF doesn't have to build or run

Choosing Delta Lake over a bespoke REST API removes an entire service from the project:

- no API server to design, write, deploy, monitor, scale, keep patched, and pay for;
- no endpoint schema to version and document alongside the table schemas;
- no authentication microservice or token lifecycle to manage;
- no client SDK for NGED to install and for us to maintain;
- availability is S3's SLA, not something we are on call for: The on-call engineer _only_ has to
  check the forecast has run at 00, 06, 12, and 18 hours. They aren't _constantly_ on call.

For a small team whose mission is forecast quality, that's a lot of engineering effort we get
to spend on the forecasts themselves instead.

## Securing it

The bucket holds data about NGED's customers, so it is not public: it is protected with
standard S3/IAM authentication. With a single consumer this is straightforward — one
authenticated principal, no entitlement matrix — and it mirrors exactly how NGED protect their
own time-series JSON bucket, which we read the same way. Every tool named above supports
authenticated S3 access natively.

## Strict data contracts (machine-verifiable)

The project makes strict use of Patito data contracts (defined in the
[`contracts`](../api/contracts/index.md) sub-package). These enforce not just the _type_ of the
data, but also the statistical properties of the data. These contracts also serve as the
human-readable documentation for the project's data inputs and data outputs. (For example, every
public function that consumes and/or returns a DataFrame must declare the exact data contract for
that DataFrame in the function's type hints).

## And it's excellent for our internal storage too

Choosing Delta Lake for delivery is also a vote for our own infrastructure, because it's the same
format we already rely on internally — and it earns its keep there on its own merits:

- **It's remarkably compact.** The same `delta_store` write policy that packs `power_forecasts`
  to ~1.8 bytes/row ([above](#what-delta-lake-is-and-who-else-uses-it)) works just as well on
  [ECMWF ENS NWP](../api/dynamical_data/index.md): a full daily run (~7.24 million rows across
  1,671 H3 cells × 51 members × 85 lead times) averages ~113 MB, and our entire local development
  table — 810 daily runs, 1.57 billion rows, April 2024 to June 2026 — is **86 GB**.
- **It's fast to query, even on a laptop.** Row-group skipping on a member-sorted table means a
  single-ensemble-member read (the common case for training) touches only a few row groups instead
  of the whole file — measured on a real 29-day, 9-cell, control-member read: **~5× faster and ~5×
  less peak memory** (0.15 s / ~1 GB → 0.02–0.04 s / ~205 MB), for a ~2% storage cost. That speed
  is what lets us run cross-validation across every ensemble member, for every fold, on a laptop,
  in a few minutes — no cluster required for day-to-day model development.
- **It scales to parallel cloud training too.** S3 is built for very high aggregate throughput to
  many concurrent readers, so when V2 needs multiple ML training runs in parallel, each worker can
  read directly from the same S3-hosted Delta tables at full bandwidth — no shared filesystem, no
  bespoke data-loading pipeline, and no separate "training data service" to build.

There's a compounding benefit to using exactly the same storage technology everywhere: every
improvement to `delta_store`'s writer properties, sort order, or precision policy
([above](#what-delta-lake-is-and-who-else-uses-it)) pays off for training, backtesting, *and*
delivery to NGED simultaneously, instead of being three separate storage stacks to maintain.

## When would a REST API earn its keep?

REST APIs have their place, and there are futures in which this project grows one:

- **Many consumers** needing per-customer permissions — e.g. if the forecasts were productised
  beyond NGED.
- **Small operational queries from non-Python systems** — e.g. a control-room application that
  just wants "the latest forecast for substation X" as JSON over HTTP.
- **Browser-based consumers** that can't speak S3 directly. (although WASM mostly solves that)

The reassuring part is that adding a REST API later is **purely additive**: a thin, stateless
service that reads from the same Delta tables and serves slices of them over HTTP. Nothing would
have to be re-written. Nothing about the Delta-first design forecloses a REST API — which is why a
REST API sits comfortably as a [v2 stretch goal](../roadmap/index.md#v20-scale-up-future-research)
rather than a v1 requirement. The Delta tables remain the system of record either way; the API would
be a convenience layer on top, added if and when a consumer appears whose needs it fits.
