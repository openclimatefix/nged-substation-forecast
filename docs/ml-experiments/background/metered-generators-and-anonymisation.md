# Metered generators and anonymisation

**NGED meters 6 solar farms, 3 wind farms, and 3 other generator or storage series in the trial area,
and the output of every one of them may leave the private data store only under an anonymous
label.** This page lists the series, the export-cap record for the one generator under active
network management, and what a plot or pull request may show. Public data about generators is
treated differently, and the last section says how.

## The series

- **The 32 series that carry data include 6 solar farms, 3 wind farms, and 3 other generators or
  storage units.** The three are one battery energy storage system, one biofuel generator, and one
  series of type "Other (Generation)" (disk).
- **Of the 3 wind series, 1 is metered in MW and 2 in MVA** (disk).
- **A type of "Other (Generation)" need not be solar.** Do not assume a daylight profile (exp 182; one
  farm behind D8; provisional).
- **Read a farm's cleaned series from `cleaned_solar_farms.parquet` in
  `data/ml-experiments/00179-*/` and keep rows with a null `drop_reason`.** The file has no site
  column, so join `studies.pv_dataset.pv_sites()` on `time_series_id` for the label, the
  coordinates, and the capacity (exp 183; 6 farms; provisional).
- **Farms miss 7.6% to 27.6% of their half-hours.** Check coverage before comparing farms (exp 179; 6
  farms; provisional).
- **A flat top in a farm's output is not proof of clipping.** The cause is not settled (exp 179,
  183, 218; 6 farms; provisional). See the [metered solar farms
  theme](../themes/metered-solar-farms/index.md).
- **A near-zero reading at a wind farm is not always low wind.** Of the half-hours above 1.5 times
  cut-in wind speed, 1.6%, 13%, and 30% sit in the idle band at W1, W2, and W3, and ERA5 wind speed
  finds only part of them (exp 210; 3 wind farms; provisional).
- **UKV wind lowered the error of the three wind farms by about 0.2 points of capacity at 24 to 54
  hours ahead**, using a research archive and not the live feed, so the gain is an upper limit (exp 54;
  3 wind farms; small gain; provisional). See the [generator forecasting
  theme](../themes/generator-forecasting/index.md).

## Active network management and the curtailment feed

- **One generator in the trial area is connected under active network management (ANM), and NGED has
  confirmed there are no others.** A generator with no cap record therefore ran free
  ([network page](../../background/network.md#active-network-management-caps-what-a-generator-may-export);
  docs).
- **Two records exist and they are different quantities.** The setpoint history is one row per change
  of the export cap, in MW. The `curtailment/` feed on NGED's S3 bucket is a derived volume of MW lost
  ([data sources](../../roadmap/data-sources.md#what-the-curtailment-feed-holds); docs).
- **The setpoint history is the one to use.** The cap's sign, its start date, and the month the
  scheme goes live are on the [network page](../../background/network.md#active-network-management-caps-what-a-generator-may-export)
  (docs).
- **The export sits in `data/studies/downloads/observations/NGED-ANM/`**, and `studies.export_cap`
  joins it onto an experiment's grid. It drops capped hours from training and keeps them at scoring
  (disk).
- **The clamp uses the cap in force, which a day-ahead forecast does not have**, so it is an upper
  bound on what the cap could buy (code).
- **Curtailment at substations is scored by cost, not by this feed.** The scoring code is a worked
  example on 2 bulk supply points and could not rank forecasts at the headline limit (exp 84; 2 BSPs,
  one scored year; provisional). See the [curtailment cost theme](../themes/curtailment-cost/index.md).

## What may be published

- **Never publish an NGED metered generator's time series with its name or identifier.** The rule
  covers charts, tables, leaderboards, dashboards, reports, and issue and pull request bodies. The
  full rule is in `CLAUDE.md` (docs).
- **Use the labels from `studies.anonymise`: "A" to "F" for the solar farms and W1 to W3 for the
  wind farms.** `site_labels_for` assigns them under a fixed seed, and it raises on a different
  count of generators (code).
- **Put no calendar dates on the axis of a generator's time series, and publish no output in MW.**
  Normalise by the generator's own capacity. The `dataviz` skill's anonymisation section and the
  `study` skill's anonymisation section own the rest (docs).
- **Keep any plot of one farm's time series, and the cleaned series, under `data/ml-experiments/`.**
  The private tables stay there (docs).
- **D1 to D22 label demand-side series, not generators in general.** The [glossary](../glossary.md#d-codes)
  defines them, and substations may be named (docs).

## Public generator data

- **Time series from public sources may be named.** The settled output of Balancing Mechanism Units
  in Elexon's published data may be shown with names, in MW, on calendar dates. A study that mixes
  public and NGED generators anonymises the NGED ones (docs).
- **The public-data studies are the [solar BMU census](../../studies/solar-bmu-census.md) and the
  [solar BMU disaggregation](../../studies/solar-bmu-disaggregation.md).** (docs)

Last checked against experiment 218.
