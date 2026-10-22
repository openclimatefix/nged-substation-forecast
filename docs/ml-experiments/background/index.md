# Background for experiment workers

**Read this page, and the pages it links, before you plan an experiment.** The pages state what
earlier experiments established about the data sets and about the scoring rules, in short bullets,
and link the page that owns each fact. They exist because most mistakes that cost an experiment a
re-run were facts that an earlier experiment had already found and filed on a page the planner did
not open.

## Reading order

1. **This page**, for the reading order and the update rule.
2. **[NGED power telemetry](nged-power-telemetry.md)**: the series, their units, the raw and
   cleaned tables, and the timestamp convention.
3. **[Weather inputs by product](weather-inputs-by-product.md)**: which weather product exists at
   forecast time, which exists only in hindsight, and the rule for choosing a training product.
4. **[Row sets and scoring](row-sets-and-scoring.md)**: the named row sets, the metric, the
   interval conventions, the tail and curtailment reporting, and the retraining defaults.
5. **[What is on disk](what-is-on-disk.md)**: where each data set sits under `data/`, and how to
   read it.
6. **[Data processing](../data-processing.md)**: the processing steps that every experiment applies
   to the power readings, the switching log, and the capacity estimates. This page links it and
   does not copy it.
7. **[Glossary](../glossary.md)** for any term the experiment pages use without defining.

## Five facts that have cost experiments time

- **The `TimeSeriesMetadata` table has 33 rows, and 32 series carry data.** Count 32 series; do not
  change 32 to 33 ([NGED power telemetry](nged-power-telemetry.md#the-series)).
- **Read power through `studies.power.scan_power`.** The raw table holds rows that cleaning flags,
  such as a transformer that was switched out and read exactly 0
  ([NGED power telemetry](nged-power-telemetry.md#raw-and-cleaned-tables)).
- **ERA5 reaches the forecast about 5 days late.** Any ERA5 value used at a lag time is a ceiling.
  Name the weather source at lag times in every write-up
  ([weather inputs](weather-inputs-by-product.md#availability-at-forecast-time)).
- **Scores on different row sets, quantile levels, or series cannot be compared.** State the
  `row_set_id`, and take the benchmark from the same row set
  ([row sets and scoring](row-sets-and-scoring.md#row-sets)).
- **Power timestamps mark the end of the half-hour.** Do not shift them
  ([NGED power telemetry](nged-power-telemetry.md#timestamps)).

## Update rule

- **The experiment that learns or overturns a fact about the data fixes the page.** Edit the
  bullet in place so that the page describes the present, and change its "last checked" line.
- **Add or correct at most five bullets per experiment.** Write the bullet where a planner would
  look first, state its scope, and link the experiment or page that owns the evidence. If nothing
  changed, write "background: no change" in the pull request body.
- **A later experiment may overturn a bullet.** Every bullet states the scope of its evidence for
  that reason.
- **Do not copy a rule from a page that owns it.** Link the owner page instead.

Last checked against experiment 183.
