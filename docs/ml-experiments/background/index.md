# Background for experiment workers

**Read this page, and the pages it links, before you plan an experiment.** The pages state what
earlier experiments established about the data sets and about the scoring rules, in short bullets,
and link the page that owns each fact. The pages exist because several mistakes that cost an
experiment a re-run were facts that an earlier experiment had already found and filed on a page
the planner did not open.

## Reading order

1. **This page**: the reading order and the update rule.
2. **[NGED power telemetry](nged-power-telemetry.md)**: series, units, tables, timestamps.
3. **[Weather inputs by product](weather-inputs-by-product.md)**: which weather exists when.
4. **[Row sets and scoring](row-sets-and-scoring.md)**: row sets, metric, intervals, gating.
5. **[Problems and the best forecast](problems-and-current-best-forecast.md)**: baselines, best
   guess, scores.
6. **[Adopted and open questions](adopted-and-open-questions.md)**: what is settled, who owns the rest.
7. **[Data processing](../data-processing.md)**: steps every experiment applies to the data.
8. **[Glossary](../glossary.md)**: terms the experiment pages use without defining.

The reference page [What is on disk](what-is-on-disk.md) says where each data set sits under
`data/` and how to read it.

## Three facts that have cost experiments time

- **The `TimeSeriesMetadata` table has 33 rows, and 32 series carry data.** Count 32 series; do not
  change 32 to 33 ([NGED power telemetry](nged-power-telemetry.md#the-series)).
- **ERA5 reaches the forecast about 5 days late.** Any ERA5 value used at a lag time is a ceiling.
  Name the weather source at lag times in every write-up
  ([weather inputs](weather-inputs-by-product.md#availability-at-forecast-time)).
- **Read power through `studies.power.scan_power`.** The raw table holds rows that cleaning flags
  ([NGED power telemetry](nged-power-telemetry.md#raw-and-cleaned-tables)).

## Update rule

- **The experiment that learns or overturns a fact about the data fixes the page.** Edit the
  bullet in place so that the page describes the present, and change its "last checked" line.
- **Add or correct at most five bullets per experiment.** Write the bullet where a planner would
  look first, and link the experiment or page that owns the evidence. If nothing changed, write
  "background: no change" in the pull request body. The merger checks the pull request body for the
  list of changed bullets or for that sentence.
- **End every factual bullet with an evidence tag** in the form `(exp 179; 6 farms, one year;
  provisional)`: the experiment ids, the scope of the evidence, and a status of `established`,
  `provisional`, or `overturned by N`. A bullet about code or disk layout says `(code)` or
  `(disk)`. A bullet that digests a linked docs page says `(docs)`, and that page owns the
  evidence. A bullet whose evidence nobody has traced says `(unchecked)`.
- **Mark a bullet `established` only when it states a count or a definition, or when a second
  experiment or row set reproduced the finding.** Every other bullet is `provisional`.
- **Check a tag before you rely on it.** Open the `result.md` of each cited experiment and confirm
  that the experiment supports the claim, that the scope matches, and that the status fits. Replace
  `(unchecked)` with a traced tag whenever you find the owning source.
- **A later experiment may overturn a bullet.** The overturning experiment changes the status in the
  tag to `overturned by N`, where N is its id, and rewrites the bullet to state the present fact.
- **Do not copy a rule from a page that owns it.** Link the owner page instead.

Last checked against experiment 183.
