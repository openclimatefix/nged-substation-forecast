# Review record for the wind/radiation forecast study

The repository study skill requires independent reviews before execution. Available Codex
agents performed these reviews; the requested Opus and Sonnet agents were unavailable.

Two independent plan reviews preceded implementation. All findings were accepted: explicit
study-date bounds, common pinball scoring across sample constructions, exact reference recipes,
padding and ladder rules, precipitation-era limitations, isolated runtimes, compatible wind
archives, public production fitting, and native temporal context before clipping.

A separate runtime-integration review checked the selective PR patches and the manually combined
feature-engineering call. Its requested native bulk/single radiation test was retained. Later PR
review updates changed comments in the executed package code; those updates were applied too.

The fresh builder/fitter review cleared the builder before its first real execution. Wind builds
remain behind the full migration-readiness gate. Three fitter findings were accepted and rechecked:
reject incomplete pilot frames, require the complete validation-file inventory, and fingerprint
the scoring and production implementations. The fitter was cleared before its first fit. Effective
booster parameters are saved, and execution caps preparation threads through the environment.

The fresh reporter review found five blockers, all accepted and independently rechecked before
execution: immutable report outputs, verified completion hashes and report provenance, finite
loss checks, lead-time pairing, and all individual pinball levels plus actual feature lists.
Further accepted checks cover positive capacities, timestamp-derived leads, provisional status,
and metric-specific units. Coverage differences remain descriptive.

The separate Part 2 plan received two independent reviews. Both identified ambiguity between
nine fitted quantile knots and a nine-atom empirical distribution. The revised plan uses native
nine-level scoring, separately reports rearrangement, leaves unsupported 90% control-member
coverage pending, and specifies its empirical-CDF diagnostic. Both reviewers cleared the revision.

The Part 2 scorer received an independent pre-execution review and was cleared. Publication
checks must disclose sparse month support, link the parent recipe/feature tables, and list
pending seed combinations. Synthetic checks exercised scoring and aggregation; missing-target
handling was code-reviewed rather than separately tested in the initial synthetic fixture.

A fresh audit/chart review found circular-bearing differences, null chart values and incomplete
text/artifact provenance. Accepted fixes use shortest angular distances, reject null or reversed
intervals, export exact helper text and annotations, hash outputs, and inspect a PNG rendered
from the optimised SVG. The timing check now injects delayed radiation and recomputes the
correlations. Synthetic checks recovered the expected one-hour movement. Both scripts' code
and chart text export were cleared; concrete figure text still requires review before rendering.

Scientific-validity reviews of results, chart/prose/persona reviews and final evidence checks
remain pending until completed results and a draft page exist. This record makes no publication
recommendation while those checks remain pending.

A preliminary scientific review independently reproduced the old-pipeline score aggregates,
checked sampled loss calculations and lag cutoffs, and found a metadata issue: a reloaded
production booster reports runtime defaults for some training parameters. Predictions were
unaffected. Future fits label that configuration as runtime metadata and separately save the
production training parameters. The reporter now reads the hash-verified production metadata,
prints historical quantile levels, and verifies matching raw training keys and, where required,
cleaned training counts. These changes received a fresh review before execution; completed
forecasts remain immutable. The Part 2 source-code allowlist includes the reviewed metadata-only
fitter revision. Old-reference contrasts are reproduction diagnostics rather than preregistered
deciding contrasts. This preliminary review does not replace either final scientific review.

The Part 2 reviewer subsequently ran a separate missing-target synthetic probe: one unsupported
target was excluded and the supported target retained all 51 samples. A publication evidence
audit corrected the pending May cycle inventory boundary to 12 May 2026 at 06Z. A report-only
addition prints delivered interval ranks and qualified continuous-iid reference coverage, leaves
the native fitted knots outside that reference, and lists pending fitting-seed combinations.
A second agent independently cleared these additions before execution and verified that scoring
calculations were unchanged and the new table enters the output hash inventory.

An independently authored publication-table extractor received a fresh pre-execution review.
It selects and formats existing aggregate reports without recalculating statistics, checks
filters and finite values, rejects known private/row-level fields, and records immutable output
provenance. Its basename allowlist is not proof that arbitrary files are public; publication
specifications must point only to the reviewed aggregate report outputs.

A fresh review cleared the optional four-treatment interaction, computed as combined minus
radiation minus wind plus old with the original case counts and shared month/seed draws.
An accepted blocker now checks each treatment role against its frame manifest. Synthetic
probes recovered zero for additive effects and the injected nonzero interaction. Both reporters
also disclose sparse month support and draws omitting unsupported series; these diagnostics use
the existing draws and leave the estimators and random-number calls unchanged.

The required studies suite passed with 3,338 tests and five skips in 238.06 seconds. An earlier
sandbox run stalled in Zarr because local socket sends were blocked; an isolated reproduction
confirmed the restriction. The affected 71 tests passed in 2.54 seconds with that restriction
lifted, followed by the full passing suite. No tests were changed or excluded to obtain the pass.

The planned precipitation-era diagnostic received a fresh reporter review before execution.
Optional era selections recover exact NWP initialisation times from the hash-verified, complete
saved forecast inventory, cover every scored key, and preserve source-time pairing. They use
the recorded 13 November 2024 availability boundary and remain descriptive because season and
training exposure differ. Reports without an era selection retain their existing calculations.

Before Part 1b fitting, a fresh reviewer cleared the recipe inventory's synthetic preparation,
historical feature order, cuts/additions and duplicate padding. Accepted provenance additions
hash the production feature YAML and pooled encoder source and identify the site identifier as
Float32 numeric. The inventory reads no study observations and fits no models.

## Canonical training-key guard

The radiation report refused its first pairing because the frame builder's IPC digest includes
Arrow chunk layout. An independent audit found exact schema, row order and key values in both
folds despite differing chunk boundaries. Completed manifests and fits remain unchanged.
The reporter now accepts explicit frame paths, verifies the fit-pinned manifest and training-file
hash, and compares canonically rechunked five-column training keys. It still rejects missing,
duplicate, reordered or genuinely different keys and differing cleaned-training counts.
A fresh review by `reference_results_science_review` found no blockers before execution
(commit `4f1def67`). The fallback serialized-key guard remains strict when frame paths are absent.

A separate fresh reviewer cleared the point-control total-gain reporter before execution.
Synthetic probes checked gain mapping, zero fill, normalization and invalid-input refusals.
The accepted provenance improvement records the fitted script and dependency hashes directly;
the reviewer checked those fields against the fitter manifest and cleared the addition.
Shares are normalized per booster, then averaged equally across series and six fold/seed refits.
The reported range is refit variability, not an interval or evidence that a feature helps.

The separate importance-chart script received a fresh pre-execution review. It validates the
complete 24-feature public aggregates, preserves zeros on a linear axis, exports exact figure
text before rendering, and describes its ranges as refit variability. Concrete text reviews and
PNG inspection remain required. The same reviewer cleared the serial corrected-fit queue, with
execution conditional on full wind readiness, all four completed frame builds and a free GPU.

A fresh pre-execution review of the Part 2 coverage chart accepted its report binding, sample
selection and mean-only dots. The absolute diagnostic does not report coverage confidence
intervals, so the chart invents none. An accepted sizing blocker now reserves the standard label
column within the 680-pixel figure. An accepted provenance addition hashes shared chart/theme
sources and records rendering-library versions. The reviewer cleared both fixes before text
export. Concrete figure-text reviews and rendered inspection remain pending.

A fresh reproducibility review textually reconstructed all three runtime treatments from pinned
PR package diffs and the published integration patch. Every one of the 71 package Python source
files matched the source-hash inventory and actual study runtimes. No target source was executed.
Accepted command hardening checks the repository root and required patch, stops on errors and
rejects an existing reconstruction directory. The reviewer cleared that addition. Source matches
remain distinct from data, dependency, hardware and numerical-output provenance.

The radiation figure's first-reader review could not follow refitting, forecast leads, predicted
distributions and the term paired estimate. Revised subtitles explain those concepts before use.
A small chart-label change describes the dot as a mean difference on matched forecasts and the
interval as repeated draws of whole months and one matched training seed. A fresh static review
confirmed that both text export and rendering use the changed constant and that it matches the
reporter's sampling method. No scores changed. Subsequent first-reader and rule-by-rule prose
reviews cleared the final radiation figure text before rendering; its optimised SVG and PNG
were inspected. A fresh code review also cleared direct two-decimal numeric labels, avoiding
double rounding through the shared utility's three-decimal intermediate values.

All four wind/combined frame builds passed their boundary checks and pinned destination 924,
power 70 and capacity 1 before the reviewed corrected-fit queue began. The producer's wrapper
remained stale, so readiness was established from the complete independent migration gate as
recorded in the plan. No producer files were changed.

The root freshly reviewed all four Part 2 execution configurations: combined-pipeline production
and exact quick-wins quantile winner, three seeds, both folds and both hyperparameter settings.
Each configuration requires six completed fits before the reviewed scorer can run. The point
reference remains conditional on the actual G3 result.

A separate agent prepared the conditional twelve-fit best-point queue, and the root reviewed
every line before execution. The queue checks both completed combined frame contexts and every
absent destination before launching any fit, preserves the public recipe and environment, and
requires an explicit root G3 result checkpoint. A free GPU slot and an exited preceding queue
remain execution prerequisites. This preparation is not evidence that G3 passed or that a point
fit ran.

The first independent Part 2 scientific review examined the completed Fold A main report and
independently checked saved-run issuance, lag availability, encoded sample labels, aggregate hashes,
finite-rank references and synthetic diagnostics. It required no fit, frame or scoring rerun.
Its wording correction was accepted: the page now distinguishes the three-seed central estimate,
the bootstrap's individual-refit variation and the t-interval's within-month seed averaging.
Results must keep the unresolved pooling loss effect separate from its lower observed coverage
and narrower intervals, and distinguish solar's 80%/90% findings from wind's. The completed
report does not establish calibration, joint coherence or an isolated effect of the input fixes.
The fresh final scientific review remains required after other folds/settings and the page finish.

A focused independent science review explained the small difference between the parent and
Part 2 intervals for the same delivered-recipe contrast. Both correctly resample whole months
and one shared individual fitting seed, but consume the deterministic random stream in a
different order. Their central estimates and separate equal-month summaries agree; the finite
bootstrap percentile endpoints differ through Monte Carlo variation. No rerun was required.
The pages disclose this difference rather than implying identical resample sequences.

The initial coverage figure's text received fresh static reviews before each revised export.
An isolated first reader required definitions before the first use of quantiles, an explanation
of the main setting's role, and plain wording for finite-sample expected coverage. All three
changes were accepted and the revised text cleared. Rule-by-rule prose review and first rendered
inspection remain pending; no rendered figure is claimed by these text checks.

The separate rule-by-rule coverage-caption review accepted a final grammatical correction,
which received fresh static review before export. The reviewer cleared the exact revised text
before its first render. The root inspected the PNG made from the optimised SVG: all eight
coverage means, interval labels, nominal lines and explanatory text are visible without overlap
or clipping. The Fold A main SVG and PNG are included as the initial Part 2 figure.

The corrected-pipeline headline and radiation figures received a fresh static review of their
numeric-label placement and exported axis text. Labels now sit beneath their row labels, clear
of the interval marks and zero reference. Isolated first-reader and rule-by-rule prose reviews
cleared the final population caption and actual axis title before rendering. The root inspected
both final PNGs: numeric labels, marks, reference lines and axis units are visible without
clipping or overlap. Their final public assets use headline text/render v8 and radiation
text/render v14; earlier draft figures are not publication inputs.

The importance report completed on the six corrected cleaned point-control refits. Its final
caption changes received fresh static review before export, followed by isolated first-reader
and rule-by-rule prose reviews. The root inspected the final v3 PNG before copying the SVG and
PNG into the page assets. All 24 columns appear for solar and wind, including zero-gain columns;
refit ranges remain descriptive variability, not confidence intervals or evidence of benefit.

The root reviewed and executed the B-main aggregate-table specification after its complete
matched report became available. Both main-fold G3 rows fall below the preregistered 0.15-point
absolute fair-CRPS threshold, despite their negative equal-month t intervals. The conditional
best-point refit was therefore not triggered; its prepared queue is not an executed study input.
Part 2's corresponding contrast remains unrun for this explicit gate decision.

The root reviewed all seven prepared Part 1b supplementary selector configurations before any
execution, including their source paths, exact planned labels, type/lead/metric filters, displayed
columns and distinct bootstrap/equal-month summaries. Only completed, actually used sources may
be exported. The long extracts belong in supplementary files; they do not silently combine
folds or settings or turn precipitation availability eras into causal comparisons.

The root also reviewed all 24 sensitivity report specifications against their main counterparts.
Static reversal checks confirmed that only each fitting path's setting segment and the top-level
setting changed; frames, seeds, labels, pairs, type restrictions and training-match checks remain
identical. Inherited preparation and main-stage gate notes are historical, non-operational
metadata. Actual sensitivity execution requires its corresponding completed sensitivity fits
and the recorded root main-results checkpoint; none of these prepared reports has yet run.

The final headline and radiation captions were shortened because the earlier explanatory text
hid their plots below the initial screen. Unused weather-member, quantile and CDF discussion was
removed together with the terms it explained. Exact scoring, periods, within-row weighting,
random-seed meaning and full-history capacity scope remain explicit. Fresh first-reader and
rule-by-rule prose checks required a plain explanation of pinball loss and an unambiguous
separation of matched forecasts from equal meter/refit weights; both were accepted. The root
reviewed every specification change before exporting text and inspected the final rendered PNGs.
The publication assets now use headline text/render v10 and radiation text/render v17.

The root reviewed the private two-worker report orchestration and every entry of its 52-report
inventory before launch. Two accepted corrections use the verified virtualenv interpreter
without dependency resolution and preserve errors in exclusive private logs rather than discard
them. Each worker waits for completed frames and fits, runs reports serially with eight threads,
and refuses existing outputs or automatic retries. The workers are running and waiting for
Part 1b fits; their launch does not authorise a sensitivity fit before its main-results gate.

The Part 2 supplementary selectors also passed fresh root review before execution: delivered
CDF-bin shares, lead diagnostics and descriptive November availability-era summaries retain
both recipes, exact scope labels and support counts. Their sensitivity sources are still pending.

The importance figure now follows the study skill's logarithmic-axis rule for shares spanning
orders of magnitude. The root freshly reviewed the display-only clamp, fixed 1–2–5 ticks and
caption changes before execution. Means and range endpoints below 0.01%, including zero, appear
at that floor; raw report shares and ranges remain unchanged. Isolated first-reader and
rule-by-rule prose reviews cleared the exact floor explanation. Rendered inspection caught
suppressed and then overlapping tick labels; freshly reviewed axis-label settings corrected
both without changing text or data. Final text v4/render v6 shows all tick labels and all 24
features clearly. The public assets supersede the earlier linear figure.

The complete study suite passed with 3,338 tests passed and five skipped in 210.55 seconds.
Subsequent code edits affect only axis-label visibility and alignment, verified by rendering.
Repository-wide Ruff checks and formatting checks passed for all 942 Python files, and the
repository-wide type check passed.

Fold A sensitivity reports are now complete for both studies. The root reviewed the exact
aggregate publication selectors before execution and checked the Part 2 draft's inserted tables,
support statements, equal-month intervals and zero-crossing scope against the completed reports.
The pooled-nine versus native-nine loss effect remains unresolved in both Fold A settings.

The root freshly reviewed optional per-row report sources in the loss-chart script before any
execution. Exact contrast selection and reviewed-text guards remain in force; provenance now
hashes every used report before and after export. A compatibility text export was byte-identical
to the previously reviewed headline text. Ruff and formatting checks passed; the required study
suite is running again because this change affects chart source selection. The new wind-addback
specification remains a preparation draft and has not been rendered or interpreted.

That study suite completed with 3,338 tests passed and five skipped in 212.53 seconds.

The root reviewed the exact unaffected-type prediction verifier, first privately and then as a
committed study script, before execution. It checks every saved forecast's complete keys and
exact unaffected values, binds complete frame/fit inventories, and rechecks hashes afterward.
An explicit privacy exception guard suppresses library errors that could expose private keys.
Ruff, formatting and type checks passed. Its first execution awaits completed lean and solar fits.

The root reviewed the aggregate input-audit table-source allowlist and all six selectors before
execution. Existing private-column and selection guards remain unchanged. The supplementary
page preserves the exact selected domain counts, precipitation missingness and timing rows,
including lead/era support. Unsupported paired-finite change shares were omitted rather than
filled with zero. The introduction distinguishes forecast input rows from distinct observed
hours and the midnight May grouping marker from the deferred 06:00 operational-cycle comparison.

A further pre-execution cross-check caught a mismatch between the verifier and the fitter's
completion inventory: the fitter fingerprints `manifest.json` separately and omits it from
`complete.json`'s artifact list. The root confirmed the producer's code and reviewed the correction
before any verifier run. Both manifest files are now checked separately and included in the exact
on-disk inventory. No fitting or scoring result required a rerun.

The supplementary feature registry preserves the inventory script's exact ordered lists and
counts, with an explicit warning that configured conditional recipes were not necessarily fitted.
The new study index links describe scope without extending any result claim.

An interim full-text privacy scan found short numeric identifier collisions in ordinary code,
aggregate counts and SVG geometry. Redacted source-field and line diagnostics distinguished these
from names, spatial indexes and private free text, for which no matches were found. Three rounded
polygon-coordinate tokens coincided with the Fold B G3 statistics; the root verified their
independent origin in the aggregate scoring report. These are context checks, rather than a
blanket exemption for numeric identifiers. A final scan and manual figure inspection remain due.

After the baseline training queue exited successfully and an independent GPU check found no
active compute process, the root launched the freshly reviewed 114-fit main feature wrapper.
Its pinned first-lean, first-solar and independent-addition queues remain separate from the
conditional later solar rungs and the sensitivity gate. The Part 2 Fold B sensitivity diagnostic
and both reviewed aggregate publication extracts completed. The parent Fold B sensitivity scorer
is running. A separate required study-suite run follows the verifier inventory correction.

The private publication-copy helper was reviewed again after adding source-specific, hash-bound
numeric collision diagnostics and exclusive mode-0600 diagnostic output. The root launched only
its redacted preflight mode; no public configuration copying or numeric exemption was authorized.
