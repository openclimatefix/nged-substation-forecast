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
