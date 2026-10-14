# Reconstructing the study's feature implementations

The study uses isolated package sources at the base commit and the two pinned PR heads below.
The commands reconstruct wind, radiation and combined treatments without modifying the study
checkout's production packages. Run them from the repository root of a clone containing all
three commit objects.
`git cat-file` checks their availability; fetching a newer PR head does not establish a match.

```bash
set -euo pipefail
STUDY_SOURCE_ROOT="$(git rev-parse --show-toplevel)"
test "$(pwd -P)" = "$STUDY_SOURCE_ROOT"
STUDY_RUNTIME_ROOT="$STUDY_SOURCE_ROOT/.claude/worktrees/reconstructed-fixed-pipeline"
test -f "$STUDY_SOURCE_ROOT/studies/quick_wins/fixed_pipeline_runtime/combined-integration.patch"
test ! -e "$STUDY_RUNTIME_ROOT"
mkdir -p "$STUDY_RUNTIME_ROOT"
git cat-file -e 4e820d24c0129c8c95c567ff7355abff8d416144^{commit}
git cat-file -e 38025f4eab457b9605b8c124006609869dbfca0d^{commit}
git cat-file -e 0f094c356da82e75c4014664ad7fb33fe1df8b7f^{commit}
git diff 4e820d24c0129c8c95c567ff7355abff8d416144...38025f4eab457b9605b8c124006609869dbfca0d -- packages/contracts/src packages/ml_core/src > "$STUDY_RUNTIME_ROOT/wind.patch"
git diff 4e820d24c0129c8c95c567ff7355abff8d416144...0f094c356da82e75c4014664ad7fb33fe1df8b7f -- packages/contracts/src packages/ml_core/src > "$STUDY_RUNTIME_ROOT/radiation.patch"
git diff 4e820d24c0129c8c95c567ff7355abff8d416144...0f094c356da82e75c4014664ad7fb33fe1df8b7f -- packages/ml_core/src/ml_core/features/_nwp.py packages/ml_core/src/ml_core/features/_radiation.py > "$STUDY_RUNTIME_ROOT/radiation-without-callsite.patch"
git worktree add --detach "$STUDY_RUNTIME_ROOT/wind" 4e820d24c0129c8c95c567ff7355abff8d416144
git worktree add --detach "$STUDY_RUNTIME_ROOT/radiation" 4e820d24c0129c8c95c567ff7355abff8d416144
git worktree add --detach "$STUDY_RUNTIME_ROOT/combined" 4e820d24c0129c8c95c567ff7355abff8d416144
git -C "$STUDY_RUNTIME_ROOT/wind" apply "$STUDY_RUNTIME_ROOT/wind.patch"
git -C "$STUDY_RUNTIME_ROOT/radiation" apply "$STUDY_RUNTIME_ROOT/radiation.patch"
git -C "$STUDY_RUNTIME_ROOT/combined" apply "$STUDY_RUNTIME_ROOT/wind.patch"
git -C "$STUDY_RUNTIME_ROOT/combined" apply "$STUDY_RUNTIME_ROOT/radiation-without-callsite.patch"
git -C "$STUDY_RUNTIME_ROOT/combined" apply "$STUDY_SOURCE_ROOT/studies/quick_wins/fixed_pipeline_runtime/combined-integration.patch"
```

The standalone PR package diffs match the corresponding study runtimes byte for byte.
The combined treatment applies the wind diff, the radiation resampler modules and the small
call-site integration patch. It supplies solar coordinates before radiation resampling and derives
wind speed/bearing after resampling components. Tests are outside these source-only patches.

`all-source-hashes.json` records every Python source file under the selected contracts and
ml_core package roots for each treatment, including unchanged files. Compare reconstructed files
against the corresponding treatment mapping before fitting. The published integration patch's
SHA-256 is `fca0456fde5f6009290908c3c9b895bba285b4af7cdfa98972745a0adcc116b3`.
It omits one blank context line from the executed patch, whose SHA-256 was
`23f4f76a30687b1c741b897273aba67fb5319dc43b6e51fdf062dc5083cf47a9`.
The removed context line changes no source code; reconstructed source hashes must still match.

Select the treatment before Python starts, for example:

```bash
export PYTHONPATH="$STUDY_RUNTIME_ROOT/combined/packages/contracts/src:$STUDY_RUNTIME_ROOT/combined/packages/ml_core/src"
```

A source-hash match verifies package source, not dependencies, frozen data, hardware or numerical
outputs. The frame and fit manifests retain those separate provenance checks. The component
archive must pass the full migration readiness gate before a wind or combined build; these source
reconstruction commands supply no data and do not replace that gate.

The executed JavaScript Object Notation (JSON) specifications are listed in
[`../fixed_pipeline_specs/index.json`](../fixed_pipeline_specs/index.json). Separate
[`../fixed_pipeline_runtime_records/index.json`](../fixed_pipeline_runtime_records/index.json)
records describe configurations recovered from completed provenance. A derived record is not an
original execution specification or a transcript of historical shell arguments. Both indexes bind
each published file to its SHA-256 and executed provenance.

The Part 2 diagnostic specifications and records have their own indexes under
[`../../forecast_calibration/diagnostic_specs/`](../../forecast_calibration/diagnostic_specs/index.json)
and [`../../forecast_calibration/diagnostic_runtime_records/`](../../forecast_calibration/diagnostic_runtime_records/index.json).
The indexed records cover the existing diagnostics; calibration fitting and coherence experiments
remain on hold.

Historical precipitation-era specifications retain their exact executed bytes. Their
`execution_gate` text saying precipitation was wholly missing before 13 November 2024 refers to
**categorical precipitation type**. Precipitation amount/rate has different availability limitations;
the study page describes the amount/rate limitations separately.

Commit `af5201f2` preserves the requested-feature producer, audit, and chart sources used by the
saved outputs. Later type-only corrections preserve calculations but change source hashes. Use
the recorded revision when reproducing those source-bound artifacts, and write to new output
directories. The private data, fitted boosters, and row-level forecasts remain necessary inputs.
