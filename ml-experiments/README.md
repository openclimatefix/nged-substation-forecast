# `ml-experiments`

**One folder per experiment, `<id>-<slug>`, holding the scripts that run it.** Each experiment tests
one hypothesis about improving the production forecast. Its write-up is
`docs/ml-experiments/<id>-<slug>/result.md`, and its predictions are under
`data/ml-experiments/<id>-<slug>/predictions/`. The `ml-experiment` skill (`.claude/skills/`) states
the rules, and `packages/studies/tests/test_study_boundaries.py` enforces the import rules, which
are the same as for `studies/`: a script may import the folder of another experiment or study.
