# ML experiments

Each folder here is one small experiment that tests one hypothesis about improving the production
forecast, named `<id>-<slug>`. The id is a five-digit integer that orders the experiments by the
order in which they were started. The folder holds `result.md` and its figures, and the front matter
of `result.md` states the hypothesis, the prediction, the decision rule, and the verdict.

When an experiment's winner is ported into production, its `result.md` also gets an `informs` field
naming the production setting it decided, and [the experiments behind the production
design](../architecture/design-evidence.md) gains a row.
