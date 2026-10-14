# ML experiments

Each folder here is one small experiment that tests one hypothesis about improving the production
forecast, named `<id>-<slug>`. The id is a five-digit integer that orders the experiments by the
order in which they were started. The folder holds `result.md` and its figures.

The front matter of `result.md` states the hypothesis, the prediction, the decision rule, and the
verdict, so `grep` over `docs/ml-experiments/*/result.md` finds results. The `ml-experiment` skill
states the rules.
