# ML experiments

Each folder here is one small experiment that tests one hypothesis about improving the production
forecast, named `<id>-<slug>`. The id is a five-digit integer that orders the experiments by the
order in which they were started. The folder holds `result.md` and its figures, and the front matter
of `result.md` states the hypothesis, the prediction, the decision rule, and the verdict.

The front matter of `result.md` also says whether the result is a candidate for the next model
release (`production_status`) and, once ported, which release adopted it (`adopted_in`). Its
`advice` field lists one-line instructions for future experiments.

The [research log](research-log.md) lists every experiment in the order it was started, with its
question, headline result, and link, and it lists the experiments that are planned.
