# ML experiments

Each folder here is one small experiment that tests one hypothesis about improving the production
forecast, named `<id>-<slug>`. The id is a five-digit integer that orders the experiments by the
order in which they were started. The folder holds `result.md` and its figures, and the front matter
of `result.md` states the hypothesis, the prediction, the decision rule, and the verdict.

The front matter of `result.md` also says whether the result is a candidate for the next model
release (`production_status`) and, once ported, which release adopted it (`adopted_in`).
Its `advice` field lists one-line imperatives for the next experiment's plan, each pointing to
its evidence, and a plan begins by reading `advice` across every `result.md`.

The [research log](research-log.md) lists every finished experiment in the order it was started,
with its question, headline result, and link. The [research plan](research-plan.md) lists the
experiments that are planned or proposed, in the order they should run.
