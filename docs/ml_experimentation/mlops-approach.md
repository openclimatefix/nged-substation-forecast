# Our Approach to MLops

A short explainer for readers who are new to modern MLops tooling: what this project's
experiment automation changes, and why it makes the path to production safer rather than
riskier.

Modern MLops changes two things:

1. **Throughput.** The grunt work of an experiment — assembling features, training,
   cross-validating, recording results — is automated, so a small team can run hundreds of
   experiments per month instead of one or two. Humans are still the ones deciding what to
   try; the machinery does the running. (See
   [Running an ML experiment end-to-end](dagster-workflow.md).)
2. **No translation gap.** The artifact we experimented on *is* the artifact we deploy. There
   is no "now rewrite the research code for production" step, because every experiment runs on
   the production pipeline from the start.

## An analogy

Traditional ML R&D is a chef inventing dishes in their home kitchen: every winning recipe has
to be laboriously re-created on the restaurant's equipment before it can go on the menu, and
much is lost (or silently changed) in translation. We are building the restaurant where R&D
happens on the service line itself: hundreds of tastings a month, every dish judged by the
same tasting panel, and the winning dish on the menu the same night — because nothing about it
needs translating.

## Nothing gets rewritten on the way to production

The model that wins the evaluation is, bit for bit, the model we deploy — not a
re-implementation of it. Promotion to production takes minutes, and that speed is a
*consequence* of rigour, not a trade against it: by the time promotion is on the table, the
candidate has already been trained, cross-validated (see
[Cross-validation folds](cross-validation-folds.md)), and evaluated on the same pipeline,
under the same standardised protocol, as every model before it.

In this world, a notebook's job shrinks to deciding what to try. Notebooks are for thinking;
the pipeline is for running.
