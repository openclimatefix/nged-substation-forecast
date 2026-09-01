# Our Approach to MLOps

A short explainer about this project's approach to MLOps tooling: what this project's
experiment automation changes, and why it makes the path to production safer rather than
riskier.

Modern MLOps (as used in this project) changes two things:

1. **Throughput.** The grunt work of an experiment — assembling features, training,
   cross-validating, recording results — is automated, so a small team can run hundreds of
   experiments per month instead of one or two. Humans are still the ones deciding what to try;
   the infrastructure runs those experiments. (See [Running an ML experiment
   end-to-end](dagster-workflow.md).) That the throughput produces a better forecast is a bet
   this project is making rather than a result the literature has settled: the
   energy-forecasting review found [no study measuring what adopting
   machine-learning-operations practice
   delivers](../background/energy-forecasting-review.md#the-field-describes-good-practice-but-does-not-measure-whether-it-works),
   and the case for [fast, comparable
   iteration](../background/energy-forecasting-review.md#the-case-for-fast-comparable-iteration-is-argument-and-testimony)
   rests on a structural argument and on practitioner testimony instead.
2. **No translation gap.** The artifact we experimented on *is* the artifact we deploy.
   There is no "now rewrite the research code for production" step, because every experiment
   runs on the exact same code as the production pipeline from the start. The gap is closed by
   raising research to the production standard, not by lowering production to accept a research
   notebook: an idea can be explored anywhere, but it only becomes a runnable experiment once
   it lives in the pipeline's own code. The debt this avoids has a name: [Sculley et al.
   (2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
   call the code written to bridge research and production *glue code*, and the tangle that
   glue code grows into *pipeline jungles*. Their list of debts also constrains how the gap is
   allowed to close, because *dead experimental codepaths* is what accrues when experiments run
   as conditional branches inside production code: an experiment here selects a different
   [configuration](model-configuration.md) and runs the same code path, rather than adding a
   branch the production pipeline then has to carry.

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
candidate has already been trained, cross-validated (see [Cross-validation
folds](cross-validation-folds.md)), and evaluated on the same pipeline, under the same
standardised protocol, as every model before it. Holding that protocol fixed is what makes two
experiments comparable at all, which is the reason [Karpathy's
autoresearch](https://github.com/karpathy/autoresearch) fixes every training run to the same
5-minute budget: a fixed budget "makes experiments directly comparable regardless of what the
agent changes".

That is what makes a one-command promotion *safe* to press rather than merely quick. The
largest risk in a conventional setup — that the artifact measured and the artifact deployed are
two different pieces of code — does not exist here; the comparison that picked the winner was
made against every other candidate on identical folds; and the way back to the previous
champion is a single command too. A fast promotion route that nobody trusts enough to use is
worth no more than a slow one. The speed comes from the protocol rather than from haste, a
distinction [Karpathy (2019)](https://karpathy.github.io/2019/04/25/recipe/) puts bluntly: "a
'fast and furious' approach to training neural networks does not work and only leads to
suffering".
