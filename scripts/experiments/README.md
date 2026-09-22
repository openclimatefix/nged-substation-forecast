# Experiments

**Code in this directory is held to a lower standard than the rest of the repository, and it is
kept anyway because the findings it produced are cited elsewhere.** An experiment answers a
question once. The answer goes into `docs/`, and a reader who doubts the answer needs the code that
produced it, so the code stays where they can find and re-run it.

## What this tier promises, and what it does not

| | `packages/` and `src/` | `scripts/experiments/` |
|---|---|---|
| Runs in production | yes | never |
| Has tests | yes | no |
| Maintained as the repository changes | yes | no |
| Backwards compatibility | within reason | none |
| Linted by CI | yes | yes |
| Validated against a Patito contract | yes | no |

The one column that matches is the linting, because a script nobody can read is no more auditable
than a script nobody kept. Everything else is deliberately weaker.

**The rule that earns an experiment its place: an experiment whose findings reach `docs/` has to
merge.** A page on `main` citing a number that only an unmerged branch reproduces is an unverifiable
claim. Merging the code is what keeps the claim checkable, and it is the whole of the argument for
this directory existing.

**An experiment that produced nothing worth citing does not belong here.** Delete it, or leave it
on a branch. The directory is not an attic.

## What to expect when reading one

- **Nothing here is imported by production code.** No experiment adds a package, touches a Patito
  contract, or enters the Dagster asset graph. An experiment that needs to do any of those has
  stopped being an experiment.
- **Paths may have rotted.** Each experiment reads its data from the directory
  `DATA_PATH_INTERNAL` names, the same variable `contracts.Settings` reads, but the downloads
  themselves are not in version control and a data directory that has been cleaned out will not
  refill itself.
- **A run command in a module docstring is the tested way to run that script.** Each one names its
  own dependencies, because these scripts run under `uv run --no-project` rather than against the
  workspace environment.
- **Read the experiment's own README first.** Each directory has one, covering what the experiment
  measured, what the arms are, and which readings the result does not support.

## The experiments

| Directory | Question it answered | Where the answer lives |
|---|---|---|
| `beam_diffuse_split/` | Does a weather product's own beam/diffuse split carry information a PV forecast can use, beyond the global horizontal irradiance alone? | [Does a weather product's beam/diffuse split help a PV forecast?](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/) |
