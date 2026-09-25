# Studies

**Code in this directory is held to a lower standard than the rest of the repository, and it is
kept anyway because the findings it produced are cited elsewhere.** A study answers a question once.
The answer goes into `docs/`, and a reader who doubts the answer needs the code that produced it, so
the code stays where they can find and re-run it.

**"Study" rather than "experiment", because `experiment` already names a column.** `PowerForecast`
carries `experiment_name` and `ml_flow_experiment_id`, and the forecasts Delta table is partitioned
by `experiment_name`, so in this repository an experiment is one MLflow-tracked run of the
production pipeline. A study of whether a weather product's published field carries information is a
different kind of thing.

## What this tier promises, and what it does not

| | `packages/` and `src/` | `studies/` |
|---|---|---|
| Runs in production | yes | never |
| Has tests | yes | the machinery does, the study does not |
| Maintained as the repository changes | yes | no |
| Backwards compatibility | within reason | none |
| Linted by CI | yes | yes |
| Validated against a Patito contract | yes | no |

**The tested half is `packages/studies/`, and the split is deliberate.** A study's arms, charts and
write-up answer one question and are then done, so tests on them would have no second reader. The
machinery those arms call is different: it is used by every arm, it will be used by the next study,
and its failures are silent — a centred rolling window off by one step, a timestamp assigned to the
wrong hour, a site label derived two different ways. Code moves into `packages/studies/` when a
study has already got it wrong once, or when getting it wrong would produce a plausible-looking
number rather than an error.

Linting is the one row where both columns agree, because a script nobody can read is no more
auditable than a script nobody kept. Everything else is deliberately weaker.

**The rule that earns a study its place: a study whose findings reach `docs/` has to merge.** A page
on `main` citing a number that only an unmerged branch reproduces is an unverifiable claim. Merging
the code is what keeps the claim checkable, and it is the whole of the argument for this directory
existing.

**A study that produced nothing worth citing does not belong here.** Delete it, or leave it on a
branch. The directory is not an attic.

## What to expect when reading one

- **Nothing here is imported by production code.** No study touches a Patito contract or enters the
  Dagster asset graph, and nothing in `src/` or `packages/` imports one. A study that needs to do
  any of that has stopped being a study.
- **Paths may have rotted.** Every study's data lives under `data/studies/`, in the directory
  `DATA_PATH_INTERNAL` names — the same variable `contracts.Settings` reads. Downloaded weather sits
  in `weather/<product>/` and NGED's active network management exports in `anm/`, so a later study
  can reuse them; what one study builds from them sits in `<name>/`. None of it is in version
  control, and a data directory that has been cleaned out will not refill itself. `data/NGED/` and
  `data/NWP/` are the pipeline's own, and a study reads them rather than writing to them.
- **A run command in a module docstring is the checked way to run that script.** Each one runs
  against the workspace environment, and names with `--with` only what the lockfile does not carry.
- **Read the study's own README first.** Each directory has one, covering what the study measured,
  what the arms are, and which readings the result does not support.

## The studies

| Directory | Question it answered | Where the answer lives |
|---|---|---|
| `beam_diffuse_split/` | Does a weather product's own beam/diffuse split carry information a PV forecast can use, beyond the global horizontal irradiance alone? | [Does a weather product's beam/diffuse split help a PV forecast?](https://openclimatefix.github.io/nged-substation-forecast/studies/beam-diffuse-split/) |
| `nwp_forecast_comparison/` | At the day-ahead lead the live service delivers, and the two days after it, which forecast product, or which blend of products, gives the most accurate power forecast? | In progress — no fit has run yet; see `studies/nwp_forecast_comparison/README.md` |
