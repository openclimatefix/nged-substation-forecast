# Plan: the `study` skill posts scores to the leaderboard and runs studies overnight (#1168)

**The problem.** The `study` skill says a study's leaderboard number comes only from
`scripts/forecasting/score_study.py`, but it never says to post one, and no study has. Study results
live only on their pages. The skill also has no procedure for the phase 1 routine in
[Experiments run by an LLM agent](https://openclimatefix.github.io/nged-substation-forecast/roadmap/auto-research/#phase-1-studies-run-overnight),
in which the maintainer and Claude agree a plan by day and Claude runs it overnight.

**The solution.** Two documentation changes, and one verification. First, a step in the `study`
skill that says what to post to the leaderboard and how to name it. Second, a new short skill,
`overnight-study`, that the `study` skill points to and that holds the overnight procedure. Before
either is written, an implementer runs `score_study.py` on one arm of the Quick wins study, because
the check may show that the posting step needs a small adapter that the step must then describe.

## Base: PR #1171 merges first

[PR #1171 (Remove the scorer checks that only stop deliberate cheating)](https://github.com/openclimatefix/nged-substation-forecast/pull/1171)
is open at the time of writing and edits the same paragraph of `.claude/skills/study/SKILL.md`, the
same section of `studies/README.md`, and `score_study.py` itself. This plan describes the post-#1171
script: no `final_test_start` guard, no single-model-name check, no environment-scrubbing
re-execution, and no `NGED_ALLOW_REPLACE` gate, so `--replace` is a plain flag. **The implementer
rebases onto `main` after #1171 merges, re-reads every line this plan quotes, and starts from the
rebased text.**

## Verdict, size, and departures

- **Verdict: worth implementing, roughly as described.** The issue's premise holds: `grep` finds no
  `score_study` call in `studies/` or `packages/studies/`, and the Quick wins README says "the study
  did not submit to `score_study.py`".
- **Size: complex under the `plan-issue` rule, although the diff is documentation.** The five
  triggers:
    - What gets stored: fires. A study that follows the new step writes `study/` rows to
      `power_forecasts` and an MLflow fold run, through an existing script.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: fires, for the shape of the overnight procedure (a new
      skill, a section of the `study` skill, or a page under `docs/`).
    - Callers not nameable without searching: does not fire. The change adds text and edits no
      function.
- **Reviews bought: both plan reviews (simplicity, then correctness), run as Sonnet sub-agents.** The
  maintainer may downgrade to medium, because no code changes unless the Quick wins check finds a
  defect. If so, the correctness review is the one to keep, since the plan's account of what
  `score_study.py` accepts is the risky part.
- **Departure from the issue, one:** the issue says to write the procedure "in the `study` skill, or
  in a short skill the `study` skill points to". This plan picks the short skill, because the
  `study` skill is already 740 lines and the overnight procedure is read only by the session that
  runs overnight, not by every session that touches a study.
- **Departure from the issue, two:** the issue says to check `score_study.py` "on one arm of a
  finished study such as Quick wins". Reading the Quick wins code shows that its forecast files may
  not pass the script's checks as they stand (see "Risks and open questions"). The check is
  therefore a step of this plan with a defined outcome for each result, not a formality.

## Solution

### Walk through

#### The happy path

To post a study's scores, the session that runs a study first picks the arms to post: the reference
arm and each arm the page reports. For each arm, the study's fit script writes one parquet file of
`PowerForecast` rows, with `power_fcst_model_name`, `power_fcst_model_version`, and the six
`PowerForecast` columns that the script's `_open_predictions` does not add itself. (The script adds
`experiment_name` and `fold_id`.) Each arm becomes its own submission, `uv run python
scripts/forecasting/score_study.py <file> <study_name> mid_2025_to_mid_2026`, because the script
stores one experiment per call and the experiment is named `study/<study_name>`. The page's arms
therefore need one `<study_name>` each, and the naming rule below keeps a study's arms findable as a
group.

The script then checks the file's row keys against those of the reference experiment
(`xgboost_no_power_lags`) for the fold, writes the rows to `power_forecasts` in batches of series,
and materialises the `metrics` asset in leaderboard scope. The `metrics` asset writes the
`forecast_metrics` rows that the page quotes. The page then states each arm's `study/<study_name>`
experiment name, and the skill requires the page to carry the bias warning.

#### The naming rule

A study named `quick-wins` posts each posted arm as `study/quick-wins--<arm>` (lower-case letters,
digits, underscores, and hyphens only, up to 64 characters, which is the pattern
`validate_study_name` enforces). The page's front matter or opening table lists those names, and
each study folder's `README.md` carries the page's address, so a reader of either finds the other.
Where an arm's name would push the study name past 64 characters, the study uses a short arm code
and the page's table maps code to arm.

#### The overnight procedure

The procedure is a new skill, `.claude/skills/overnight-study/SKILL.md`, in six parts:

1. **The plan.** A file under `plans/` is the wrong place, because `plans/` holds one file per
   branch and is deleted on merge. The agreed plan is the study's own `plan.md` in `studies/<folder>/`
   (committed before the run starts), listing the planned arms, the planned contrasts, the seeds,
   the folds, the unplanned-arm budget, and the stopping rule. The commit is the proof that the plan
   preceded the run.
2. **The budget for unplanned arms.** A number the plan sets, with a default of 3 for a night. Each
   unplanned arm is named in the report with the label "post hoc", and combines arms that the plan
   named. An unplanned arm never replaces a planned one in the headline.
3. **Seeds and folds.** A planned arm gets 1 seed. An arm that the morning report would call a gain
   gets a re-run with 2 more seeds (3 in all) before the report says "gain", and the report gives
   the spread across seeds beside the paired interval. A second fold is run only where the plan
   names one, because the leaderboard has one fold today.
4. **The morning report.** One file, `report.md` in the study's data folder, of at most one screen:
   what ran, what moved (planned contrasts first, post hoc second, each with its interval), what
   failed, and up to 5 proposed ideas for the next plan, each with a rough cost.
5. **Stopping.** A fit that fails is retried once. If it fails again, the arm is recorded as failed
   in the report with the exception text, and the night continues with the remaining arms. The night
   stops when the plan's budget (wall-clock hours or arm count) is spent, never mid-fit: the current
   fit finishes, then the report is written. The session never spends money and never orders data,
   as the `study` skill already requires.
6. **What the overnight session does not do.** It posts to the leaderboard through
   `score_study.py` only, never re-submits an arm to improve a score, and does not skip the
   `study` skill's reviews: an overnight run produces results, and the next day's reviews decide
   which of them reach a page.

### The main branches off the happy path

- **A study with no forecasting arm** (a data-download or solar-geometry study) posts nothing. The
  new step in the `study` skill says so in its first sentence.
- **A study that fits on fewer series than the reference forecasts** cannot pass the row-key check.
  The step says to forecast every series, init time, lead time, and ensemble member of the
  reference experiment, and says that a study whose design uses fewer series reports its own scores
  on the page and posts nothing. The step names this case because Quick wins is in it today.
- **A re-submission of the same arm** needs `--replace`, which overwrites the earlier rows and
  fold run and raises `study_submission_number`. The skill tells the session to use it only after a
  reviewer-requested re-run, never to improve a score.

### The main error-handling paths

- **`score_study.py` refuses a file.** The script raises `RowKeyMismatchError` or `ValueError`
  before it writes anything and logs the refusal. The session fixes the file or, if the design
  cannot satisfy the check, drops the post and says why on the page. The session never edits the
  scorer, as the skill already says.
- **The Quick wins check fails.** The implementer reports the failure and the smallest adapter that
  would fix it (see below), and does not write the adapter without approval.
- **An overnight fit fails or the budget runs out.** See "Stopping" above: retry once, record, carry
  on, report.

## What changes, file by file

- **`.claude/skills/study/SKILL.md`.**
    - In the paragraph that begins "A study's leaderboard number comes only from", add the posting
      rule: which arms to post, the row-key requirement, the naming rule, and the bias warning.
    - In "The order of work", add a step between steps 3 and 4: post the arms the page will report,
      after the first run and before the first scientific-validity review, so that the reviewer sees
      the leaderboard rows. (A review-requested re-run re-posts with `--replace`.)
    - In "Writing the page", add a requirement: the page lists the `study/` experiment names and
      states that `study/` rows on the leaderboard fold are biased low, because the study chose its
      arms on that fold.
    - Add a pointer to `overnight-study` under the first paragraph's autonomy text.
- **`.claude/skills/overnight-study/SKILL.md` (new).** The six parts above, under 120 lines, with
  frontmatter naming when to load it: planning or running a study overnight.
- **`CLAUDE.md`.** Add a row for `overnight-study` to the skills table, and widen the `study` row
  only if the pointer needs it. Sweep the other summaries that list skills (the memory note "Sweep
  the summaries when a workflow changes" names `CLAUDE.md`'s table and `SKILL.md` frontmatter).
- **`studies/README.md`.** One sentence in "A study's leaderboard number comes only from
  `score_study.py`" saying that a study posts its reported arms, with a link to the `study` skill's
  step. No restatement of the rules.
- **`studies/quick_wins/README.md`.** Replace "the study did not submit to `score_study.py`" with
  the outcome of the check: either the `study/` names the posted arms carry, or one sentence on why
  the study posts nothing.
- **`docs/roadmap/auto-research.md`.** Rewrite the "Each study will post its scores to the
  leaderboard" paragraph in the present tense (the roadmap's own rule), link the `overnight-study`
  skill's procedure by its path in the repository only if the docs-link checker accepts it, and
  leave the phase 2 text alone. The status banner stays: phase 1 is still under way.
- **Possibly one script or adapter, only if the Quick wins check needs it.** See "Risks".

## Design-philosophy check

- **Production or R&D.** Everything here is R&D, which fails fast. The overnight procedure retries
  a failed fit once and then records the failure, which is the fail-fast rule applied to an
  unattended run: the failure is named in the report, not hidden.
- **No production code, asset, or asset check changes**, so the `WARN` and `blocking=False` rules do
  not apply.
- **Hypotheses.** T2.1 in `engineering-hypotheses.md` already describes the `study/` prefix and the
  promotion exclusion. The plan changes no hypothesis and renumbers none.
- **Principle traded away: none.** The bias warning is the honest price of scoring a study on the
  fold it chose its arms on, and the roadmap already states it.

## Tests

No code changes, so there is no new test, and "the assertion that fails on `main` today" does not
apply. Three mechanical checks stand in:

- **The Quick wins check** (see "Verification commands") is the only end-to-end test of the posting
  step. Its result decides whether the step is correct as written.
- **A dry-read of the naming rule**: `validate_study_name` accepts every name the step gives as an
  example. The implementer runs it on each example string.
- **If an adapter is added** (see "Risks"), it ships with a test that builds a small predictions
  frame lacking a `PowerForecast` column and fails on `main` for the missing column.

## Docs to update

`.claude/skills/study/SKILL.md`, the new skill, `CLAUDE.md`, `studies/README.md`,
`studies/quick_wins/README.md`, and `docs/roadmap/auto-research.md`. Each is written for the present
(CLAUDE.md, "Write about the present, not the past"). This issue does not complete a roadmap item,
because phase 1 is not done, so there is no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pymarkdown scan .claude/skills/overnight-study/SKILL.md .claude/skills/study/SKILL.md studies/README.md studies/quick_wins/README.md
uv run mkdocs build --strict    # and read the rendered auto-research page
uv run python scripts/check_docs_links.py
uv run pre-commit run --all-files
```

**The Quick wins check.** The development environment where this plan was written has no `data/`
folder, so the check must run on the maintainer's machine:

1. Pick one finished arm of Quick wins that stores per-run forecasts (a point-forecast arm, since a
   quantile arm stores them only when `save_samples` is set).
2. Concatenate its `forecasts/init=*.parquet` files, add `power_fcst_model_name`,
   `power_fcst_model_version`, and any other column `PowerForecast` requires, and write one parquet
   file.
3. Run `uv run python scripts/forecasting/score_study.py <file> quick-wins--<arm>
   mid_2025_to_mid_2026`.
4. Record which of the script's checks passed and which refused, and the `forecast_metrics` row.

## Risks and open questions

1. **Quick wins probably cannot post as it stands.** Reading `studies/quick_wins/fit_arms.py`: the
   forecasts carry `time_series_id`, `power_fcst_init_time`, `nwp_init_time`, `valid_time`,
   `ensemble_member`, and `power_fcst` only, and the study fits 31 series. A quantile arm renumbers
   `ensemble_member` as `member * levels + level`, so its keys cannot equal the reference's. The
   reference experiment may hold a different number of series, because the roster of the v1 pages
   is 32 series. If the check fails, I recommend editing the posting step to say "a study that
   cannot match the reference's row keys posts nothing" and filing a separate issue for an adapter,
   rather than growing this issue. **Decision for the maintainer:** accept that, or have this issue
   add a tested adapter in `packages/studies/`.
2. **One experiment per arm or one per study?** `score_study.py` stores one experiment per call, so
   a multi-arm study posts several experiments. The naming rule above (`<study>--<arm>`) keeps them
   findable. The alternative is one experiment per study holding only the headline arm, which loses
   the arms' comparison on the leaderboard. I recommend one per reported arm.
3. **Where the plan lives.** A committed `plan.md` per study folder adds a file per study. The
   alternative is the GitHub issue body. I recommend the file, because the commit orders the plan
   before the run and a session starting overnight reads the repository, not the issue.
4. **The default budget of 3 unplanned arms and 3 seeds are guesses.** They are labelled defaults
   in the skill, and the plan overrides them per night.
5. **Should the correctness reviewers' models stay Sonnet?** The maintainer's note says Sonnet by
   default while credits are low. I recommend Sonnet here, because no code changes.
