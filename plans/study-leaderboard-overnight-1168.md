# Plan: the `study` skill posts scores to the leaderboard and runs studies overnight (#1168)

**The problem.** The `study` skill says a study's leaderboard number comes only from
`scripts/forecasting/score_study.py`, but it never says to post one, and no study has. Study results
live only on their pages. The skill also has no procedure for the phase 1 routine in
[Experiments run by an LLM agent](https://openclimatefix.github.io/nged-substation-forecast/roadmap/auto-research/#phase-1-studies-run-overnight),
in which the maintainer and Claude agree a plan by day and Claude runs it overnight.

**The solution.** Documentation changes in three files, and one check. The `study` skill gains a
posting step and a short "Running a study overnight" section. The roadmap page and the Quick wins
README are brought in line. Before the text is written, an implementer runs `score_study.py` on one
arm of the Quick wins study. The correctness review found that Quick wins cannot pass the row-key
check, so the check becomes an offline comparison that confirms why.

## Base: PR #1171 merges first

[PR #1171 (Remove the scorer checks that only stop deliberate cheating)](https://github.com/openclimatefix/nged-substation-forecast/pull/1171)
edits the same paragraph of `.claude/skills/study/SKILL.md` and `score_study.py` itself. This plan
describes the post-#1171 script (no single-model-name check, no environment re-execution, `--replace`
a plain flag). **The implementer rebases onto `main` after #1171 merges and re-reads every line this
plan quotes.**

## Verdict, size, and departures

- **Verdict: worth implementing, roughly as described.** `grep` finds no `score_study` call in
  `studies/` or `packages/studies/`, and the Quick wins README says "the study did not submit to
  `score_study.py`".
- **Size: complex under the `plan-issue` rule, although the diff is documentation.** The issue sized
  itself medium with the same two triggers firing; the rule says any firing trigger makes the issue
  complex, so the plan follows the rule and the maintainer may downgrade it.
    - What gets stored: fires, because a study that follows the new step writes `study/` rows.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: fires, for the shape of the overnight procedure.
    - Callers not nameable without searching: does not fire.
- **Reviews: both plan reviews, the simplicity review (Sonnet) and the correctness review (Opus).**
- **Departure from the issue:** none of substance. The issue lets the procedure live in the `study`
  skill or in a short skill it points to. This plan picks the `study` skill, because the overnight
  session must load that skill anyway.

## Solution

### Walk through

#### The happy path

To post a study's scores, the session first picks the arms to post: the study's control arm and
each arm the page reports. (The skill says "control arm", because the scorer's "reference" means
`xgboost_no_power_lags`.) For each arm, the study's fit script writes one parquet file holding
exactly the `PowerForecast` columns the script accepts and no others (`power`, `arm`, and `seed`
columns are refused): `valid_time`, `time_series_id`, `ensemble_member` (Int8), `power_fcst_init_time`,
`power_fcst` (Float32), `power_fcst_model_name`, and `power_fcst_model_version`. The file has one row
per series, init time, lead time, and ensemble member of the reference experiment
`xgboost_no_power_lags`, because `score_study.py` refuses a file whose row keys differ. A study
therefore has to forecast every daily run of the validation year, about 7 times the compute of a
study that samples one run a week. Each arm is
a separate call, `uv run python scripts/forecasting/score_study.py <file> <study_name>
mid_2025_to_mid_2026`, because the script stores one experiment per call. The study name is
`<page-slug>--<arm>`, so the experiment `study/<page-slug>--<arm>` is found from the page by its
slug and the page from the experiment. The skill gives the normalisation: lower-case the arm name,
map `.` to `_`, and keep the whole name within 64 characters (`validate_study_name`'s pattern is
`^[a-z0-9_-]{1,64}$`; the longest page slug is 27 characters, which leaves 35 for the arm). Each
posted arm is the arm's seed 0 fit, and the page says so, because the page's intervals average over
three seeds and the leaderboard row holds one.

The script writes the rows to `power_forecasts` and runs the `metrics` asset in leaderboard scope.
The `metrics` asset writes the `forecast_metrics` rows that the page quotes. The page lists each
posted experiment name and says that `study/` rows on the leaderboard fold are biased low, because
the study chose its arms on that fold.

**The overnight session** runs a plan that the maintainer and Claude agreed the previous day. The
plan is the study's `plan-issue` plan, committed before the run starts, and it names the planned
arms and contrasts and the budget for unplanned arms. The session runs the planned arms, then up to
the budgeted number of unplanned arms that combine the night's winners, each labelled post hoc. An
arm that the morning report would call a gain is re-run with more seeds, or on a second fold where
the plan names one, before the report says "gain". The session writes a short morning report: what
ran, what moved, what failed, and the ideas it proposes for the next plan.

### The main branches off the happy path

- **A study with no forecasting arm** (a data download, solar geometry) posts nothing. The step says
  so in its first sentence.
- **A study that cannot match the reference's row keys** posts nothing and reports its own scores on
  the page. Quick wins is in this case today, because it forecasts one run every 7 days.
- **A study whose native output is not postable**, such as a quantile arm that renumbers
  `ensemble_member` to 459 codes (an Int32, where `PowerForecast` has an Int8), posts nothing.
  Collapsing it to 51 members would change the forecast being scored.
- **A study whose arms were fitted past the fold's `train_end`**, such as month-block folds, matches
  the row keys and is still in sample. It refits on data up to `train_end` or posts nothing.
- **A re-submission of the same arm** needs `--replace`. The step allows it after a
  reviewer-requested re-run, and after a failure that happened once the rows were written (see
  below). It does not allow it to improve a score. Any new attempt, a new name included, raises
  `study_submission_number`.

### The main error-handling paths

- **`score_study.py` refuses a file.** The pre-write checks raise before the script writes anything,
  and the script logs the refusal. The session fixes the file or, if the design cannot satisfy the check, posts nothing and
  says why on the page. The session never edits the scorer, as the skill already says.
- **The script fails after the write.** `_score_study` writes the partition before it materialises
  `metrics`, so a failure in `metrics` or MLflow leaves the rows stored and the log saying
  "refused". The retry uses `--replace`.
- **An overnight fit fails.** The session records the failure and the exception text in the morning
  report and carries on with the remaining arms.
- **The night's budget runs out.** The session stops between fits, never mid-fit, then writes the
  report.

## What changes, file by file

- **`.claude/skills/study/SKILL.md`.**
    - Extend the paragraph "A study's leaderboard number comes only from `score_study.py`" with the
      posting rule: which arms to post, the `<page-slug>--<arm>` name, the one-sentence case of a
      study that cannot match the reference's row keys, and `--replace` only after a requested
      re-run.
    - Add a step to "The order of work" after step 5: post the arms the page reports, once a
      reviewer finds nothing more to fix and before step 6 finishes the page. Posting earlier would
      force `--replace` re-posts after each reviewer-requested re-run, and would put numbers on the
      leaderboard before the evidence reviewer has checked the fit stopped at `train_end`.
    - Add a bullet to "Writing the page": list the `study/` experiment names and state the bias
      warning once.
    - Add a section "Running a study overnight" after the opening autonomy text, in about 25 lines,
      with the five parts the issue lists: the plan, the budget for unplanned arms, seeds and folds,
      the morning report, and stopping. The plan states its own numbers (arm budget, seeds). The
      skill ties "gain" to the existing rule: a contrast is a gain when its month-and-seed
      resampled interval over the three seeds excludes zero. A "second fold" can only be a
      study-internal fold, because `conf/cv/default.yaml` has one leaderboard fold. A post hoc
      arm that combines winners keeps the equal-column rule and `colsample_bytree=1`. Posting N arms
      re-scores the reference N times, and that counts against the night's budget.
- **`docs/roadmap/auto-research.md`.** Rewrite the paragraph "Each study will post its scores to the
  leaderboard" in the present tense, naming "the `study` skill" in words. Also rewrite the sentence
  at lines 207-210 ("The leaderboard scorer will keep ... will drop ..."), which PR #1171 makes
  stale and does not touch. Leave the phase 2 text and the status banner alone.
- **`studies/quick_wins/README.md`, and `docs/studies/forecasts/quick-wins.md` at lines 208 and
  639.** Say why the study posts nothing: it forecasts one run every 7 days (about 53 runs) and
  the leaderboard scores every daily run, and its quantile arms renumber `ensemble_member`. The
  page's wording "did not submit to the scorer" stays true, so only add the reason.

Not changed: `CLAUDE.md` (no new skill), `studies/README.md` (the `study` skill owns the rule), and
any script. A helper in `packages/studies/` that builds the `PowerForecast` file and calls the
script would give every study one posting path. It adds code, tests, and review, and the Quick wins
check will show whether it is justified, so it is a separate issue if wanted.

## Design-philosophy check

- **Production or R&D.** Everything here is R&D, which fails fast. A failed overnight fit is named in
  the report and not hidden.
- **No production code, asset, or asset check changes**, so the `WARN` and `blocking=False` rules do
  not apply.
- **Hypotheses.** T2.1 in `engineering-hypotheses.md` already describes the `study/` prefix and the
  promotion exclusion. The plan changes no hypothesis.
- **No principle is traded away.** The bias warning states the cost of scoring a study on the fold it
  chose its arms on, which the roadmap already states.

## Tests

No code changes, so there is no new test. Two checks stand in:

- **The Quick wins check** is the end-to-end test of the posting step. Its result decides the Quick
  wins README sentence.
- **Name check:** `validate_study_name` accepts every example name the skill gives, including one
  built from an arm name with a dot, such as `colsample0.8`. The implementer runs it on each example
  string.

## Docs to update

The three files above, each written for the present. This issue does not complete a roadmap item, so
there is no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pymarkdown scan .claude/skills/study/SKILL.md studies/quick_wins/README.md
uv run mkdocs build --strict    # and read the rendered auto-research page
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
```

**The Quick wins check runs on the maintainer's machine**, because the environment where this plan
was written has no `data/` folder. It does not call `score_study.py`, because the script logs the
attempt before any check, so a refusal known in advance would raise `study_submission_number` for
every later study on the fold:

1. Read the distinct `power_fcst_init_time` values of the Quick wins validation frames
   (`build_frames.frames_dir(fold_id=...)/val/init=*.parquet`) and of the reference experiment's
   rows for the fold.
2. Confirm that the Quick wins values are a 7-day-stride subset (about 53 of about 365).
3. Record the counts in the Quick wins README sentence.

## Risks and open questions

1. **Quick wins cannot post, and no existing study has been scored end to end.** The 7-day stride
   (`SAMPLE_STRIDE_DAYS = 7`) alone fails the row-key check, and quantile arms also fail the dtype
   check. The issue's wish to check `score_study.py` "end to end after #1129" therefore has no
   existing study to run on. **Decision for the maintainer:** accept an offline check now, with the
   first study built to forecast every daily run as the first real end-to-end test; or have the
   implementer submit a copy of the reference's own rows under a throwaway name, which exercises the
   whole path but adds one logged attempt and one `study/` experiment. I recommend the first.
2. **One experiment per reported arm.** The script stores one experiment per call. The alternative,
   one experiment per study holding only the headline arm, loses the arms' comparison on the
   leaderboard. I recommend one per reported arm.
3. **The agreed plan is the study's `plan-issue` plan**, committed before the run, so the commit
   orders the plan before the run. The alternative is the GitHub issue body, which a session
   launched overnight can also read.

## Findings considered and rejected

- **Downgrade the size to medium.** Rejected: the `plan-issue` rule says any firing trigger makes an
  issue complex. The maintainer may still downgrade it.
- **Drop the retry-once rule and the "stop between fits" rule.** Retry-once is dropped (accepted).
  "Stop between fits" stays, because it answers the issue's question about what the agent does when
  the budget runs out.
