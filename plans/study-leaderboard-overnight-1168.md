# Plan: the `study` skill says when a study posts to the leaderboard, and where the boundary with ml-experiments lies (#1168)

**The problem.** The `study` skill says a study's leaderboard number comes only from
`scripts/forecasting/score_study.py`, but never says when a study should post one, and no study has.
The skill also has no sentence that tells a session when to use a study and when to use something
smaller. The maintainer has decided that "agentic ML research" is two kinds of work: **studies**,
which are long reports that answer several scientific questions, and **ml-experiments**, which are
many small, focused experiments whose aim is to improve the Flexpectation production forecast. The
overnight procedure that issue #1168 first asked for belongs to ml-experiments, so it moves to the
`ml-experiment` skill (a separate issue).

**The solution.** Three small edits to the `study` skill and its neighbours, and no overnight
procedure. The `study` skill gets one optional posting rule, a boundary sentence that names
ml-experiments, and a requirement that a page which posts says so. The Quick wins page and README
get the reason they post nothing.

## Base

PR #1171 has merged, and this branch contains it. `score_study.py` has no environment re-execution
and no `NGED_ALLOW_REPLACE` gate, so `--replace` is a plain flag.

## Verdict, size, and departures

- **Verdict: worth implementing, in a smaller form than the issue asks.** The issue's overnight
  procedure goes to the new `ml-experiment` skill, because overnight runs are mostly ml-experiments
  and not studies.
- **Size: medium.**
    - What gets stored: does not fire. The change is text; a study that later follows the rule uses
      an existing script.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: does not fire for this scope. The design choice (where the
      overnight procedure lives) moved to the `ml-experiment` issue.
    - Callers not nameable without searching: does not fire.
- **Reviews: none further.** The earlier, larger version of this plan had a Sonnet simplicity review
  and an Opus correctness review. Their findings that still apply are kept below, and this version
  only deletes scope.
- **Departures from the issue body:** the five-part overnight procedure is removed from this issue,
  and the posting step is optional, not required. The issue body and title are edited to match.

## Solution

### Walk through

#### The happy path

A session running a study reaches the order-of-work step "post the arms the page reports" only if
the study's forecasts can pass `score_study.py`. The study's fit script writes one parquet file per
arm holding exactly the `PowerForecast` columns the script accepts (`valid_time`, `time_series_id`,
`ensemble_member` as Int8, `power_fcst_init_time`, `power_fcst` as Float32, `power_fcst_model_name`,
`power_fcst_model_version`, and nothing else), with one row per series, init time, lead time, and
ensemble member of the reference experiment `xgboost_no_power_lags`. The session runs
`uv run python scripts/forecasting/score_study.py <file> <page-slug>--<arm> mid_2025_to_mid_2026`
once per arm, because the script stores one experiment per call. The page lists the `study/`
experiment names and says that `study/` rows on the leaderboard fold are biased low, because the
study chose its arms on that fold.

#### The main branches off the happy path

- **A study that cannot match the reference's row keys** posts nothing and reports its own scores on
  the page. Quick wins is in this case: it forecasts one run every 7 days, about 53 runs, and the
  leaderboard scores every daily run.
- **A study whose output does not fit `PowerForecast`** (quantile samples with 459 ensemble codes,
  19-level quantiles, reordered members) posts nothing.
- **A study fitted past the fold's `train_end`** matches the row keys and is still in sample, so it
  refits or posts nothing.
- **A study that wants to run a small, focused experiment** is directed to the `ml-experiment` skill
  by the boundary sentence.

#### The main error-handling paths

- **`score_study.py` refuses a file.** The pre-write checks raise before anything is written, and
  the script logs the refusal. The session fixes the file, or posts nothing and says why on the page.
- **The script fails after the write.** `_score_study` writes the partition before it materialises
  `metrics`, so a failure leaves the rows stored. The retry uses `--replace`. Any new attempt, a new
  name included, raises `study_submission_number`.

## What changes, file by file

- **`.claude/skills/study/SKILL.md`.**
    - Extend the paragraph "A study's leaderboard number comes only from `score_study.py`" with
      "post when the output fits": which arms, the `<page-slug>--<arm>` name (lower-case the arm,
      map `.` to `_`, at most 64 characters in all; the page's slug leaves 35 for the arm), seed 0
      posted and stated on the page, exactly the `PowerForecast` columns, and the cases above that
      post nothing. Say "control arm" for the study's own baseline, because the scorer's "reference"
      is `xgboost_no_power_lags`.
    - Add a step to "The order of work" after step 5: post the arms the page reports, once a
      reviewer finds nothing more to fix and before step 6 finishes the page.
    - Add a bullet to "Writing the page": list the `study/` experiment names, and state the bias
      warning once.
    - Add one boundary sentence under the opening paragraph: a study answers several questions in a
      long report; a small, focused experiment aimed at improving the production forecast uses the
      `ml-experiment` skill instead.
- **`studies/quick_wins/README.md` and `docs/studies/forecasts/quick-wins.md` (lines 208 and 639).**
  Add the reason the study posts nothing: its 7-day stride, and its quantile arms' ensemble codes.
  The wording "did not submit to the scorer" stays true.

Not changed: `CLAUDE.md`, `studies/README.md`, any script, `docs/roadmap/auto-research.md` (its
rewrite belongs to the `ml-experiment` issue, which also removes the stale lines 207-210 text).

## Design-philosophy check

Everything here is R&D and fails fast. No production code, asset, or asset check changes, and no
hypothesis or principle is touched.

## Tests

No code changes, so no new test. The implementer runs `validate_study_name` on every example name in
the skill, including one built from an arm name with a dot such as `colsample0.8`. The step that says
Quick wins posts nothing is checked offline, on the maintainer's machine, by counting the distinct
`power_fcst_init_time` values of the Quick wins validation frames against the reference's, without
calling `score_study.py`: a refusal known in advance would still log an attempt and raise
`study_submission_number` for every later study.

## Docs to update

The three files above, each written for the present. No ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
```

## Risks and open questions

1. **No existing study has been scored end to end.** Quick wins cannot post, so the issue's wish to
   check `score_study.py` end to end has no study to run on. I recommend an offline check now and
   the first ml-experiment or study built to forecast every daily run as the first real test.
2. **The posting rule may be rarely used.** Phase 1 experiments are free to use their own metrics,
   so few outputs will fit the scorer. Keeping the rule short is the point of making it optional.

## Findings considered and rejected

- **Downgrade the earlier size to medium** (earlier simplicity review): moot, the plan is now medium.
- **Keep the overnight procedure here, in about 25 lines** (earlier simplicity review): rejected in
  favour of the `ml-experiment` skill, on the maintainer's decision.
