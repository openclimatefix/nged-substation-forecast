---
name: update-dependencies
description: >-
  The routine for upgrading every locked dependency in openclimatefix/nged-substation-forecast with
  `uv lock --upgrade`: baseline-test on a new branch, upgrade, open a draft PR whose body lists
  every upgraded package, re-run every test that pre-commit and CI run, fix what broke, research
  whether any upgrade changed an API in a way that needs a wider code review or a skill update, put
  that report in the PR body and in the chat, then wait for approval and never merge. Load when the
  user asks to upgrade, refresh, or bump the dependencies or the lockfile.
---

# Update the locked dependencies

**The lockfile upgrade is one PR, and the maintainer merges it after reading the report.** The
routine ends at step 8 and waits. Never merge, and never mark the PR ready for review before the
maintainer has read the report.

1. **Start a new branch from `origin/main`**, named `upgrade-locked-dependencies-<YYYY-MM-DD>`.
   Run `uv sync --locked --all-packages` so the environment matches `main`'s lockfile and holds
   every workspace member, as CI's does. Without `--all-packages`, `ty` can pass locally and fail
   in CI.

2. **Run the full verification set on the untouched lockfile**, and do not fix any failure
   in this PR. A failure here is already on `main`. Record it, so that it is not blamed on the
   upgrade, and tell the maintainer about it in the report. The hooks in this set rewrite files
   when they find a problem, so discard those rewrites with `git checkout .`. The set is the union
   of `.github/workflows/ci.yml` and `.github/workflows/studies_tests.yml`, which are authoritative
   if this list has drifted, plus the pre-commit hooks:

    ```bash
    uv run ruff check .
    uv run ruff format --check .
    uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
    uv run ty check
    uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
    uv run mkdocs build --strict --site-dir <scratchpad>/site
    uv run python scripts/lint/check_docs_links.py
    uv run pytest --run-studies
    uv run pre-commit run --all-files --show-diff-on-failure
    ```

    `pytest --run-studies` takes about 3 minutes, and CI runs the studies tests in a separate
    job, so run both. Check the CPU load first, because one spinning core slows the XGBoost tests.

3. **Upgrade**, keeping the list of changes for the PR body:

    ```bash
    uv lock --upgrade 2>&1 | grep -E "^(Updated|Added|Removed)" > <scratchpad>/changes.txt
    uv sync --all-packages
    ```

    If nothing changed, stop and tell the maintainer.

4. **Commit `uv.lock` and open a draft PR.** Load `github-issue-pr-workflow` first, because it
   owns the PR checklist. Create the PR with `gh pr create --draft`. Write the body to a file in the session
   scratchpad, never in the repo, with the `github-issue-pr-workflow` rules: the attribution line
   first, one line per paragraph, no hard wraps. The body lists every upgraded package, one bullet per package
   as `` `name` old → new ``, generated from `changes.txt` so that no package is missing:

    ```bash
    sed -E 's/^Updated ([^ ]+) (v[^ ]+) -> (v[^ ]+)/- `\1` \2 → \3/' <scratchpad>/changes.txt
    ```

    Packages that `uv` added or removed need their own bullets.
 Then set the label and the `JackKelly` assignee.

5. **Re-run the whole verification set from step 2**, and fix every failure.
    - Fix the code for the new version rather than pinning the old version, unless the new version
      has a bug that the code cannot work around. Pin only with a comment saying what is broken and
      which version fixes it, in the style of the ANTLR pin in `pyproject.toml`.
    - A new `ty` diagnostic is not always a bug in the code. Load `ty-workarounds` before adding
      any `# ty: ignore`, and give each ignore a comment saying why.
    - A hook that rewrites a generated file (`studies/*/report.md`) means the generator must change
      as well, or the next run reverts the fix.
    - Commit each fix separately from the lockfile and push.

6. **Research what the upgrades changed**, with one Sonnet sub-agent per group of related packages.
   Give each sub-agent the packages, their old and new versions, and all of step 7, because the
   sub-agent reads the changelogs, greps this repository for the changed APIs, and answers the four
   questions for each package. Do not read the changelogs of all packages yourself. Spot-check
   one answer that says "no impact" by running its grep. Where the changelogs live:
    - **GitHub releases** are the quickest source: `gh release view <tag> --repo <owner>/<repo>`.
      The tag has a leading `v` for some projects and none for others (`ruff` uses `0.17.0`), so
      run `gh release list --repo <owner>/<repo> --limit 5` first.
    - **Intermediate versions count.** A jump of three releases needs all three sets of notes.
    - **`WebFetch` fails on some hosts** with a 403 or a redirect it will not follow. Follow the
      redirect, or fall back to `gh` or the project's `CHANGELOG`.
    - **Packages without a changelog** (some transitive dependencies) get no research, and the
      report says so.

7. **Judge each upgrade against the repository.** For each changelog entry, ask whether this
   repository uses the changed behaviour. Grep for the changed API in `src/`,
   `packages/`, and `studies/` before concluding either way, and answer four questions for every
   package with a breaking change, a deprecation, or a behaviour change:
    - **Does it need a wider code review?** For example, a Polars 2 upgrade changed row order after
      a lazy join without raising, and green tests did not prove the code was still right.
    - **Does it make a skill or a docs page stale?** Check `.claude/skills/`, `CLAUDE.md`, and
      `docs/architecture/` for the package name. `polars-patito-gotchas`, `ty-workarounds`,
      `marimo-notebooks`, and `dataviz` name specific versions and bugs. An upstream fix to the bug
      a `ty-workarounds` entry describes shows as `unused-ignore-comment` warnings, so look for
      those.
    - **Does it change a rule we configure?** `pyproject.toml` writes out the ruff `select` list,
      so ruff's changes to its default rule set do not apply, but a changed formatter does.
    - **Does it need a one-off operation before the code runs?** Search every changelog for
      "database upgrade", "schema", "migration", "migrate", and "backup". MLflow 3.17.0 needed
      `mlflow db upgrade <database-url>` with every writer stopped, and nothing failed in the test
      suite, because the tests use a fresh database. Name the command, the files it touches
      (`mlflow.db` in the main checkout, for MLflow), and whether the maintainer must run it. Never
      run a migration against the maintainer's data without being asked, and back the file up
      first.

    Write down the packages whose notes you read, the packages you did not read, and why.

8. **Write the report, put it in the PR body, tell the maintainer, and wait.**
    - The PR body keeps the package list and gains the report: the fixes made and why, and the
      answer to each question in step 7 for each package that needed one. Say plainly what was not
      checked. Edit the body from the same scratchpad file with `gh pr edit --body-file`.
    - Add a short list of every behaviour change that affects this repository, good or bad, each
      as one line naming the package and the code it touches, such as "the upgrade of `foo` should
      make our `baz` function roughly twice as fast" or "`foo` now returns rows in a different
      order, which `baz` relied on". Leave out a change that nothing in the repository uses.
    - Mark the PR ready for review only when the maintainer asks.
    - In the chat, give the same report as short bullets, with the PR linked in the form
      `[PR #N (title)](url)`.
    - Do not fix a skill or a docs page that the research found stale in this PR. Say which ones in
      the report, and make the change in a separate PR after the maintainer decides.

**Why a baseline run comes first:** the upgrade PR is the only place a regression from a new
version is cheap to find, and it can only be found if the failures that were already there are
known. **Why the research comes after the fixes:** an API change that raised is already fixed by
step 5, so step 6 can spend its time on the changes that raise nothing.
