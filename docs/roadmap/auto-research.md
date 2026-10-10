# Experiments run by an LLM agent ("auto-research")

> **Status: 🚧 Planned.** The infrastructure is planned for the v0.3 milestone and tracked in
> [issue #1031 (Tracking: infrastructure for safe experiments run by an LLM agent
> (auto-research))](https://github.com/openclimatefix/nged-substation-forecast/issues/1031): a
> separate Unix account for the agent, one shared command that scores every idea, a written record
> of every idea tried, and the instructions the agent follows. The first research sessions are
> tracked in [issue #1131 (Start running the AI researcher on the XGBoost forecasting
> model)](https://github.com/openclimatefix/nged-substation-forecast/issues/1131), and can start
> before the training-history extension in [issue #959 (Extend ECMWF ENS training
> history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) lands. Neither
> the infrastructure nor the search is built yet.

**We plan to have a large language model (LLM) agent screen the [XGBoost
improvements](xgboost-improvements.md) backlog, the list of ideas for improving this project's
XGBoost forecasting model, so that people implement properly only the promising ideas.**
Flexpectation is the forecasting system this project builds for National Grid Electricity
Distribution (NGED). Flexpectation's current forecaster is an XGBoost model, and XGBoost is a
standard machine-learning method. The agent implements each idea, scores each implementation, and
combines the ideas that help. The agent does not write production code. The agent works in the style
of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), running experiments and
reading its own results with no human in the loop.

**Every forecasting experiment here is trained on one stretch of past data and scored on a later
stretch, the validation window, that the training never saw.** A training stretch paired with its
validation window is called a fold. The leaderboard, the table that ranks every experiment, scores
every experiment on the same fold. The agent never sees the leaderboard fold's validation window.
The agent scores ideas on earlier folds. The maintainer then scores the best implementation of each
of the most promising ideas on the leaderboard fold, and re-implements the ideas that survive.

**Each attempt at an idea is kept as a git branch, a separate copy of the code where a change can be
tried without touching the reviewed code on `main`.** A change reaches `main` only through a pull
request, which a person reviews before merging.

**Every idea on the XGBoost improvements page is a candidate, and the maintainer leans towards
screening every idea.** Larger ideas from other roadmap pages, such as a new estimator of generator
capacity, may also go to the agent.

## Why energy forecasting suits automated research

**Energy forecasting has an advantage over the fields some research agents are built for: a genuine
check on results that the system being judged does not control.** Two of the research agents
reviewed [below](#agents-that-generate-rank-and-critique-ideas), the AI Scientist and Co-Scientist,
judge whether a result is good by a simulated review or a tournament run by the system's own agents.
A forecasting idea found by the agent is instead checked against power measured at the substation,
in a validation window the agent never saw, and, once promoted, in [live
monitoring](live-service.md#production-monitoring).

**Live monitoring is a check no forecast can game, because the power each forecast is scored against
has not been measured when the forecast is made.** The historical validation window is a weaker
check, because a person or an agent who has seen that window can steer towards ideas that suit that
window. The [proposed design](#proposed-design) keeps the agent away from the validation window for
that reason.

**Data is comparatively plentiful and each experiment is cheap.** Once the training-history
extension for the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble (ENS)
([issue #959 (Extend ECMWF ENS training
history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959)) lands, each series
will have several full years of half-hourly data. Each small experiment is an XGBoost training run
scored against a fixed fold. That run is cheap and fast compared with a wet-lab experiment or a
large pretraining run for a neural network. That combination is why an autonomous research session
is worth building here, even though a 2026 survey of the obstacles across scientific discovery,
embodied artificial intelligence (AI), and software engineering ([Duan et al.,
2026](https://arxiv.org/abs/2609.11873)) finds genuine recursive self-improvement still blocked in
most domains.

## Published evidence on research agents

**Treat the older results below with caution, because LLMs have improved very quickly and the LLMs
available in 2026 are far more capable than the LLMs these papers tested.** [Du et al.
(2023)](https://arxiv.org/abs/2305.14325) and [Lu et al. (2024)](https://arxiv.org/abs/2408.06292)
measured LLMs that are now several generations old. A gain from debate or from an automated reviewer
on a 2023 LLM may be smaller, or absent, on a 2026 LLM that already gets most of those answers
right. [Aygün et al. (2025)](https://arxiv.org/abs/2509.06503) saw the same effect in their own
results: on one task, GPT-5's single attempt was already good enough that the tree search made
little difference, and Aygün et al. expect more tasks to saturate as LLMs improve. The papers'
methods and failure modes carry over more reliably than their numbers do.

### Agents that generate, rank, and critique ideas

**The AI Scientist's automated reviewer agreed with the average human reviewer more closely than
individual human reviewers agreed with each other.** [Lu et al.
(2024)](https://arxiv.org/abs/2408.06292)'s AI Scientist generates an idea, writes code, runs the
experiment, writes the result up as a paper, and then runs an automated peer review. The review
alone costs $0.25 to $0.50 in application programming interface (API) calls per paper. F1 measures
agreement with accept-or-reject decisions, and higher is better. The AI Scientist's automated
reviewer reaches an F1 score of 0.57 against a human baseline of 0.49 from NeurIPS, a large
machine-learning conference. That automated reviewer's scores correlate more closely with the
average human reviewer's score than individual human reviewers' scores correlate with each other.

**In the proposed design, agent reviews look for broken and leaky implementations before the
implementations are scored. What certifies a finding is a reviewed re-implementation, scored on a
validation window the agent's search never sees.** Adversarial review can be a large part of the
answer to "is this finding real". But the design here has a stronger check available than a
simulated paper review: a person re-implements every finding, and the leaderboard scores the
re-implementation on data outside the search's reach.

**Co-Scientist's hypotheses kept improving as its tournament between agents ran more rounds.**
[Gottweis et al. (2025)](https://arxiv.org/abs/2502.18864)'s Co-Scientist ranks candidate hypotheses
through an Elo-rated tournament between specialised agents (generation, reflection, ranking,
evolution, proximity, and meta-review). Across 203 research goals, hypothesis quality (measured by
Elo rating) kept rising through more tournament rounds rather than plateauing quickly. That rise is
evidence that spending more compute on ranking and revision continues to improve the hypotheses.

**Any search has to balance breadth (trying many different ideas) against depth (pushing the best
few ideas further). Co-Scientist's tournament is one concrete way to strike that balance, and the
proposed design strikes the balance differently.** In Co-Scientist, breadth comes from generating
many hypotheses up front, depth comes from repeated tournament rounds against the current top of the
ranking, and the balance between breadth and depth emerges from running more rounds. The [proposed
design](#a-research-session) below sets the balance with an LLM research lead that screens broadly
and then goes deeper.

**A few rounds of debate between separate instances of a language model beat both a single instance
and simple majority voting.** [Du et al. (2023)](https://arxiv.org/abs/2305.14325) show that three
agents debating over two rounds raised arithmetic accuracy from 67.0% to 81.8%, and
grade-school-math accuracy from 77.0% to 85.0%.

**Debate between agents bears on how agents here could critique each other's work.** Arithmetic and
word-problem accuracy resemble the reasoning a session does when checking feature-engineering logic,
so Du et al.'s result suggests that a second agent's critique can catch errors here, subject to the
caution above about older LLMs. The proposed design adds that critique as a [review of every
substantial change](#reviewing-an-implementation-before-it-is-scored) before the screening harness
scores the change.

### Google's ERA: a tree search over code variants

**Google's Empirical Research Assistance (ERA) system searches a tree of code variants against a
fixed score, and two of its tasks are time-series forecasting benchmarks close to Flexpectation's.**
[Aygün et al. (2025)](https://arxiv.org/abs/2509.06503) have an LLM rewrite code to improve a
quality score, and choose which candidate to extend next with an upper-confidence-bound rule applied
across the whole tree. Each node of the tree is one version of the code. The upper-confidence-bound
rule balances extending the versions that score best against trying versions tested less often.
Research ideas enter the prompt, either written by the user or summarised from papers. On 16 Kaggle
Playground competitions, the tree search beat both a single LLM call and the best of 1,000 LLM
calls, and also beat AIDE, an earlier agent for machine-learning engineering. Aygün et al. report
that the score typically stops improving after 300 to 1,000 nodes of the tree.

**On the GIFT-Eval time-series benchmark, ERA's solutions converged on gradient boosting and beat
every entry on the 18 May 2025 leaderboard.** The benchmark spans 28 datasets across 7 domains. The
entries ERA beat included foundation models, deep-learning models, and standard time-series methods.

**ERA's COVID-19 hospitalisation forecasts beat the CovidHub Ensemble on a probabilistic score, in a
comparison that favoured ERA.** The forecasts were for weekly COVID-19 hospitalisations in the USA,
scored on weighted interval score (WIS). Aygün et al. ran the study retrospectively over the 2024/25
season, selecting a forecasting model each week on the preceding 6 weeks. ERA's retrospective
forecasting model averaged a WIS of 26 against the CovidHub Ensemble's 29. The comparison favours
ERA, because the retrospective study used the hospitalisation data available on 1 May 2025 for the
whole season, whereas the ensemble forecast in real time from the data available each week.

**ERA's gains often came from recombining known methods, which matches Flexpectation's backlog of
known ideas.** In the COVID-19 task, a separate retrospective 3-week comparison of forecasting
strategies found 14 strategies that beat the CovidHub Ensemble. Of those 14 strategies, 10 were
recombinations of two existing methods. In ERA's single-cell genomics task, Aygün et al. prompted
the search with each of the 55 pairs of 11 methods. Of the 55 recombinations, 24 beat both of their
parent methods.

### One implementation is weak evidence about an idea

**How a coding agent happens to implement an idea moves the score far more than re-running the
implementation does.** [Ning et al. (2026)](https://arxiv.org/abs/2607.26587) froze written
descriptions of candidate ideas and had coding agents implement each idea three times, in fresh
sessions, across 13 tabular tasks and 2 agent setups. Variance between implementations of the same
idea was more than 5 times the variance between re-runs of one implementation in one setup, and more
than 10 times in the other. The idea that won on a single implementation lost on the mean of the
other two implementations in 25.6% and 43.6% of comparisons in the two setups (10 and 17 of 39
comparisons, each comparison being one of the 13 tasks with one of its 3 implementations held out;
95% intervals 7.7–46.2% and 20.5–66.7%).

**Keeping the best of several implementations is right for the forecast that ships, but wrong for
judging the idea.** Ning et al. draw this distinction: the maximum score finds a good artifact, but
the maximum can favour ideas whose implementations vary more. A search that uses a single
implementation's score to decide which idea to build on next is crediting the idea. The score
therefore has to cover several implementations wherever the decision is close.

### Coding agents change experiments without saying so

**An agent that implements both the baseline and the new method can quietly change either one.** [Si
et al. (2024)](https://arxiv.org/abs/2409.04109) built an execution agent as a side experiment to a
human study of LLM-generated research ideas. On 2 sets of 30 ideas, about safety prompting and about
factuality prompting, the agent produced code that ran for 17 and 18 ideas respectively. But Si et
al. found the automated experiments could be misleading, because the agent often skipped or modified
steps in the baselines or the proposed methods, and in some cases defined the metric functions
incorrectly. One baseline the agent wrote was a five-keyword filter that any LLM-based method would
beat.

### The score that guides a search should not also confirm the result

**He et al. argue that a search steered by a score overfits to that score, so a separate, protected
evaluation has to certify the result.** [He et al. (2026)](https://arxiv.org/abs/2608.09855) argue,
in a position paper with a simulated physics demonstration, that automated research should be
organised like coverage-guided fuzz testing of software. A cheap signal of intermediate progress
guides the choice of the next experiment. A validator the search cannot query adaptively determines
whether a result counts as a discovery. The [selection-bias
section](metrics-and-leaderboard.md#fold-hygiene-selection-bias-and-a-final-test-window) of the
leaderboard design names the risk of scoring hundreds of experiments on one fold. He et al.'s
separation addresses that risk.

## Proposed design

### Six roles and objects

**The design rests on six roles and objects:**

- **The research lead** is an LLM session that decides what to try next, and which implementations
  need a review before they are scored.
- **A worker** is a separate LLM session that implements one idea once, on its own git branch.
- **A reviewer** is a fresh LLM session that reads an implementation's diff (the change the branch
  makes to the code), or a large idea's plan, against the written idea, without seeing the worker's
  reasoning.
- **The screening harness** is the one command every worker uses to score an implementation.
- **The hypothesis store** is the written record of every idea, every implementation, and every
  score. The store is a folder of Markdown files, kept on a long-running git branch, with an index
  file holding one row per idea.
- **The finalists** are the few top-ranked ideas the maintainer chooses after a session, to be
  reviewed adversarially and then scored on the leaderboard fold.

### The agent screens ideas; people decide what ships

**A finding from the agent reaches production only as a reviewed re-implementation, scored on the
leaderboard like any other experiment.** The person writing the pull request reads the winning
implementation's diff, writes the idea up, and re-implements the idea through the usual review. No
code the agent writes is merged into `main`. The re-implementation discards the agent's code. A
score that came from a bug or from an edited metric therefore ends in a wasted re-implementation
rather than a false result on the leaderboard. A leak that is part of the idea itself would survive
a faithful re-implementation. Two finalist reviews therefore read every finalist for lookahead, the
use of data that would not exist when the forecast is made, before the finalist is scored, as [From
a session's results to a shipped improvement](#from-a-sessions-results-to-a-shipped-improvement)
describes. The person re-implementing reads the diff again before porting the idea, and the
pipeline's own guard sets leaky power lags to null.

**The design guards against one risk above the others: selection bias on the leaderboard's
validation window.** The only leaderboard fold, `mid_2025_to_mid_2026`, trains on data up to
2025-06-30 and validates on 2025-07-01 to 2026-06-30. If the agent chose its shortlist by scoring
ideas on that validation window, the re-implementations' leaderboard scores would come out too high
even if the agent behaved honestly. The agent therefore never reads power observed after 2025-06-30.
Earlier folds steer the search, and the validation window, which the search never sees, certifies
the finalists. [He et al.](#the-score-that-guides-a-search-should-not-also-confirm-the-result)
recommend this split between folds that steer the search and a window that certifies the result. The
recommendation in [issue #960 (Design rolling-origin CV folds, assuming at least monthly
retraining)](https://github.com/openclimatefix/nged-substation-forecast/issues/960) makes the same
split for the leaderboard as a whole.

**File permissions stop the agent reading the validation window by accident, but do not try to stop
a determined agent from working around the permissions.** A determined agent's code still never
reaches the leaderboard unread: the re-implementation discards the agent's code, and the finalist
reviews and the person re-implementing each finalist read each finalist's diff first.

### Which Unix user runs what

**Everything the agent does runs as a separate Unix user, `researcher`. Everything that touches the
full power data or the leaderboard runs as the maintainer's own user.**

| Unix user | What runs as that user |
|---|---|
| The maintainer's user | The one-off setup; Dagster, which runs the data pipeline; the leaderboard; the maintainer's MLflow store, which records every experiment's settings and scores; the finalist reviews and the scoring of finalists; and the re-implementation of the winning ideas |
| `researcher` | Claude Code (the research lead, its workers, and its pre-screening reviewers), the screening harness, and every line of code the agent writes |

**Unix file permissions enforce the cutoff, because a convention would not hold.** Any script that
opens the cleaned power table directly would read past a cutoff that existed only in a helper
function or in the agent's instructions. The agent might not notice the leak. [Issue #1093 (Set up
the research Unix user and the truncated power copy for
auto-research)](https://github.com/openclimatefix/nged-substation-forecast/issues/1093) tracks the
setup. The one-off setup is:

1. Create the `researcher` user, with its own Claude login and no AWS credentials, including those
   for NGED's S3 bucket, so `researcher` cannot read the raw files NGED delivers.
2. Give `researcher` its own clone of this repository, which can pull from GitHub but holds no token
   to push.
3. Write two copies into a data folder `researcher` owns: the cleaned power table (NGED's
   half-hourly power, with a flag on each row the cleaning rules reject), truncated at 2025-07-01,
   and the `TimeSeriesMetadata` table, which holds each series' location and type but no power. The
   power copy needs rewriting when the leaderboard fold changes or a cleaning rule changes, and both
   copies need rewriting when a series joins the `TimeSeriesMetadata` table.
4. Give `researcher` read-only access to the weather data, which holds no power observations.
5. Deny `researcher` read access to the maintainer's MLflow store, and to everything else in the
   maintainer's data folder: the full power tables, the maintainer's forecasts and leaderboard
   metrics, the study outputs that hold derived power, and the `effective_capacity` table. The
   `effective_capacity` table holds each series' 99th percentile (P99) of absolute power over the
   full history, validation window included. The leaderboard divides each error by that P99.

**The maintainer's MLflow store is off limits because it holds every experiment's score on the
validation window.** Reading those scores would let the research lead steer towards ideas that did
well on the validation window. The research lead learns from the hypothesis store instead. The
`researcher` user has an MLflow store of its own for the screening runs.

### A research session

**A session is a Claude Code session run as `researcher`, following the research-lead skill, in
rounds.** [Issue #1036 (Write the research-lead skill for
auto-research)](https://github.com/openclimatefix/nged-substation-forecast/issues/1036) tracks the
skill. Each round has six steps:

1. The research lead merges the latest `main` into the hypothesis store's branch, then reads the
   store's index.
2. The research lead chooses the ideas for the round, starting from the order on the XGBoost
   improvements page.
3. For each implementation, the research lead starts a fresh worker on a new git branch, and gives
   the worker only the written idea, not the earlier branches.
4. The worker writes the code and commits the code.
5. For a substantial change, a reviewer reads the diff and the worker fixes the reviewer's findings,
   as [Reviewing an implementation before it is
   scored](#reviewing-an-implementation-before-it-is-scored) describes.
6. The worker scores the commit with the screening harness. The research lead then reads the new
   scores, writes up what the round showed, and chooses the next round: which ideas to deepen with a
   variant or an extension, which to combine, which to implement again, and which to abandon.

**The research-lead skill frames each round as finding out which ideas genuinely improve the
forecast, not as raising a score.** An abandoned idea, and why the idea was abandoned, is a finding
in its own right. The rounds above are the maintainer's current best guess at how to sequence the
search.

**ERA's upper-confidence-bound rule is one tool the research lead can use to choose which
implementation to extend, and a baseline against which to measure the research lead's choices.**

### The screening harness

**Every implementation is scored by one shared command, the screening harness, so that every idea is
scored the same way.** [Si et al.](#coding-agents-change-experiments-without-saying-so) found that
an agent left to run its own experiments quietly changed baselines and defined metric functions
wrongly. Either fault makes the scores of different ideas incomparable. [Issue #1130 (Build the
screening harness for
auto-research)](https://github.com/openclimatefix/nged-substation-forecast/issues/1130) tracks the
harness.

**The harness reuses the existing cross-validation pipeline, on screening folds that end before
2025-07-01.** A screening fold is marked `leaderboard: false` in `conf/cv/default.yaml`, like the
`smoke_test` fold. Reusing the pipeline means the harness scores ideas with the same code as the
leaderboard does. The pipeline also sets to null every power-lag feature (past power used as an
input) whose value had not yet been measured when the forecast was made.

**The harness refuses any implementation that could change how the implementation is scored, and
records the commit behind every score.** The harness refuses code that is not committed, and refuses
a branch whose diff against `main` touches `conf/cv/` or the metrics code. A worker therefore cannot
change the folds or the metric by accident. The harness records the score and the commit hash in the
hypothesis store, so every score points at the exact code that produced it.

**Screening scores are comparable with each other but not with leaderboard scores.** From the
truncated power copy, the harness builds `researcher`'s own `eligible_time_series` table, which
lists the series with enough history to be scored, and `researcher`'s own `effective_capacity`
table. Both tables depend on how much power history they see. A screening score can therefore cover
different series from a leaderboard score, and divides by different capacities.

**Until the ECMWF ENS history is extended, the screening folds can validate only on October 2024 to
June 2025.** The ENS archive starts on 2024-04-01, the pipeline's folds train before they validate,
and a series needs 6 months of history before the series is scored. Two short folds therefore fit
before the cutoff. The first fold trains on April to September 2024 and validates on October to
December 2024. The second fold trains on April to December 2024 and validates on January to June
2025. Screening can under-rate an idea whose benefit falls mostly in summer. Small effects will also
not stand out from noise. Once issue #959 adds about 3 more years of ENS history, screening folds
can cover every season.

**The first sessions can still run before issue #959 lands, because a large effect should stand out
even on the short folds.** The Tier 1 ideas are quick to try, because each Tier 1 idea is a
configuration change that takes hours.

### Reviewing an implementation before it is scored

**The research lead decides which implementations need a pre-screening review, and every substantial
change gets a pre-screening review.** A small diff, such as a new calendar feature or a changed
XGBoost setting, goes straight to the harness. A substantial change always gets a pre-screening
review: for example, a new module, a new upstream data product, or a change to how the pipeline
joins data or trains. The review exists so that a good idea is not abandoned because its one
implementation was broken. [Ning et al.](#one-implementation-is-weak-evidence-about-an-idea) found
that how an idea happens to be implemented moves the idea's score far more than re-running the
implementation does. A broken implementation is the extreme case of that effect.

**The pre-screening reviewer sees the written idea and the diff but not the worker's reasoning, so
the worker's rationale cannot anchor the review.** The reviewer checks that the diff implements the
idea, that no feature uses data from after the forecast was made, and that the code has no plain
bug. The reviewer's findings go back to the worker to fix before the harness scores the
implementation. The finalist reviews come later, as [From a session's results to a shipped
improvement](#from-a-sessions-results-to-a-shipped-improvement) describes.

### Small ideas and large ideas

**The design has two modes: a search over small ideas and one long agent session per large idea.** A
small idea is a [Tier 1](xgboost-improvements.md#tier-1-config-level-changes-hours-each) or [Tier
2](xgboost-improvements.md#tier-2-low-effort-feature-engineering-about-a-day-each) entry in the
XGBoost backlog, such as a calendar feature or an XGBoost setting, which a worker can implement in
one session. A large idea is a research project lasting days: a new estimator of the effective
capacity of metered generators, a detector of switching events, a full differentiable-physics
forecaster, or a method for disaggregating unmetered generation. A large idea usually adds a new
upstream data product or a new forecaster, so a search over many quick variants does not suit the
large idea. Tier 3 and Tier 4 entries go to whichever mode fits each entry.

**A large idea is built in small steps, each validated before the next, and the review of a large
idea's plan asks for the smallest test of the idea first.** Building a complex pipeline in one
session is risky, because a failure at the end says little about which step was wrong. Before a
worker builds a large idea, the research lead launches a fresh reviewer to read the worker's plan.
The reviewer asks the question John Jumper put to the AlphaFold team when someone proposed an idea
that would take two months to test: "What's the two day version of testing the same idea?" ([Jumper,
2024](https://www.nobelprize.org/prizes/chemistry/2024/jumper/1925168-interview-transcript/)). The
worker then builds the idea step by step, validates each step, and runs the smallest test that shows
whether the idea is worth continuing before building the rest.

### Several implementations of one idea

**Ideas are ranked on the mean score across their implementations. An idea is implemented a second
and third time only when the idea's score is close to another idea's score or the idea ranks near
the top of the index.** Ranking on the mean is the policy [Ning et
al.](#one-implementation-is-weak-evidence-about-an-idea) propose for crediting an idea rather than
one implementation of the idea. Ning et al. did not test that policy inside a search like this one.

**Each implementation of an idea comes from a fresh worker given the same written idea and not the
earlier branches**, so a later implementation is not a copy of an earlier implementation. Each
implementation is its own branch, for example `idea/holiday-flags/impl-1` and
`idea/holiday-flags/impl-2`. A combination of ideas is also a branch, built on top of the branches
it combines.

### Recording what was learned

**Every implementation is a git branch in `researcher`'s clone, and no branch is ever deleted.**
After each session, the maintainer fetches the branches into the maintainer's own clone with one
`git fetch`, so the code survives even if `researcher`'s clone is lost.

**The hypothesis store records what each idea taught the project, in a form a person and the
research lead can both read.** [Issue #1034 (Set up the auto-research hypothesis
store)](https://github.com/openclimatefix/nged-substation-forecast/issues/1034) tracks the store.
The store holds one Markdown file per idea under `studies/auto_research/`, with the same fields in
every file:

- the hypothesis;
- each implementation, with its branch and commit hash;
- each implementation's screening score;
- the idea's status: worth deepening, needs another implementation, combined, or abandoned;
- the reason for abandoning the idea, where the idea was abandoned;
- lessons learned, such as "needs feature X first" or "slow to train".

**The index file holds one row per idea, with the idea's status, its mean screening score, and the
spread of its screening scores across its implementations.** The research lead reads the index at
the start of every round instead of opening every idea file. The harness writes the scores, and the
research lead writes only the prose fields, so no score is copied by hand.

**The store lives on a long-running branch in `researcher`'s clone, because `researcher` holds no
token to push to GitHub.** The harness writes scores to a dedicated worktree of the store's branch,
so the workers' branches carry only code. The maintainer merges the store into `main` by pull
request after each session, so the next session starts from everything earlier sessions learned. The
pull request also gives the maintainer a readable summary of the session.

**The store carries aggregate scores only, because NGED's generator data may leave the project only
anonymised.** A per-series score could identify a metered generator, whose output can be
commercially sensitive.

### Which score steers the search

**The search steers on normalised mean absolute error (NMAE) over forecast lead times of 3 to 10
days, the headline score that the [XGBoost
improvements](xgboost-improvements.md#how-each-win-is-evaluated) page names.** The scorer does not
yet report that band. [Issue #1033 (Report NMAE over 3–10 day lead times as a leaderboard horizon
slice)](https://github.com/openclimatefix/nged-substation-forecast/issues/1033) adds the band.
Steering on the same metric as the leaderboard keeps the screening ranking and the leaderboard
ranking comparable. The two rankings can still disagree, because screening and the leaderboard
validate on different windows.

### From a session's results to a shipped improvement

**The maintainer turns a session's results into shipped improvements in five steps:**

1. Fetch the session's branches, and merge the hypothesis store into `main` by pull request.
2. Choose the finalists from the index.
3. For each finalist, have two fresh Opus agents review the diff of the idea's best-screening
   implementation adversarially, one after the other. Opus is the most capable of Anthropic's three
   tiers of Claude model, above Sonnet and Haiku.
4. Run each finalist that passes review through the cross-validation pipeline on the leaderboard
   fold, as the maintainer's own user. The agent never sees these scores.
5. Re-implement each idea that survives in a reviewed pull request, scored on the leaderboard as
   usual.

**Each finalist reviewer sees the written idea and the diff, never the worker's reasoning. The
second reviewer sees the diff with its comments stripped.** The reviewers hunt for lookahead, an
edited metric or fold, dropped rows, a refit on validation-window power, and an implementation that
does not match the idea. Stripping the comments means that a code comment arguing a feature is safe
cannot steer both reviews. A gain much larger than the gains of the other finalists is a reason for
more scrutiny.

**A finalist that won the screening but loses on the validation window is probably not worth
re-implementing.** The finalist's experiment stays out of production, because promotion is a
deliberate act the maintainer takes on a re-implementation only.

**Running a finalist's branch gives the agent's code the full power data, the same access the
maintainer gives any pull request run before review.** The two finalist reviews are the check on
that step.

**The finalist reviews go to the finalists because choosing the top-ranked implementations
concentrates bugs among the finalists.** Choosing the top few of many implementations tends to
choose the implementations whose bugs happened to raise the score, whether or not any agent meant to
cheat. A bug in an implementation that ranks low, and that the research lead judged too small to
review, wastes one screening run or loses one implementation of an idea. A bug in a finalist wastes
a look at the validation window and a re-implementation. A leak in the idea itself could survive
into the re-implementation. Giving the two Opus reviews only to the finalists keeps the number of
Opus reviews small.

**Comparing the screening ranking with the leaderboard ranking of the finalists tests the screening
itself.** If the two rankings disagree often, the screening harness or its folds need changing.

### Risks that remain

- **Scoring finalists on the validation window adds selection bias, a little at a time.** Each
  finalist scored there is one more look at the window. Keep the number of finalists per session
  small. The planned [Ladder
  guard](metrics-and-leaderboard.md#fold-hygiene-selection-bias-and-a-final-test-window) limits the
  effect on the leaderboard. The Ladder guard publishes a new best score only when the new best
  score beats the standing best score by a declared margin.
- **A missed file permission leaks power from the validation window.** The likeliest gap is a data
  folder created later, such as a study output that holds aggregated power. Creating a data folder
  includes checking that `researcher` cannot read the new folder.
- **No file permission hides what the public repository says.** Docs pages, study write-ups, and
  pull-request bodies can quote scores on the validation window, so the research-lead skill tells
  the agent not to read validation-window scores in docs pages, study write-ups, or pull-request
  bodies.
- **A good idea can be abandoned because its implementation was broken.** The research lead can
  misjudge a substantial change as small and skip its review. Implementing an idea a second time
  when the idea's score is close to another idea's score, as [Several implementations of one
  idea](#several-implementations-of-one-idea) describes, limits the damage.
- **A broken or leaky implementation can win the screening.** The finalist reviews and the
  re-implementation stand between that implementation and the leaderboard, so a broken
  implementation wastes effort rather than producing a false result. A leak that is part of the idea
  itself survives the re-implementation, and only reading the diff can catch that leak.
- **Two reviewers from the same family of LLMs can share a blind spot and miss the same subtle
  leak.** Three measures reduce that risk: stripping comments for the second reviewer, scrutinising
  unusually large gains, and the reviewed re-implementation.
- **The LLM may already know what happened during the validation window**, an open question below.

## Open questions

**None of the papers this page reviews tests how well an LLM research lead chooses what to try
next.** ERA's upper-confidence-bound rule selects which node of its tree to extend, and
Co-Scientist's tournament ranks hypotheses that already exist. He et al. propose an intermediate
signal of progress that guides the choice of the next experiment, but test the proposal only in a
simulated physics environment. Internal discussion has raised the idea that the choice of what to
try next is the crucial component of an autonomous research loop. The idea comes from
self-driving-lab practice in materials science, not from a paper this project has reviewed directly,
and is worth checking against the self-driving-lab literature before the design relies on the idea.

**It is open whether to screen every small idea or to let the research lead choose which small ideas
to try.** An exhaustive search would make the research lead's choice matter less. Most entries on
the [XGBoost improvements](xgboost-improvements.md) page are quick to implement and quick to run, so
the maintainer leans towards screening every entry. The research lead's choices would then matter
for the order of work, for which combinations to try, for which ideas get a second or third
implementation, and for the large ideas. A worker writing one implementation while the previous
implementation trains would shorten the search. That overlap is future work.

**Whether tail skill should steer the search instead of NMAE is open.** Tail skill is how well a
forecast predicts rare high-power periods, which matters for planning the distribution network. Tail
skill would be scored by [threshold-weighted continuous ranked probability
score](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
(CRPS).

**The LLM may already know what happened during the validation window.** An LLM trained on text
written after the start of a validation window may know about events inside the window, such as a
heatwave or a change in electricity prices, and propose ideas that exploit that knowledge. The
`mid_2025_to_mid_2026` fold overlaps the training data of current LLMs. It is not yet settled
whether the overlap matters for ideas about feature engineering, and whether the certifying window
should postdate the LLM's training data.
