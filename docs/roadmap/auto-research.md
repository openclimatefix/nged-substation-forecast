# Experiments run by an LLM agent ("auto-research")

> **Status: 🚧 Planned.** The infrastructure — the trusted submit command, the review step, the
> leakage test, the research repository, the hypothesis store, and the leaderboard query — is
> planned for v0.3, alongside the leaderboard. The experiments themselves run in v0.5, alongside the
> rest of the [XGBoost improvements](xgboost-improvements.md). Gated on [Protect the leaderboard
> scorer for autonomous
> research](https://github.com/openclimatefix/nged-substation-forecast/issues/958). The
> infrastructure is tracked in [issue
> #1031](https://github.com/openclimatefix/nged-substation-forecast/issues/1031), and the
> experiments in [issue
> #1038](https://github.com/openclimatefix/nged-substation-forecast/issues/1038). Neither the
> infrastructure nor the search is built yet.

**We plan to have a large language model (LLM) agent run some or all of the
[XGBoost improvements](xgboost-improvements.md) backlog as a search: implement each idea, score it,
and combine the ideas that help.** Larger ideas from other roadmap pages, such as a new estimator of
generator capacity, may also go to the agent. Each idea becomes an experiment: a variant of
Flexpectation's forecasting pipeline, usually of the XGBoost forecasting model, scored the same way
as every other variant. The leaderboard is the table that ranks those experiments, and the champion
is the experiment currently promoted to production. The agent works in the style of
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch), running experiments and reading
the leaderboard with no human in the loop.

**Which of the XGBoost ideas go to the agent is not yet decided.** The options are:

- every idea, screened and combined by the agent;
- some of the ideas, with the rest tested by hand in the usual way;
- a broad-but-shallow screen of every idea by the agent, after which a person chooses the ideas for
  the deeper look.

The design below works for all three options. Under the third option, a person makes the research
lead's choice of which ideas to deepen.

**An agent session is judged on whether a finding moves the leaderboard, not on whether the finding
is publishable.** A result reaching production has to beat the champion on the honest scorer planned
in [issue #958](https://github.com/openclimatefix/nged-substation-forecast/issues/958). The honest
scorer runs from the reviewed `main` branch as the maintainer's Unix user. Issue #958 also plans for
agent sessions to run as a separate Unix user barred from the validation data the scorer holds back.

## Why energy forecasting suits automated research

**Energy forecasting has an advantage over the fields some research agents are built for: a genuine
check on results that the system being judged does not control.** Two of the research agents
reviewed [below](#agents-that-generate-rank-and-critique-ideas), the AI Scientist and Co-Scientist,
judge whether a result is good by a simulated review or a tournament run by the system's own agents.
The system being judged had a hand in constructing that check. A promoted forecasting model is
instead checked against power measured at the substation after the forecast was made, both in a
held-back historical window and in [live monitoring](live-service.md#production-monitoring). A
session that has read the held-back validation data could game the historical-window check but not
live monitoring. Issue #958 plans to stop a session reading the held-back validation data. The
[proposed design](#protecting-the-evaluation-code) sets out the part of the plan in issue #958 that
is still undecided.

**Data is comparatively plentiful and each experiment is cheap.** Once the training-history
extension ([#959](https://github.com/openclimatefix/nged-substation-forecast/issues/959)) lands,
each series will have several full years of half-hourly data. Each small experiment is an XGBoost
training run scored against a fixed fold. That run is cheap and fast compared with a wet-lab
experiment or a large pretraining run for a neural network. That combination is why an autonomous
research session is worth building here, even though the wider literature finds genuine recursive
self-improvement still blocked in most domains
([Duan et al., 2026](https://arxiv.org/abs/2609.11873), surveying the obstacles across scientific
discovery, embodied artificial intelligence (AI), and software engineering).

## What the experiment platform provides, and the leaderboard query it lacks

**The experiment platform was designed to support an agent from day one.** The pipeline runs on
Dagster, experiments are registered programmatically rather than through the Dagster user interface
(UI), MLflow records every run's metrics in a form a program can read, and a manual retirement job
prunes experiments nobody needs any more.

**The one piece the platform lacks for an agent to read results is a machine-readable leaderboard
query.** That query is a thin, typed Python surface answering "fetch the aggregate leaderboard
metrics for experiment X" and "rank every experiment by metric Y", so the agent reads results
without scraping a UI. The visual leaderboard
([#4](https://github.com/openclimatefix/nged-substation-forecast/issues/4)) needs the same query
underneath it. Writing the query as a reusable function, rather than burying the query in the chart
script, leaves the agent's surface at a few lines of code once the agent is built.

**MLflow's own Model Context Protocol (MCP) server, the standard interface through which an AI agent
calls a tool, does not serve this need.** The server's tools are generated by capturing the stdout
of a curated subset of MLflow command-line interface (CLI) commands, so a client receives rendered
text tables rather than structured data. And the two run-reading tools the server exposes
(`list_runs`, `describe_run`) accept neither a filter nor an `order_by`. Ranking N experiments
therefore takes N+1 round trips of text to parse — precisely the operation a leaderboard exists to
perform.

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

### Agents that generate, rank and critique ideas

**The AI Scientist's automated reviewer agreed with the average human reviewer more closely than
individual human reviewers agreed with each other.**
[Lu et al. (2024)](https://arxiv.org/abs/2408.06292)'s AI Scientist generates an idea, writes code,
runs the experiment, writes the result up as a paper, and then runs an automated peer review. The
review alone costs $0.25 to $0.50 in application programming interface (API) calls per paper. The AI
Scientist's automated reviewer reaches an F1 score of 0.57 against a human NeurIPS baseline of 0.49.
That automated reviewer's scores correlate more closely with the average human reviewer's score than
individual human reviewers' scores correlate with each other.

**Co-Scientist's hypotheses kept improving as its tournament between agents ran more rounds.**
[Gottweis et al. (2025)](https://arxiv.org/abs/2502.18864)'s Co-Scientist ranks candidate
hypotheses through an Elo-rated tournament between specialised agents (generation, reflection,
ranking, evolution, proximity, and meta-review). Across 203 research goals, hypothesis quality
(measured by Elo rating) kept rising through more tournament rounds rather than plateauing quickly.
That rise is evidence that spending more compute on ranking and revision continues to improve the
hypotheses.

**A few rounds of debate between separate instances of a language model beat both a single instance
and simple majority voting.** [Du et al. (2023)](https://arxiv.org/abs/2305.14325) show that three
agents debating over two rounds raised arithmetic accuracy from 67.0% to 81.8%, and
grade-school-math accuracy from 77.0% to 85.0%.

**For deciding whether a finding is real, the proposed design relies on the honest scorer, and
agents review each other's code only to catch broken implementations before the scorer runs.**
Adversarial review can be a large part of the answer to "is this finding real". But an autonomous
session here has a stronger check available than a simulated paper review: the leaderboard's honest
scorer.

**Debate between agents bears directly on how agents here could critique each other's work.**
Arithmetic and word-problem accuracy resemble the reasoning a session does when checking its own
feature-engineering logic or reading a metrics table, so the gains Du et al. measured are evidence
about agents here, not just an analogy. The proposed design adds that critique as a
[review before training](#the-research-lead-the-submit-command-and-the-review).

**Co-Scientist's tournament is one concrete answer to the breadth-versus-depth question, and the
proposed design answers the question differently.** In Co-Scientist, breadth comes from generating
many hypotheses up front, depth comes from repeated tournament rounds against the current top of the
ranking, and the balance between breadth and depth emerges from running more rounds. The
[proposed design](#an-llm-research-lead-decides-what-to-try-next) below sets the balance with an LLM
research lead that screens broadly and then goes deeper.

### Google's ERA: a tree search over code variants

**Google's Empirical Research Assistance (ERA) system searches a tree of code variants against a
fixed score, and two of its tasks are forecasting problems.** [Aygün et al.
(2025)](https://arxiv.org/abs/2509.06503) have an LLM rewrite code to improve a quality score, and
choose which candidate to extend next with an upper-confidence-bound rule applied across the whole
tree. Research ideas enter the prompt, either written by the user or summarised from papers. On 16
Kaggle Playground competitions, the tree search beat both a single LLM call and the best of 1,000
LLM calls, and also beat AIDE, an earlier agent for machine-learning engineering. Aygün et al.
report that the score typically stops improving after 300 to 1,000 nodes of the tree.

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
known ideas.** In the COVID-19 task, a separate retrospective three-week comparison of forecasting
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
comparisons; 95% intervals 7.7–46.2% and 20.5–66.7%).

**Keeping the best of several implementations is right for the forecast that ships, but wrong for
judging the idea.** Ning et al. draw this distinction: the maximum score finds a good artifact, but
the maximum can favour ideas whose implementations vary more. A search that uses one score to decide
which idea to build on next is crediting the idea. The score therefore has to cover several
implementations wherever the decision is close.

### Coding agents change experiments without saying so

**An agent that implements both the baseline and the new method can quietly change either one.**
[Si et al. (2024)](https://arxiv.org/abs/2409.04109) built an execution agent as a side experiment
to a human study of LLM-generated research ideas. On 2 sets of 30 ideas, about safety prompting and
about factuality prompting, the agent produced code that ran for 17 and 18 ideas respectively. But
Si et al. found the automated experiments could be misleading, because the agent often skipped or
modified steps in the baselines or the proposed methods, and in some cases defined the metric
functions incorrectly. One baseline the agent wrote was a five-keyword filter that any LLM-based
method would beat.

### The score that guides a search should not also confirm the result

**He et al. argue that a search steered by a score overfits to that score, so a separate, protected
evaluation has to certify the result.** [He et al. (2026)](https://arxiv.org/abs/2608.09855) argue,
in a position paper with a simulated physics demonstration, that automated research should be
organised like coverage-guided fuzz testing of software. A cheap signal of intermediate progress
chooses the next experiment. A validator the search cannot query adaptively decides whether a result
counts as a discovery. The [selection-bias
section](metrics-and-leaderboard.md#fold-hygiene-selection-bias-and-a-final-test-window) of the
leaderboard design names the risk of scoring hundreds of experiments on one fold. He et al.'s
separation addresses that risk.

## Proposed design

### Two modes of work

**The design has two modes: a search over small ideas, and one long agent session per large idea.**
A small idea is a Tier 1 or Tier 2 entry in the XGBoost backlog, such as a calendar feature or a
model setting, which a worker can implement in one session. A large idea is a research project
lasting days: a new estimator of the effective capacity of metered generators, a detector of
switching events, a full differentiable-physics forecaster, or a method for disaggregating unmetered
generation. A large idea usually adds a new upstream data product or a new forecaster, so a search
over many quick variants does not suit the large idea. Tier 3 and Tier 4 entries go to whichever
mode fits each entry. In the long-session mode, the worker trains and scores freely on the training
window during development. The review, the leakage test, and the honest scorer apply to the
implementation the session submits.

**Both modes share the same vocabulary.** A worker is a Claude Code session, Claude Code being
Anthropic's coding agent, that implements one idea. A node is one implementation of an idea: a
commit in the research repository, built on its parent node's commit. Both modes also share the
protections, the review, and the record described below.

### Workers change the pipeline; the output is fixed

**A worker may change any part of the pipeline except a short list of protected paths, and every
worker must end with the same output.** That output is a parquet file of `PowerForecast` rows for
the fold, scored through the study route planned in #958: `scripts/score_study.py`, run from the
`main` checkout as the maintainer's Unix user, scores the file the research session hands over. An
extension point narrower than the whole pipeline could not express the large ideas above. Each
implementation must also run end to end from the raw power, weather, and metadata tables, so that
the leakage test below can re-run the implementation on perturbed copies of those tables.

### An LLM research lead decides what to try next

**An LLM research lead prioritises the ideas, screens them broadly and shallowly, and then goes
deeper on the promising ones, in repeated rounds.** The research lead works in four steps, and steps
2 and 3 repeat as rounds:

1. Prioritise the ideas, starting from the order in which [XGBoost
   improvements](xgboost-improvements.md) lists the ideas.
2. Implement each chosen idea once, and score the implementation.
3. Read the scores and the hypothesis store, then choose which ideas or combinations to explore
   further, and which ideas get a second or third implementation.
4. Repeat from step 2 on the chosen subset.

The four steps are the maintainer's current best guess at how to sequence the search, not a settled
decision. ERA's upper-confidence-bound rule is one tool the research lead can use to choose which
node to extend. The same rule is also a baseline against which to measure the research lead's
choices.

### The research lead, the submit command, and the review

**The design splits into an agent loop, which reuses Claude Code, and a small trusted submit
command, which no agent can touch.** The agent loop is the research lead choosing ideas, launching
workers, and reading results. The agent loop is a standard harness, so the design does not build
one: the research lead is itself a Claude Code session, run interactively or through the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk), which packages Claude Code as a
Python library. For each node the research lead chooses, the research lead launches a worker in the
worker's own git worktree to implement one idea on top of the parent node's code.

**The trusted submit command runs as the maintainer's Unix user in a process no agent can reach, and
is the only process that trains and runs inference for a scored node.** The research lead submits a
node by calling one narrow command, such as `submit_node <commit>`, through a `sudo` rule, the same
pattern #958 plans for `scripts/score_study.py`. The submit command is plain Python in this
repository, tested with the rest of the code, and does every step the agents must not control:
launching the review, building the scored checkout, training the node and running inference, running
the leakage test, handing the predictions to the scorer, logging to MLflow, pushing node branches,
and opening the hypothesis-store pull request.

**No agent trains or predicts for a scored run, and the worker's code never runs as the maintainer's
user.** The submit command executes the worker's code as a subprocess running as the restricted
research user, because the worker's code is untrusted. A worker may still train and score freely on
the training window while developing an idea, but those runs never reach the leaderboard. Workers
never report their own scores. Every node is logged as an MLflow run under a `study/`-prefixed
experiment name, so the search is visible on the leaderboard but outside every promotion path. The
plan in issue #958 uses the same prefix for every autonomous study.

**The trusted checks do not live in a Claude Code mod.** A mod is a JavaScript or TypeScript plugin
that runs inside a Claude Code session's process, with the session's permissions. A mod holding the
trusted checks would sit inside the trust boundary the checks exist to guard, because the research
lead's session could edit the mod's source or tamper with the mod's state. A mod could still draw a
live pane of the search tree, with each node's score and review verdict, for a person watching an
interactive research-lead session. That pane is optional and can wait until the search runs.

**A fresh reviewer agent reads every implementation before the implementation trains a model.** The
submit command, not the research lead, launches the reviewer, so no agent can skip the review. The
reviewer is given the written idea and the worker's diff, but not the worker's reasoning, so the
worker's rationale cannot anchor the review. The reviewer checks three properties: that the diff
implements the idea the worker was given, against the
[silent changes Si et al. found](#coding-agents-change-experiments-without-saying-so); that no
feature uses data from after the forecast was made; and that the code has no plain bug. The
reviewer's findings go back to the worker for a capped number of rounds. An implementation the
reviewer still rejects after the last round is recorded in the hypothesis store as failed review,
and is not trained. A review does not guarantee a correct implementation. A review does make it less
likely that a good idea is discarded because one implementation was broken, which is one source of
the variance between implementations that
[Ning et al.](#one-implementation-is-weak-evidence-about-an-idea) measured.

### Protecting the evaluation code

**The submit command builds the scored run's checkout itself, so a worker's edits to its own copy of
a protected file change nothing in the scored run.** The submit command checks out the node's
commit, restores every protected path from the commit the session started from, removing any file
the node added under a protected path, and places the checkout in a directory the worker's Unix user
cannot write. The protected paths are:

- `conf/cv/`, which defines the cross-validation folds;
- `packages/contracts/`, which defines the data schemas;
- `metrics.py` under `packages/ml_core/src/ml_core/`, which computes the scores;
- `cv_helpers.py` under the same directory, which turns fold dates into training and validation
  windows and decides which series are eligible;
- `features/_lags.py` under the same directory, which builds the lag features and holds the
  function that nulls any power lag the forecast could not have known at its initialisation time.

The submit command also rejects, and logs, any diff against the same session base that touches a
protected path, so that an implementation is never scored on evaluation code its author did not
expect.

**The restored files guarantee only that the protected code is present, not that the worker's code
calls the protected code.** A worker may stop calling `cv_helpers.py` and write its own eligibility
filter, write a new lag module that nulls nothing, or point the workspace in `pyproject.toml` at a
modified copy of `ml_core`. The worker's code also runs in the same Python process as the protected
code, so the worker's code could replace the lag-nullification function in memory. Two checks cover
behaviour instead. The row-set refusal planned in #958 stops a worker dropping hard series from the
forecast. The leakage test stops lookahead.

**The submit command, not the worker's code, truncates the training data at the fold's `train_end`,
and the leakage test re-runs the pipeline on perturbed data.** For a sample of cut-off times, the
leakage test perturbs every power observation, and every weather-forecast run published, after the
cut-off. The test re-runs the worker's whole pipeline from the perturbed raw tables, including any
upstream data product the worker added, and rejects the node if any forecast initialised at or
before the cut-off changes. The test also checks that the pipeline's unperturbed output equals the
rows handed to the scorer, so the tested code and the scored code are the same code.

**A worker's prediction code needs power observed inside the validation window, and #958 plans to
bar the research sessions' Unix user from reading that window.** Forecasting a validation row needs
power observed before that row's initialisation time, and those observations lie inside the
validation window. Two options remain open. A harness running as the maintainer's user could stage,
for each initialisation time, only the power observed before that time, and run the worker's code as
the research user against the staged copy. Or the design could accept that a session sees in-window
power, and rely on a separate certifying window and on live monitoring to catch any gaming. The
choice between the two options belongs in issue #958.

**A guard inside the worker's Claude Code session adds no protection beyond the submit command's
checks.** A worker can bypass a deny rule in `.claude/settings.json` with a shell command, and can
edit the `.claude/settings.json` in its own worktree to remove a deny rule or a `PreToolUse` hook. A
Claude Code mod handling tool-call events would add JavaScript or TypeScript code that runs with the
user's permissions, without adding any protection the submit command does not already give. A deny
rule is still worth having to save a worker from wasting a run on an edit the submit command would
reject.

### Ranking and steering

**Ideas are ranked on the mean score across their implementations, and an idea is implemented a
second and third time only when its score is close to a competitor's or the idea is a finalist.**
Ranking on the mean is the policy [Ning et al.](#one-implementation-is-weak-evidence-about-an-idea)
propose for crediting an idea rather than one implementation of the idea. Ning et al. did not test
that policy inside a search like this one.

**The search steers on the leaderboard's headline score, normalised mean absolute error (NMAE)
([How each win is evaluated](xgboost-improvements.md#how-each-win-is-evaluated)), over forecast lead
times of 3 to 10 days.** The scorer does not yet report that band, so the search can steer on the
band only once the scorer does. Using the leaderboard's own score means the search and the
leaderboard cannot disagree about which experiment is best. It is open whether tail skill, scored by
[threshold-weighted continuous ranked probability score](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
(CRPS), should steer the search instead. Steering on the headline score is the goal-oriented
optimisation He et al. criticise. The protection this design relies on is therefore the separate
certifying evaluation, not the steering signal.

**Issue [#960](https://github.com/openclimatefix/nged-substation-forecast/issues/960)'s
recommendation already separates steering from certifying.** The recommendation is a "discovery
lane" that ranks ideas cheaply by cross-validation over whole-month blocks, and a rolling-origin
evaluation that confirms the winners. Issue #960 says both can be designed and built against the
current single fold, `mid_2025_to_mid_2026`, without waiting for more history. Until that design
lands, the search would steer on the same fold on which promotion is decided. The planned [Ladder
guard](metrics-and-leaderboard.md#fold-hygiene-selection-bias-and-a-final-test-window), which
publishes a new best only when the new best beats the standing best by a declared margin, would then
be the only protection against selection bias. The scale of search this page proposes is a reason to
build the design in issue #960 before the search runs.

### Recording what was learned

**Every implementation is kept as a branch in a separate research repository, which starts as a
private mirror of this repository.** Before each session, the submit command's setup step copies the
latest `main` into the research repository, so every session starts from the current reviewed code
and the champion's configuration. Each node is a branch created from its parent node's commit, so a
child node inherits every change its parent made. A worker pushes to the research repository only.
The node's MLflow run records the commit hash, so every score links to the exact code behind the
score. Old branches stay pinned to the commit they started from and are never updated after a
refactor, which is the rule `studies/` already follows.

**A separate repository keeps the workers' credentials away from this repository without any rules
per branch.** Workers, reviewers, and the research lead hold a token for the research repository
only. The submit command runs as the maintainer's user and is the only process holding a token for
this repository. A separate repository also keeps hundreds of node branches out of the branch list
people work from.

**A structured hypothesis store records what each idea taught the project, in a form a person can
read.** The store holds one Markdown file per idea under `studies/auto_research/` on `main`, with the
same fields in every file:

- the hypothesis;
- each implementation, with its commit in the research repository and its MLflow run;
- each implementation's score, and the spread between implementations;
- the reviewer's verdict on each implementation;
- the idea's status: worth deepening, or abandoned;
- the reason for abandoning the idea, where the idea was abandoned.

Within a session, the research lead updates the store on the session's branch in the research
repository. At the end of the session, the submit command opens one pull request carrying the
updated store into this repository, and the maintainer merges the pull request. A new session starts
only after the previous session's pull request has merged, so every session reads every earlier
finding. No worker, reviewer, or research-lead agent holds write access to `main`. The store carries
aggregate scores only, because a per-series score could identify a metered generator. A person
curates the findings worth publishing into `docs/`, the way studies are written up today.

**A winning idea reaches production as a reviewed re-implementation, never as merged agent code.**
The person writing the pull request adds the research repository as a git remote, reads the winning
node's diff against the commit the session started from, writes the idea up as a specification, and
re-implements the idea in a reviewed pull request, as #958 requires of every autonomous study.

## Open questions

**None of the papers this page reviews tests how well an LLM research lead chooses what to try
next.** ERA's upper-confidence-bound rule chooses which node to extend, and Co-Scientist's
tournament ranks hypotheses that already exist. He et al. propose an intermediate signal of progress
that chooses the next experiment, but test the proposal only in a simulated physics environment. An
idea raised in internal discussion, drawn from self-driving-lab practice in materials science rather
than from a paper this project has reviewed directly, is that this choice is the crucial component
of an autonomous research loop. That idea is worth checking against the self-driving-lab literature
before the design relies on the idea.

**The LLM may already know what happened during the evaluation period.** An LLM trained on text
written after the start of a validation window may know about events inside the window, such as a
heatwave or a change in electricity prices, and propose ideas that exploit that knowledge. The
`mid_2025_to_mid_2026` fold overlaps the training data of current LLMs. It is not yet settled
whether the overlap matters for ideas about feature engineering, and whether the certifying window
should postdate the LLM's training data.
