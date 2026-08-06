# Design Philosophy

**The one-minute description:** We believe that five things can be true at once:

- **A service that mostly runs itself** — manual attention needed only when an upstream data format
  changes, with the forecast degrading gracefully rather than stopping when an input goes missing.
  The mechanism matters as much as the outcome: we plan to get there by training an ML model that
  can itself cope with missing inputs, rather than by wrapping fallback logic around a model that
  assumes complete data. A consequence of that, if it holds, is a service an operator can run day
  to day from the runbooks alone, without knowledge of the implementation details.
- **A hundred experiments per person in a peak month** — most research ideas fail, so the number of
  good ones a project finds is set by how many it can attempt.
- **Safe one-click promotion, and one-click rollback** — one *command*, not one leap of faith. By
  the time that command is available, the candidate has been scored against every other model on
  identical folds, has run on the very same code that will serve it, and can be reverted just as
  cheaply if it disappoints.
- **It runs for pocket money** — under £50/month at v1 scale, under £200/month at v2.
- **Scale without redesign** — 32 time series to ~2,500, with no structural change.

These are written down as [hypotheses with numbers and deadlines](engineering-hypotheses.md) rather
than as aims: none of them is settled yet, and a threshold we miss gets published as a negative
result rather than quietly revised.

The rest of this section is the **portable "why"** of the project: it would survive a rewrite of
every line of code, and it is what another team could adopt without adopting any of our stack. It is
written to be readable without knowing Python or Polars — code names appear only as evidence that a
claim is practised, never as a prerequisite for following the argument.

These pages therefore sit deliberately *above* the level of software-engineering practice.
Everything here is about the shape of the system rather than the shape of the code: which language
we write in, how functions are named, how the dataframe library is used, how the tests are wired —
none of that is settled here. Those finer-grained rules live in
[Code Style](../architecture/code-style.md) and [Testing](../architecture/testing.md), and a team
could disagree with every one of them while still adopting everything in this section.

Flexpectation is a greenfield project, and that is a rare opportunity to research the best practices
of several industries, test-drive them against real data and a real production service, and report
what we find. Those industries are not only energy forecasting: some of the most useful ideas here
are borrowed from vehicle dynamics, avionics, manufacturing and site reliability engineering. The
intended output is a field report, not a rulebook: a list of principles that any energy-forecasting
project might find useful *to consider*, together with honest results about which practices earned
their keep here, which we declined, which we have not yet absorbed — and, in time, which failed. A
practice that did not survive contact with our data is as useful a finding as one that did.

Three pages, in reading order:

- **[Design Principles](design-principles.md)** — the constraints we impose on our own decisions,
  each with the failure it prevents, a real decision it made, and the hypothesis it serves. Includes
  the practices we considered and deliberately declined, and the ones we know we have not yet
  absorbed.
- **[Engineering Hypotheses](engineering-hypotheses.md)** — the falsifiable claims the engineering
  is meant to deliver, each with a numeric threshold and the window in which it resolves. The
  principles are the bets; this page is the scoreboard.
- **[Inherent Stability](inherent-stability.md)** — the largest principle argued in full: how the
  service behaves as its inputs degrade, the degradation ladder, and the rules to follow when
  changing production code.

The boundary with the [Architecture](../architecture/overview.md) section is deliberate: this
section holds the transferable argument, while `architecture/` describes what we actually built,
with the *local* rationale — why this table layout, why this orchestrator — recorded next to each
component.
