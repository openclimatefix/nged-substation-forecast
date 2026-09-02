# Design Philosophy

**The one-minute description:** we are betting that five claims can all hold at once, each stated as
a falsifiable [engineering hypothesis](engineering-hypotheses.md) with a number and a deadline:

- [H1: a service that mostly runs itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
- [H2: a hundred experiments per person in a peak month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month)
- [H3: safe one-click promotion, and one-click rollback](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback)
- [H4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money)
- [H5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign)

None of them is settled yet, and a threshold we miss gets published as a negative result rather than
quietly revised.

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
are borrowed from vehicle dynamics, avionics, manufacturing, and site reliability engineering. The
intended output is a field report, not a rulebook: a list of principles that any energy-forecasting
project might find useful *to consider*, together with honest results about which practices were
worth their cost here, which we declined, which we have not yet absorbed — and, in time, which
failed. A practice that did not survive contact with our data is as useful a finding as one that
did.

Four pages, in reading order:

- **[Design Principles](design-principles.md)** — the constraints we impose on our own decisions,
  each with the failure it prevents, a real decision it made, and the hypothesis it serves. Includes
  the practices we considered and deliberately declined, and the practices we know we have not yet
  absorbed.
- **[Engineering Hypotheses](engineering-hypotheses.md)** — the falsifiable claims the engineering
  is meant to deliver, each with a numeric threshold and the window in which it resolves. The
  principles are the bets we make; this page scores them.
- **[Inherent Stability](inherent-stability.md)** — the largest principle argued in full: how the
  service behaves as its inputs degrade, the degradation ladder, and the rules to follow when
  changing production code.
- **[Common Incident Classes](common-incident-classes.md)** — the recurring failure shapes
  production forecasting services see in practice, and which mechanism above (if any) targets
  each one.

The boundary with the [Architecture](../architecture/overview.md) section is deliberate: this
section holds the transferable argument, while `architecture/` describes what we actually built,
with the *local* rationale — why this table layout, why this orchestrator — recorded next to each
component.
