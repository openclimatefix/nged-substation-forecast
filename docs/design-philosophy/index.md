# Design Philosophy

This section is the **portable "why"** of the project: the part that would survive a rewrite of
every line of code, and the part another team could adopt without adopting any of our stack. It is
written to be readable without knowing Python or Polars — code names appear only as evidence that a
claim is practised, never as a prerequisite for following the argument.

Flexpectation is a greenfield project, and that is a rare opportunity: we get to research the best
practices of several industries — not only energy forecasting; some of the most useful ideas here
are borrowed from vehicle dynamics, avionics, manufacturing and site reliability engineering —
test-drive them against real data and a real production service, and report what we find. The
intended output is a field report, not a rulebook: a list of principles that any energy-forecasting
project might find useful *to consider*, together with honest results about which practices earned
their keep here, which we declined, and which failed. A practice that did not survive contact with
our data is as useful a finding as one that did.

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
