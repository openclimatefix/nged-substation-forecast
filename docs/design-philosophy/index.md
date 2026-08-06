# Design Philosophy

This section is the **portable "why"** of the project: the part that would survive a rewrite of
every line of code, and the part another team could adopt without adopting any of our stack. It is
written to be readable without knowing Python or Polars — code names appear only as evidence that a
claim is practised, never as a prerequisite for following the argument.

An aim of this project is to gather **industry best practice into a single codebase for energy
forecasting** — and not only best practice from the energy-forecasting industry; some of the most
useful ideas here are borrowed from vehicle dynamics, avionics and site reliability engineering.
Articulating and refining this section is itself intended as a transferable output of the project,
alongside the forecasts.

Three pages, in reading order:

- **[Design Principles](design-principles.md)** — the constraints we impose on our own decisions,
  each with the failure it prevents, a real decision it made, and the hypothesis it serves. Includes
  the practices we considered and deliberately declined.
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
