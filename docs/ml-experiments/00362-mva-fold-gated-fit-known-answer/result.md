---
id: "00362"
slug: mva-fold-gated-fit-known-answer
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  medium. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one gate would defensibly test the
  hypothesis (a constraint inside the fit, a start-selection rule, or a prediction-time clip); the
  start-selection rule is chosen because it needs no change to `packages/studies`. Callers: reuses
  experiment 358's and experiment 110's scripts, whose callers are all in those folders.
hypothesis: >-
  A gate that accepts reverse flow only where an independent solar forecast exceeds the fitted
  demand stops the fit through the meter from inventing reverse flow at a substation with none,
  without losing the reverse flow that real solar output produces.
prediction: >-
  At the natural level (no injection) at D5, the gated fit's sunny-half-hour direction accuracy is
  at least 0.95 (the signed fit's is 0.98; the ungated fit's was 0.35). At injection levels 1.0 and
  2.0 the gated fit stays within 0.02 of the signed fit's accuracy at 4 or more of 5 sites, and its
  fitted capacity stays within 15% of the signed fit's at 4 or more of 5 sites at level 1.0.
decision_rule: >-
  Passes (and step 2 runs) when, at the scored cut with reactive power zero, all three hold: D5's
  natural-level accuracy is within 0.02 of the signed fit's; at 4 of 5 sites at both levels the
  gated accuracy is within 0.02 of the signed fit's; and the capacity ratio to the signed fit is
  within 15% at 4 of 5 sites at level 1.0. Fails (and the sequence stops) otherwise. No
  cost-based threshold, because the experiment ports nothing.
benchmark: >-
  The ungated fit through the meter (experiment 358's L0, recomputed here), the signed fit, and the
  naive fit
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16; scored year 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30; frozen fits on past weather
metric: >-
  Share of sunny half-hours (true flow at least 5 percentage points of capacity from zero) with the
  flow direction right; ratio of fitted capacity to the signed fit's
---

# MVA fold: gated fit through the meter, known-answer test (planned)

Experiment 358 found that the fit through the meter invents reverse flow at D5, which has none.
This experiment adds a gate and repeats the known-answer test.

## Plan

- **Gate.** The fit tries six starting points and keeps the lowest loss. The gate rejects a start
  when more than 0.5% of the training half-hours have negative fitted power while the independent
  solar output is under 0.75 times the fitted demand. The independent solar output is the pinned
  capacity times the solar shape, plus (in this lab test only) the injected farm's output. Among the
  starts that pass, the lowest loss wins. If none passes, the start with the smallest rejected share
  is used and the fit is flagged. The same gate clips the forecast power to zero on test
  half-hours where it is closed. Both 0.75 and 0.5% are fixed here, before any gated fit runs.
- **Rest of the design** is experiment 358's: five sites (D5, D10, D11, D14, D16), injection levels
  none, 1.0, and 2.0, zero reactive power (plus power factor 0.97 reported), cuts 2024-07-01 and
  2025-07-01, past weather, point forecasts, arms signed (S), naive (N), ungated (L0), and gated
  (G).
- **Checks.** Ungated L0 must reproduce experiment 358's numbers (a replication check). The share
  of starts the gate rejects is reported per cell. Where all starts are rejected, the cell is flagged.
- **Lab shortcut.** The gate reads the injected farm's true output. A real MVA-only substation has no
  such input, and step 2 must supply an independent solar capacity there.
