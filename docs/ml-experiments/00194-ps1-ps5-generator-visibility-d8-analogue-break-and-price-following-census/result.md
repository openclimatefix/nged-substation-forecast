---
id: "00194"
slug: ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census
theme: switching-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (a descriptive
  reproduction of design-time numbers). Callers: this folder's scripts only, which read the stored
  residual panel, the power table, and the register spreadsheets.
hypothesis: >-
  D8's price-following drops (experiments 159, 182, 186, 191) are one metered gas engine, G2,
  whose output NGED adds back to D8's reading while G2's own reading arrives, and which shows in
  D8 as a negative residual once G2's reading stops arriving in late August 2023.
prediction: >-
  The share of D8 half-hours with a residual below -2.5 MW is at most 0.02 in July 2023 and at
  least 0.10 in every month from September 2023 to January 2026. The fitted change point of the
  weekly share lies between 2023-08-10 and 2023-09-10. No control series (D3, D5, D14) has a
  change point within 30 days of it. In PS5, no series other than D8 has a change point of at least
  0.10 in share.
decision_rule: >-
  Descriptive. PS1: if the prediction holds, the page states that D8's drops follow from a failed
  analogue on an added-back engine. If the change point lies outside the window, or a control shares
  it, the account is wrong. Stop if the stored panel and G2's readings cannot reproduce the 2,054
  joined half-hours to within 5%, and report the difference. PS5: descriptive.
benchmark: Design numbers in the brief (price_sensitive_generation_design.md, section 1).
row_set_id: none (describes stored residuals and power, no forecast is scored)
metric: >-
  Share of half-hours with a residual below -2.5 MW per month, quarter and week; change-point date
  with an interval; G2 running share by price fifth.
code: ml-experiments/00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/
---

# Experiment 194: does D8's drop pattern follow one engine's reading stopping?

*Planned; the write-up replaces this page when the scripts have run.*

Prediction, decision rule and stop rule are in the front matter. Controls: D3, D5, D14. PS5 counts
series whose `information` field says an added-back generator or a non-working reading, and
Primaries with gas or oil engines, batteries or demand-side response in the register, by D-label
only.
