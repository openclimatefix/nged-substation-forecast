# Planning-docs restructure — remaining GitHub pass

The docs side of the restructure is complete (PRs #218, #219, and the plans-absorption PR):
`docs/techniques/` exists, `docs/roadmap/index.md` is the planning front door, all plan content
lives in roadmap pages, and CLAUDE.md carries the conventions. What remains is making GitHub
match — fill the gaps found in the 2026-07-02 audit of epic
[#132](https://github.com/openclimatefix/nged-substation-forecast/issues/132)'s sub-issue tree.

## Create missing sub-issues

Batch-draft for Jack's review before creation; each body ~2 lines + a link to its roadmap
section:

- Live inference asset (`live_forecasts`) — under the v0.1 epic
  [#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137)
  ([#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208) is only its
  verification step) → `docs/roadmap/live-service.md`
- Champion-model container — under #137 → `docs/roadmap/live-service.md`
- Production monitoring — under the v0.3 epic
  [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6) →
  `docs/roadmap/live-service.md`
- Probabilistic evaluation (horizon slices, PICP, spread-skill, CRPS, calibration) — under #6
  → `docs/roadmap/metrics-and-leaderboard.md`
- Leaderboard fold hygiene (final-test window) — under #6 →
  `docs/roadmap/metrics-and-leaderboard.md`
- Reproducibility stamping — under the v0.2 epic
  [#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138) →
  `docs/roadmap/engineering-health.md`
- Drop Hydra — under #138 → `docs/roadmap/engineering-health.md`
- Scientific-rigor tests — under #138 (relates to
  [#62](https://github.com/openclimatefix/nged-substation-forecast/issues/62)) →
  `docs/roadmap/engineering-health.md`
- XGBoost Tier-1 quick wins (lead-time feature, ordinal time, early stopping, holidays) —
  under the v0.5 epic
  [#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145) →
  `docs/roadmap/xgboost-improvements.md`

After creation, fill the empty GitHub cells in `docs/roadmap/index.md`'s map table.

## Mirror dependencies as `blocked by` links

Live asset → container → AWS deployment; monitoring blocked by live forecasts accumulating;
XGBoost item 15 blocked by item 14 (batched training); per-horizon models (#149) blocked by
horizon-sliced metrics. Confirm `gh`/GraphQL support for the issue-dependencies API at
execution time; fall back to a "Blocked by #n" body-text convention if the API isn't
available yet.

## Fix drift

- Close or rewrite [#207](https://github.com/openclimatefix/nged-substation-forecast/issues/207)
  (references the deleted `dagster_plan.md`; its remaining phases are now the live-service and
  engineering-health roadmap pages).
- Re-parent [#144](https://github.com/openclimatefix/nged-substation-forecast/issues/144)
  (data cleaning) from #136 to the v0.4 epic
  [#150](https://github.com/openclimatefix/nged-substation-forecast/issues/150).
- Slim epic bodies to ~2 lines + a link to their roadmap page/section; check roadmap pages
  link back.

## Triage calls for Jack

- Are [#96](https://github.com/openclimatefix/nged-substation-forecast/issues/96)
  (NGED-agreed schema) and
  [#5](https://github.com/openclimatefix/nged-substation-forecast/issues/5) (backups)
  v0.1-gating (→ map-table rows)?
- Does [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161)'s
  placement under the v0.1 epic reflect real intent? (The map table currently says v0.1,
  matching GitHub.)
- Do the #136 strays
  ([#179](https://github.com/openclimatefix/nged-substation-forecast/issues/179),
  [#197](https://github.com/openclimatefix/nged-substation-forecast/issues/197),
  [#153](https://github.com/openclimatefix/nged-substation-forecast/issues/153)) stay
  GitHub-only?

## Verification

Every 🚧 map-table row in `docs/roadmap/index.md` has an open issue; #207 is closed or
rewritten; #144 sits under #150. Delete this file when done.
