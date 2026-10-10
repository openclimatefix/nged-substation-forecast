---
name: dataviz
description: >-
  This project's own charting rules, on top of the bundled `dataviz` skill's general method, for any
  chart drawn in this repository — a study page, a dashboard, a notebook, or anywhere else: the
  OCF-brand palette (`plotting.ocf_theme`), colour before shape, sizing a chart to a docs page's text
  column, hour-of-day axes, exporting and optimising SVG, and the extra care an NGED metered
  generator's time series needs so it cannot be re-identified. Load before drawing any chart in this repository, whether or
  not the chart is for a study. Load the bundled `dataviz` skill too: this skill supplements it and
  does not repeat its form heuristic, its six accessibility checks, its palette-validation script,
  or its interaction rules.
---

# Charting in this repository

Every chart drawn in this repository follows the rules below, on top of the bundled `dataviz`
skill's general method. Load the bundled skill too, for choosing a chart's form, the six
accessibility checks, `validate_palette.py`/`.js`, and its interaction rules. This skill gives this
project's own answers to the questions the bundled skill leaves brand-neutral. The `study` skill adds
a further layer of rules that apply only to a study page's charts, on top of both.

## Colours

**Take colours from `plotting.ocf_theme`, which encodes OCF's brand guidelines.** On a published
chart use only the guidelines' main data colours: Brand Orange `#FF4901`, Data Blue `#306BFF`, Data
Sky `#10C5F7`, Data Purple `#B701FF`, and Data Green `#17E58F`, with their light shades for a second
condition of the same series. The guidelines mark the additional data colours, such as the dark teal
`#009C75` and the amber `#FC9700`, for internal use only. `PALETTE`'s default order includes both of
those, so set every colour explicitly rather than relying on the default order.

**The maintainer can approve an additional data colour on a published chart when a chart's series
count genuinely needs one.** The ENS-horizons page's leaderboard colours nine series — three ENS
ways, four no-weather baselines, and two reference rows — each its own colour rather than sharing a
colour within a group, so it uses Data Amber and Data Magenta alongside the five main colours. The
two reference rows first shared Data Burnt Orange, which the maintainer later asked to swap for Data
Magenta because it clashed with Data Amber. Ask before reaching for an internal-use colour — a
maintainer request to use one, as here, is itself that ask — but don't treat this precedent as
blanket permission to reach for one unasked.

**Distinguish series by colour first; add a point shape or line style only as a backup, never as the
only thing distinguishing two series.** A chart that relies on shape or style alone loses its meaning
in grayscale or under colour blindness; a chart that adds shape on top of colour survives both.

**Validate every colour set with the bundled `dataviz` skill's `validate_palette.py`/`.js`.** One
colour per product can still fail the check: Data Purple and Data Blue sit 2.0 ΔE apart under
deuteranopia. Where a chart holds too many products for the check to pass one colour per product,
colour a group of products instead and put every product's name on the axis.

## Sizing and rendering

**Size a chart to the page's text column, so the chart shows at about 1:1 scale in the built site.**
Stack panels vertically rather than side by side, let the plot area fill the column's width, put
every key above the plot, and keep subtitles short. `studies.charts` draws every figure
`CONTENT_WIDTH_PX` (`packages/studies/src/studies/charts.py`) wide for this reason, the width of
`docs/`'s text column; match that width for a chart outside `docs/studies/` too, and check that the
text is legible in the built site, not only in the SVG.

**Write SVG, then optimise it, and check the render before committing.** `CLAUDE.md`'s "Chart
images" section has the full rule: run the export through `npx svgo@4 --multipass --precision=1
--final-newline`, and look at every chart rendered to PNG before committing it.

## Axes

**Label an hour-of-day axis at human divisions: 0, 6, 12, 18, and 24, or 6, 12, and 18.** A reader
finds 6, 12, and 18 o'clock at once, and cannot find 5, 10, and 15. In Altair, set
`axis=alt.Axis(values=[6, 12, 18])` (or `tickCount` with a step that divides 24) on the `x`
encoding, because Vega-Lite's automatic ticks pick steps of 5 or 10. The same holds for any other
cyclic unit with a familiar rhythm, such as months labelled at the quarter starts.

**Clip an hour-of-day axis to the hours that hold data.** A chart of solar output or irradiance has
no data at night, and the empty hours squeeze the plotted series into the middle of the panel. Round
the range of the plotted data out to human divisions, such as 6 to 18 for a few chosen days, or 4 to
20 for a chart that must hold midsummer. Do not leave the axis at 0 to 24. Give the domain
explicitly with `scale=alt.Scale(domain=[6, 18])`, and say in the subtitle that daylight hours only
are drawn.

## Self-contained charts

**Put the basics in the chart's own text, because many readers see the chart and nothing else.** A
reader landing on a chart must be able to answer every one of these questions from its title,
subtitle, axis titles, and legend alone:

- what is plotted, and in what unit;
- which direction is better, in words: "Mean absolute error (% of capacity; smaller is better)", or
  "Mean absolute error minus ERA5's (points of capacity; more negative is better)";
- what zero means, on a labelled reference rule ("same as ERA5");
- what a dot, a line, and a shade mean ("Dot: estimate. Line: 95% interval from resampling whole
  months");
- the scope: which generators, which region, and which period.

## Anonymising an NGED metered generator's time series

**The rules in this section cover NGED's private generators only.** A chart of public data, such as
the settled output of the Balancing Mechanism Units in Elexon's published data, may name each unit
and show megawatts on calendar dates.

**Put no calendar dates on the axis of an NGED metered generator's time series.** Even a handful of
generators can be matched against publicly available generation data once an hourly series sits on
known dates, which identifies the generator behind its anonymised label. Count the days of the week
on the axis (days 1 to 7), and give each plotted period's month and year in the surrounding text,
never the day. Turn off per-point accessibility text (`aria=False` on the mark), because Vega
otherwise writes each point's date and value into the SVG's ARIA labels. `CLAUDE.md` has the full
anonymisation rule this follows from, and the `study` skill's "Anonymisation" section has the rest of
the labelling workflow for a study.
