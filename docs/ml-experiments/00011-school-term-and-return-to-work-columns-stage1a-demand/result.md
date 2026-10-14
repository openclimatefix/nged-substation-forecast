---
id: "00011"
slug: school-term-and-return-to-work-columns-stage1a-demand
status: planned
size: complex
hypothesis: >-
  For the ERA5-trained stage 1a XGBoost demand model at Primary substations in Lincolnshire,
  columns for the return to work after New Year and for Lincolnshire's school terms lower the mean
  absolute error of the median in the weeks after Christmas, at Easter, and in the half-term
  holidays, against the same model with only the 14-day-window days-to-Christmas column.
prediction: >-
  Against the base arm, the arm with both sets of columns lowers the Primary error by at least 0.5
  points of capacity from 2 to 20 January, and by at least 0.3 points over the Easter window and
  over the three half-term holidays. Its whole-year error moves by less than 0.1 points. Its gain
  is larger than that of the same columns computed from a calendar shifted by 17 days.
decision_rule: >-
  A window contrast counts as a gain when its 99.2% interval (95% with a Bonferroni correction
  across the six planned contrasts, resampling whole local days) excludes zero, its point estimate
  has the same sign in all three folds, and its size meets the prediction. Supported if C1, C2, and
  C3 are gains and C5 and C6 hold. Partly supported if at least one of C1 to C3 is a gain and C6
  holds for the pooled windows. Refuted if no interval of C1 to C3 excludes zero below zero.
  Inconclusive otherwise.
benchmark: >-
  The base arm (weather block W, hour and weekday, days to Christmas inside a 14-day window,
  trained from 2021-09-01), and a month-by-half-hour climatology row.
row_set_id: era5_val_3folds_2023_07_to_2026_06_v1
metric: mean absolute error of the median, points of capacity (power over effective capacity, times 100)
---

# Do school-term and return-to-work columns lower the stage 1a demand error after Christmas, at Easter, and in school holidays?

**Status: planned.** This page is the plan. It fixes the arms, the windows, and the contrasts before
any fit. It continues experiment 00006, which found that a windowed days-to-Christmas column lowers
the Primary error by 1.8 points from 15 December to 7 January and by almost nothing from 5 to 8
January, and which left Easter and the school holidays without a column.

## Size

**The size is complex, because the experiment adds data and admits more than one defensible
design.**

- **What gets stored:** one new committed file of hand-collected term dates, and private predictions.
  No schema in `packages/contracts/` changes.
- **Production serving path:** not touched. The columns are candidates for a later model release.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The window lengths, the encoding of a term boundary
  (a flag, signed day counts, or both), and the control all admit alternatives. This page fixes one.
- **Code whose callers cannot be named without searching:** no. The fitting code is a copy of
  experiment 00006's, with new column groups.

## Which earlier advice this plan follows

- **Followed:** window the day counts and leave them missing elsewhere (00006, Christmas control);
  score a new calendar column on special-day windows as well as the whole year; add a control before
  crediting a signed or cyclic calendar column; split every score by substation type, and make
  Primary the headline.
- **Departed from:** 00006's recommended set includes the COVID indicator and the full history. This
  experiment trains every arm from 2021-09-01, because school-term dates before the 2021-22
  academic year are not collected. The indicator is then constant at zero and is dropped. One
  reference arm keeps the full history, to show what the shorter history costs.

## Data: Lincolnshire County Council's term dates

**All 20 series lie in Lincolnshire, so one calendar covers them.** The metadata places every Primary,
BSP, and GSP series between latitude 52.7 and 53.3 and longitude -0.2 and 0.33, in the East Midlands
licence area. The planned split "by nation and by local authority" therefore has one nation (England)
and one local authority. The calendar is Lincolnshire County Council's recommended calendar, which
binds community and voluntary-controlled schools. Academies and voluntary-aided schools set their own
dates, often within a day or two of it. The columns are therefore a proxy for the calendar of the
schools near a substation, not a count of pupils in school.

**The dates are in the CSV file `lincolnshire_term_dates.csv` in the experiment's code folder, with
sources and licence in the README beside it.** The first and last day of 30 terms (2021-22 to
2025-26) were read by hand from the council's web pages and two school calendars. The 2024-25 year,
which is the validation year of the second fold, rests on one school's copy of the council's
calendar, confirmed against the amendment the council's page states. `e00011_term_dates.py`
validates the file. The checks cover six terms per year, order and overlap, a weekday start, term
length, holiday length, Christmas and Easter inside a holiday, and the 2025-26 dates against two
other calendars. All 30 terms pass. The file is a small hand-collected table, so the checks for
gridded products in the `data-validation` skill (orientation, running means, stuck runs) do not
apply.

## Columns

**Every column is computed from the local date alone and is known months ahead.** The council
publishes the current and the next school year, so each column exists at forecast time. The
experiment uses no lagged weather or power.

| Column | Definition |
| --- | --- |
| `school_term_flag` | 1 from a term's first to last day (weekends inside a term included), else 0 |
| `days_from_term_start_w` | Signed days to the nearest term's first day, kept within 13 days either way, else missing |
| `days_from_term_end_w` | The same for a term's last day |
| `days_from_return_w` | Signed days from the first working day of the year (the first weekday that is not a bank holiday in England), kept from 7 days before to 14 days after, else missing |
| `placebo_*` | The same four columns read from the real calendar 17 days earlier |

## Arms

**All arms train from 2021-09-01 except the reference arm, use three fitting seeds, and use
`reg:absoluteerror` with the settings of experiment 00006.** Folds are the three validation years of
experiment 00006 (2023-07 to 2024-06, 2024-07 to 2025-06, and 2025-07 to 2026-06), each trained to a
week before it starts.

| Arm | Columns |
| --- | --- |
| Base | W, hour, weekday, `days_to_christmas_window` |
| Return | Base plus `days_from_return_w` |
| Term | Base plus `school_term_flag`, `days_from_term_start_w`, `days_from_term_end_w` |
| Both | Base plus the Return and Term columns |
| Placebo | Base plus the four placebo columns |
| One harmonic | Base plus one harmonic of the day of the year (a smooth-season control) |
| Reference | Base, trained from the full history with the COVID indicator, as in 00006 |
| Climatology | Month, half-hour, and weekend flag, trained on each fold's window |

**The placebo arm controls for extra columns and for calendar memorisation.** It has the same
number and shape of columns as Both. If its gain matches Both's, the gain comes from the tree having
more calendar columns to split on and not from the real dates. The one-harmonic arm tests whether a
smooth seasonal curve already explains a gain.

## Windows

**Windows are fixed by local date, and each has three occurrences, one per fold.**

- **Return window:** 2 to 20 January.
- **Easter window:** from the first day of the Easter holiday (the day after term 4 ends) to the sixth
  day of term 5.
- **Half-term holidays:** the autumn, February, and May half-term holidays, from the day after a term
  ends to the day before the next begins (nine holidays across the three folds).
- **Summer holiday:** from the day after term 6 ends to 2 September.
- **Whole year:** every day of the validation year.

## Planned contrasts

**There are six planned contrasts, all for Primary substations, against the Base arm unless stated.**
The level is 99.2% (a Bonferroni correction of 95% across six). Each window interval resamples
whole local days, so it describes these three years only, as in experiment 00006. A window contrast
also needs the same sign in all three folds. C5 uses the month-resampling bootstrap of
`studies.bootstrap`.

| Contrast | Arms | Where | Prediction |
| --- | --- | --- | --- |
| C1 | Both against Base | Return window | At most -0.5 |
| C2 | Both against Base | Easter window | At most -0.3 |
| C3 | Both against Base | Half-term holidays | At most -0.3 |
| C4 | Both against Base | Summer holiday | No prediction; reported as a check for harm |
| C5 | Both against Base | Whole year | Interval upper bound below +0.1 |
| C6 | Both against Placebo | C1, C2, and C3 windows pooled | At most -0.2 |

**The report also gives, without a decision attached:** Return and Term separately; Both against
One harmonic and against Reference; the Christmas weeks (15 December to 7 January), where Both must
not be worse than Base by more than 0.1; the BSP and GSP series (two each); and absolute scores for
every arm, with the climatology row repeated once per seed.

## Resources and reporting

**The fits use the 10 cores of slot B on the CPU.** `e00011_fit.py` copies the structure of
`e00006_fit.py`. Reading experiment 00006's private feature table is allowed, and the new columns
join on the local date at fit time. The experiment scores no valid time on or after 2026-07-01.
Predictions are written with `experiment_store.write_predictions` under row set
`era5_val_3folds_2023_07_to_2026_06_v1`.
