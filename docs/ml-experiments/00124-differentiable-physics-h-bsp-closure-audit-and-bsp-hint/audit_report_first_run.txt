# Experiment 124, step 0: the BSP closure audit

**First-run report, kept as printed before the code review; counts of series per BSP are removed. The re-run after the review is `audit_report.md`.**

### Bulk supply point 1


#### Cut tuning (2023-07-01): 15119 training rows (0.9 years)
- cannot tell: fewer than one year of complete rows before the cut

#### Cut replicate (2024-07-01): 30881 training rows (1.8 years)
- step penalty 1000; a_P 1.15 (0.91 to 1.34); a_G 0.24 (0.18 to 0.28); c 0.64 (0.53 to 0.77); unmetered PV capacity q 19.35 MW (17.01 to 22.16); heating h 8.58
- in-sample residual: robust half-hourly scale 5.47 MW = 11.5 points of effective capacity
- in-sample standard deviation of the residual's block means (MW): 6 h 5.41, 24 h 4.24, 72 h 3.19, 30 d 1.05, 90 d 0.13
- injected stuck 24 h: detected 1% with a refit, 1% with the pre-injection fit (300 injections; false-alarm rate 0.6%)
- injected scale 30 d 10%: detected 0% with a refit, 0% with the pre-injection fit (300 injections; false-alarm rate 2.1%)
- injected scale 30 d 25%: detected 0% with a refit, 8% with the pre-injection fit (300 injections; false-alarm rate 2.1%)
- injected scale 30 d 40%: detected 1% with a refit, 35% with the pre-injection fit (300 injections; false-alarm rate 2.1%)
- injected scale 90 d 10%: detected 36% with a refit, 65% with the pre-injection fit (300 injections; false-alarm rate 42.8%)
- injected scale 90 d 25%: detected 41% with a refit, 93% with the pre-injection fit (300 injections; false-alarm rate 42.8%)
- injected scale 90 d 40%: detected 53% with a refit, 100% with the pre-injection fit (300 injections; false-alarm rate 42.8%)
- scored-year residual (frozen fit): median 0.63 MW, robust scale 6.45 MW = 13.6 points; block-mean sd (MW): 6 h 6.21, 24 h 5.24, 72 h 4.60, 30 d 3.27, 90 d cannot tell

#### Cut scored (2025-07-01): 47142 training rows (2.7 years)
- step penalty 10; a_P 1.37 (1.21 to 1.51); a_G 0.24 (0.19 to 0.29); c 0.63 (0.55 to 0.70); unmetered PV capacity q 19.31 MW (17.43 to 20.85); heating h 7.74
- in-sample residual: robust half-hourly scale 5.30 MW = 11.1 points of effective capacity
- in-sample standard deviation of the residual's block means (MW): 6 h 5.35, 24 h 4.20, 72 h 3.07, 30 d 0.64, 90 d 0.21
- in-sample: median residual (MW) by hour band: night +0.61, morning +1.10, midday +1.19, evening +1.14
- in-sample: median residual (MW) by calendar month: 1 +1.40, 2 +1.23, 3 +0.91, 4 +1.24, 5 +0.92, 6 +0.85, 7 +0.66, 8 +0.52, 9 +0.76, 10 +1.06, 11 +1.14, 12 +1.51
- in-sample: median residual at local 10:00 to 14:00: +1.25 MW
- in-sample, winter ending 2023: 151 days; median +1.10 MW, robust scale 5.75 MW, 24 h block-mean sd 4.29 MW; cold-snap days (16): median +2.76 MW
- in-sample, winter ending 2024: 141 days; median +1.32 MW, robust scale 6.58 MW, 24 h block-mean sd 4.92 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample, winter ending 2025: 151 days; median +1.26 MW, robust scale 5.69 MW, 24 h block-mean sd 4.90 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample: 3 independent winters with at least 80 days of residuals
- scored year: median residual (MW) by hour band: night +1.11, morning +0.53, midday -1.11, evening +0.04
- scored year: median residual (MW) by calendar month: 1 -4.77, 2 -1.89, 3 -0.41, 4 -2.18, 5 +0.76, 6 +2.85, 7 +4.34, 8 +4.84, 9 +3.18, 10 +3.69, 11 -2.69, 12 -3.51
- scored year: median residual at local 10:00 to 14:00: -1.19 MW
- scored year, winter ending 2026: 151 days; median -2.84 MW, robust scale 6.69 MW, 24 h block-mean sd 5.11 MW; cold-snap days: cannot tell (fewer than 3)
- scored year: 1 independent winters with at least 80 days of residuals
- scored-year residual (frozen fit): median 0.14 MW, robust scale 7.28 MW = 15.3 points; block-mean sd (MW): 6 h 6.90, 24 h 5.73, 72 h 4.81, 30 d 3.53, 90 d cannot tell
### Bulk supply point 2


#### Cut tuning (2023-07-01): 59275 training rows (3.4 years)
- step penalty 10; a_P 1.01 (0.92 to 1.09); a_G 0.85 (0.81 to 0.89); c 1.03 (0.96 to 1.11); unmetered PV capacity q 13.23 MW (12.15 to 14.11); heating h 1.23
- in-sample residual: robust half-hourly scale 1.24 MW = 2.2 points of effective capacity
- in-sample standard deviation of the residual's block means (MW): 6 h 1.49, 24 h 1.23, 72 h 0.99, 30 d 0.42, 90 d 0.11
- injected stuck 24 h: detected 0% with a refit, 0% with the pre-injection fit (200 injections; false-alarm rate 1.9%)
- injected scale 30 d 10%: detected 0% with a refit, 0% with the pre-injection fit (200 injections; false-alarm rate 1.2%)
- injected scale 30 d 25%: detected 0% with a refit, 50% with the pre-injection fit (200 injections; false-alarm rate 1.2%)
- injected scale 30 d 40%: detected 12% with a refit, 61% with the pre-injection fit (200 injections; false-alarm rate 1.2%)
- injected scale 90 d 10%: detected 0% with a refit, 56% with the pre-injection fit (200 injections; false-alarm rate 2.8%)
- injected scale 90 d 25%: detected 2% with a refit, 100% with the pre-injection fit (200 injections; false-alarm rate 2.8%)
- injected scale 90 d 40%: detected 26% with a refit, 100% with the pre-injection fit (200 injections; false-alarm rate 2.8%)
- scored-year residual (frozen fit): median 1.72 MW, robust scale 2.56 MW = 4.6 points; block-mean sd (MW): 6 h 2.81, 24 h 2.29, 72 h 2.15, 30 d 2.06, 90 d cannot tell

#### Cut replicate (2024-07-01): 75543 training rows (4.3 years)
- step penalty 1000; a_P 1.01 (0.95 to 1.17); a_G 0.90 (0.85 to 0.93); c 1.08 (1.01 to 1.14); unmetered PV capacity q 13.91 MW (12.99 to 14.59); heating h 1.19
- in-sample residual: robust half-hourly scale 1.40 MW = 2.5 points of effective capacity
- in-sample standard deviation of the residual's block means (MW): 6 h 1.78, 24 h 1.44, 72 h 1.21, 30 d 0.87, 90 d 0.45
- injected stuck 24 h: detected 0% with a refit, 0% with the pre-injection fit (200 injections; false-alarm rate 1.7%)
- injected scale 30 d 10%: detected 0% with a refit, 0% with the pre-injection fit (200 injections; false-alarm rate 0.9%)
- injected scale 30 d 25%: detected 3% with a refit, 8% with the pre-injection fit (200 injections; false-alarm rate 0.9%)
- injected scale 30 d 40%: detected 8% with a refit, 28% with the pre-injection fit (200 injections; false-alarm rate 0.9%)
- injected scale 90 d 10%: detected 0% with a refit, 8% with the pre-injection fit (200 injections; false-alarm rate 0.0%)
- injected scale 90 d 25%: detected 2% with a refit, 28% with the pre-injection fit (200 injections; false-alarm rate 0.0%)
- injected scale 90 d 40%: detected 8% with a refit, 62% with the pre-injection fit (200 injections; false-alarm rate 0.0%)
- scored-year residual (frozen fit): median -0.72 MW, robust scale 3.10 MW = 5.6 points; block-mean sd (MW): 6 h 4.14, 24 h 2.36, 72 h 2.19, 30 d 1.89, 90 d cannot tell

#### Cut scored (2025-07-01): 91621 training rows (5.2 years)
- step penalty 1000; a_P 1.16 (1.07 to 1.34); a_G 0.89 (0.84 to 0.91); c 1.10 (1.03 to 1.17); unmetered PV capacity q 15.95 MW (14.41 to 17.07); heating h 0.96
- in-sample residual: robust half-hourly scale 1.79 MW = 3.2 points of effective capacity
- in-sample standard deviation of the residual's block means (MW): 6 h 2.28, 24 h 1.55, 72 h 1.33, 30 d 0.99, 90 d 0.58
- in-sample: median residual (MW) by hour band: night -0.21, morning +0.05, midday +0.51, evening -0.09
- in-sample: median residual (MW) by calendar month: 1 -0.09, 2 -0.40, 3 -0.35, 4 -0.09, 5 -0.39, 6 +0.55, 7 +0.32, 8 +0.02, 9 -0.40, 10 +0.08, 11 +0.08, 12 +0.54
- in-sample: median residual at local 10:00 to 14:00: +0.50 MW
- in-sample, winter ending 2020: 141 days; median +0.03 MW, robust scale 2.21 MW, 24 h block-mean sd 3.03 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample, winter ending 2021: 151 days; median -0.44 MW, robust scale 1.05 MW, 24 h block-mean sd 0.74 MW; cold-snap days (16): median -0.71 MW
- in-sample, winter ending 2022: 151 days; median -0.96 MW, robust scale 1.96 MW, 24 h block-mean sd 1.35 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample, winter ending 2023: 151 days; median -0.17 MW, robust scale 1.25 MW, 24 h block-mean sd 0.86 MW; cold-snap days (15): median -0.66 MW
- in-sample, winter ending 2024: 152 days; median +0.69 MW, robust scale 1.44 MW, 24 h block-mean sd 0.92 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample, winter ending 2025: 149 days; median +0.86 MW, robust scale 2.55 MW, 24 h block-mean sd 1.50 MW; cold-snap days: cannot tell (fewer than 3)
- in-sample: 6 independent winters with at least 80 days of residuals
- scored year: median residual (MW) by hour band: night +4.39, morning +2.97, midday +2.59, evening +4.19
- scored year: median residual (MW) by calendar month: 1 +4.80, 2 +4.30, 3 +4.57, 4 +2.83, 5 -1.20, 6 -0.70, 7 +1.48, 8 +1.46, 9 +0.95, 10 +3.11, 11 +5.39, 12 +5.87
- scored year: median residual at local 10:00 to 14:00: +2.19 MW
- scored year, winter ending 2026: 148 days; median +4.93 MW, robust scale 1.25 MW, 24 h block-mean sd 1.28 MW; cold-snap days: cannot tell (fewer than 3)
- scored year: 1 independent winters with at least 80 days of residuals
- scored-year residual (frozen fit): median 3.67 MW, robust scale 3.02 MW = 5.5 points; block-mean sd (MW): 6 h 3.66, 24 h 2.33, 72 h 2.04, 30 d 1.80, 90 d cannot tell
