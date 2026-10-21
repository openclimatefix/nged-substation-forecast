"""Checks for experiment 151: how well the detected intervals match the injected transfers.

For every injected transfer at a scored site, find the detected interval whose onset is within
`ONSET_TOLERANCE` stamps and compare its MW and end with the truth, by transfer size (mean moved MW in points of the site's capacity). Also
compares the stored `known_from` of the R0_M48 intervals with the 60-half-hour call lag of the
6-hour lookahead rule of experiment 148.
"""

from __future__ import annotations

import e00151_fed as fed
import e00151_nra as nra
import e00151_run as run
import numpy as np
import polars as pl


def main() -> None:
    """Print the match table."""
    rows = []
    lags = []
    for site in fed.SERIES_OF_SITE:
        data = fed.get_site(site)
        offset = fed.stamp_offset(data)
        for seed in run.worlds_list():
            intervals, delta, events = run.load_world(seed=seed)
            d = np.nan_to_num(delta[f"delta_{fed.SERIES_OF_SITE[site]}"])
            if not np.any(d):
                continue
            pts = 100.0 * d / fed.CAPACITY_MW[site]
            spans = nra.detected_spans(data=data, site=site, intervals=intervals)
            r0 = intervals.filter(
                (pl.col("setting") == "R0_M48") & (pl.col("series") == fed.SERIES_OF_SITE[site])
            )
            lags.extend((r0["known_from"] - r0["end"]).to_list())
            for on, en in run.site_events(delta=d, events=events):
                mean = float(pts[on:en].mean())
                match = [s for s in spans if abs(s.onset - (on + offset)) <= nra.ONSET_TOLERANCE]
                step = match[0].amount if match else float("nan")
                rows.append(
                    {
                        "site": site,
                        "world": seed,
                        "hours": (en - on) / 2,
                        "true_mean": mean,
                        "detected": bool(match),
                        "detected_step": step,
                        "ratio": step / mean if match else float("nan"),
                        "end_error": (match[0].end - (en + offset)) if match else float("nan"),
                        "n_matches": len(match),
                    }
                )
    t = pl.DataFrame(rows)
    print(f"injected transfers at scored sites: {t.height}; detected {t['detected'].mean():.3f}")
    t = t.with_columns(
        pl.col("ratio").fill_nan(None),
        pl.col("end_error").fill_nan(None),
        size=pl.col("true_mean").abs().cut([1, 3, 6], labels=["<1", "1-3", "3-6", ">6"]),
    )
    print(
        t.group_by("size")
        .agg(
            n=pl.len(),
            detected=pl.col("detected").mean(),
            ratio_median=pl.col("ratio").median(),
            ratio_p10=pl.col("ratio").quantile(0.1),
            ratio_p90=pl.col("ratio").quantile(0.9),
            end_err_median=pl.col("end_error").median(),
        )
        .sort("size")
    )
    print(f"matched ratio of detected step to true mean: median {t['ratio'].median():.2f}")
    print(f"transfers with 2+ matched intervals: {(t['n_matches'] > 1).mean():.3f}")
    lag = np.array(lags)
    print(
        f"stored known_from minus end of R0_M48 intervals: median {np.median(lag):.0f}, "
        f"10th {np.percentile(lag, 10):.0f}, 90th {np.percentile(lag, 90):.0f} half-hours; "
        f"share under 60: {(lag < 60).mean():.3f}"
    )
    t.write_parquet(run.WORK_DIR / "match_table.parquet")


if __name__ == "__main__":
    main()
