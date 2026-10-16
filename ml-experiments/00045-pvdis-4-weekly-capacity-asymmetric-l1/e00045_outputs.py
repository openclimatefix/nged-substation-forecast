"""Assemble SW2-10's two files and print the real-series report (experiment 00045).

Reads the per-Primary tables that `e00045_real.py` and `e00045_placebo.py` wrote and writes, into
the experiment's private data folder:

- `capacity_weekly.parquet`: the final fit of every variant, one row per Primary, variant, and week.
- `capacity_weekly_vintages.parquet`: the same columns plus `origin_monday`, for `frozen` and
  `alt_post_hoc`. The vintage rows carry no bootstrap interval (`low_mw` and `high_mw` are null).
- `real_report.txt`: every number the summary quotes (per-Primary capacity, rises and falls, false
  steps on the placebo sky, the gap diagnostic, and the vintage ratio).

The file metadata of both parquet files names the recommended variant, with the reason, and
defines every column. The gap diagnostic compares each asymmetric variant with the free weekly fit
and with the symmetric fit (r = 1, the same `a`). A Primary is flagged when the mean of the
asymmetric path minus the comparison path over observed weeks is 5% of the asymmetric path's mean
or more (the plan's threshold, the weekly detection floor of a summer step), or when the
comparison path falls by 10% or more below its own trailing 26-week maximum for 8 or more
consecutive observed weeks and the asymmetric path is below that level in fewer than half of those
weeks. The flagged falls are listed for SW2-0b to check against switching labels; a held fall is
never a real fall.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_outputs.py
"""

import argparse
import json
from dataclasses import dataclass
from typing import Final

import e00045_common as common
import e00045_inject as inject
import e00045_model as model
import e00045_placebo as placebo
import e00045_real as real
import e00045_score as score
import numpy as np
import polars as pl

CAPACITY_PATH: Final = common.DATA_DIR / "capacity_weekly.parquet"
VINTAGES_PATH: Final = common.DATA_DIR / "capacity_weekly_vintages.parquet"
REPORT_PATH: Final = common.DATA_DIR / "real_report.txt"
MOVE_SHARE: Final[float] = 0.02
"""A weekly change of at least this share of the Primary's mean capacity is a rise or a fall."""
GAP_SHARE: Final[float] = 0.05
FALL_DEPTH: Final[float] = 0.10
FALL_WEEKS: Final[int] = 8
TRAILING_WEEKS: Final[int] = 26
EDGE_WEEKS: Final[int] = 4
"""The start and end capacity are means over this many first and last observed weeks."""
COLUMN_DOCS: Final[dict[str, str]] = {
    "week_start": "Monday of the week, Europe/London date",
    "time_series_id": "TimeSeriesMetadata time_series_id of the Primary",
    "variant": "frozen | alt_post_hoc | season_year_constant | free_weekly",
    "capacity_mw": "99th percentile of half-hourly AC output of the embedded solar, divided by "
    "the variant's recovery ratio; null where the free weekly fit has no value",
    "low_mw": "5th percentile of the 200-draw residual block bootstrap (null in the vintages file)",
    "high_mw": "95th percentile of the bootstrap (null in the vintages file)",
    "status": "clean | low-power | range-only | not-clean; read range-only as a range, not a value",
    "unobserved": "true when the week has fewer than 3 usable days; the interval is widened",
    "rooftop_low_mw": "capacity_mw minus 10%: farm-versus-rooftop shape scenario",
    "rooftop_high_mw": "capacity_mw plus 10%: farm-versus-rooftop shape scenario",
    "scenario_dayL_mw": "capacity_mw divided by 1 + B (experiment 00040's bound of an undetected "
    "day-long load)",
    "scenario_sunload_mw": "capacity_mw divided by 1 - 0.3336 (undetected sunshine-scaled load)",
    "recommended": "true for the variant SW2-10 should read for now",
}


POST_HOC_COVERAGE_PATH: Final = score.SCORES_DIR / "coverage_summary_post_hoc.parquet"
NOTES_MARKER: Final[str] = "Notes on the intervals and the recommendation:"
"""Heading of the closing section of `real_report.txt`, which `--notes-only` replaces."""


def clipped_table(*, final: pl.DataFrame) -> pl.DataFrame:
    """Count the observed weeks whose bootstrap interval was clipped to include the point.

    `e00045_real.fit_variant` replaces a bootstrap percentile that falls on the wrong side of the
    point by the point itself, so a clipped end of an observed week equals `capacity_mw` exactly.
    Unobserved weeks are widened afterwards and are left out.

    Args:
        final: `capacity_weekly.parquet`.

    Returns:
        One row per variant: the observed weeks with an interval, the weeks whose lower end, upper
        end, and either end equal the point, and the share of weeks with either end clipped.
    """
    seen = final.filter(
        ~pl.col("unobserved") & pl.col("capacity_mw").is_not_null() & pl.col("low_mw").is_not_null()
    ).with_columns(
        low_clipped=pl.col("low_mw") == pl.col("capacity_mw"),
        high_clipped=pl.col("high_mw") == pl.col("capacity_mw"),
    )
    return (
        seen.group_by("variant")
        .agg(
            observed_weeks=pl.len(),
            low_end_clipped=pl.col("low_clipped").sum(),
            high_end_clipped=pl.col("high_clipped").sum(),
            either_end_clipped=(pl.col("low_clipped") | pl.col("high_clipped")).sum(),
        )
        .with_columns(share_of_weeks=pl.col("either_end_clipped") / pl.col("observed_weeks"))
        .sort("variant")
    )


def post_hoc_coverage_text() -> str:
    """Return the coverage of the post hoc pair's 90% interval, from the post hoc scoring run."""
    coverage = pl.read_parquet(POST_HOC_COVERAGE_PATH).filter(pl.col("group") == "all")
    by_kind = {r["weeks"]: r["coverage_90"] for r in coverage.iter_rows(named=True)}
    return (
        f"non-step weeks {by_kind['non-step']:.3f} and step weeks {by_kind['step']:.3f} "
        "(target 0.90; the plan's condition (vi) asks for 0.85 to 0.95 and at least 0.60)"
    )


def interval_notes(*, final: pl.DataFrame) -> str:
    """Return the statements about the intervals and the recommendation that both outputs carry.

    Args:
        final: `capacity_weekly.parquet`.

    Returns:
        The statements: when the pair was scored, how the interval is clipped, and when the
        recommendation was made.
    """
    clipped = clipped_table(final=final)
    shares = ", ".join(
        f"{r['variant']} {r['share_of_weeks']:.1%}" for r in clipped.iter_rows(named=True)
    )
    return (
        "The interval of alt_post_hoc (a=2, r=10) was scored only post hoc: after the frozen "
        "pair failed the injection test, the 90% residual block bootstrap of alt_post_hoc was run "
        "on the same scoring units and 100 draws per unit, and covered the known path in "
        + post_hoc_coverage_text()
        + ". The frozen configuration, tuning code, limits, and verdict logic were not changed. "
        "Every interval is clipped to include the point estimate (a bootstrap percentile on the "
        "wrong side of the point is replaced by the point); the share of observed weeks with at "
        f"least one clipped end is {shares}. The recommendation of alt_post_hoc was made after "
        "seeing the scoring results, not before."
    )


def metadata(*, final: pl.DataFrame) -> dict[str, str]:
    """Return the parquet file metadata: the recommendation, its reason, and the column notes."""
    return {
        "recommended_variant": real.RECOMMENDED,
        "recommended_reason": real.RECOMMENDED_REASON,
        "columns": json.dumps(COLUMN_DOCS),
        "uncertainty": "low_mw/high_mw is sampling only; the rooftop and scenario columns are "
        "separate systematic scenarios and are never merged into it",
        "interval_and_recommendation_notes": interval_notes(final=final),
    }


def read_parts(*, prefix: str) -> pl.DataFrame:
    """Concatenate the per-Primary tables whose file name starts with `prefix`."""
    files = sorted(real.REAL_DIR.glob(f"{prefix}_*.parquet"))
    if len(files) != len(common.PRIMARIES):
        raise FileNotFoundError(f"{len(files)} {prefix} tables, expected {len(common.PRIMARIES)}")
    return pl.concat([pl.read_parquet(f) for f in files]).sort(
        "time_series_id",
        "variant",
        *(["origin_monday"] if prefix == "vintage" else []),
        "week_start",
    )


def check_final(*, final: pl.DataFrame) -> None:
    """Raise when the final table breaks its own contract."""
    assert final.select("time_series_id", "variant", "week_start").is_duplicated().sum() == 0
    seen = final.filter(pl.col("capacity_mw").is_not_null())
    assert (seen["capacity_mw"] >= 0).all()
    bounded = final.filter(pl.col("low_mw").is_not_null())
    assert (bounded["low_mw"] <= bounded["capacity_mw"] + 1e-9).all()
    assert (bounded["capacity_mw"] <= bounded["high_mw"] + 1e-9).all()
    assert set(final["variant"]) == set(real.VARIANTS)
    assert final.filter(pl.col("recommended"))["variant"].unique().to_list() == [real.RECOMMENDED]


def first_last(*, capacity: np.ndarray) -> tuple[float, float]:
    """Return the mean of the first and the last observed weeks' capacity."""
    seen = capacity[np.isfinite(capacity)]
    return float(seen[:EDGE_WEEKS].mean()), float(seen[-EDGE_WEEKS:].mean())


def rises_and_falls(*, capacity: np.ndarray) -> tuple[int, int]:
    """Count the weekly rises and falls of at least 2% of the mean capacity."""
    seen = capacity[np.isfinite(capacity)]
    delta = np.diff(seen)
    limit = MOVE_SHARE * seen.mean()
    return int((delta >= limit).sum()), int((delta <= -limit).sum())


def fall_episodes(*, path: np.ndarray, observed: np.ndarray) -> list[tuple[int, int]]:
    """Return runs of 8 or more observed weeks 10% or more below the trailing 26-week maximum.

    Args:
        path: Capacity of each week (NaN where absent).
        observed: Mask of observed weeks.

    Returns:
        (first index, last index) of each run, into `path`.
    """
    weeks = np.nonzero(observed & np.isfinite(path))[0]
    values = path[weeks]
    low = np.zeros(len(weeks), dtype=bool)
    for i in range(1, len(weeks)):
        peak = values[max(0, i - TRAILING_WEEKS) : i].max()
        low[i] = peak > 0 and (peak - values[i]) / peak >= FALL_DEPTH
    runs, start = [], None
    for i, flag in enumerate((*low, False)):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start >= FALL_WEEKS:
                runs.append((int(weeks[start]), int(weeks[i - 1])))
            start = None
    return runs


def below_level(*, path: np.ndarray, observed: np.ndarray, episode: tuple[int, int]) -> float:
    """Return the share of an episode's observed weeks in which `path` is 10% below its own peak."""
    weeks = np.nonzero(observed & np.isfinite(path))[0]
    values = path[weeks]
    inside = [i for i, w in enumerate(weeks) if episode[0] <= w <= episode[1]]
    hits = 0
    for i in inside:
        peak = values[max(0, i - TRAILING_WEEKS) : i].max() if i else values[i]
        hits += bool(peak > 0 and (peak - values[i]) / peak >= FALL_DEPTH)
    return hits / len(inside)


@dataclass(frozen=True)
class GapRow:
    """One Primary's gap diagnostic for one pair of paths."""

    primary: str
    asymmetric: str
    comparison: str
    mean_gap_share: float
    held_falls: int
    flagged: bool
    falls: str


def gap_rows(*, final: pl.DataFrame) -> list[GapRow]:
    """Compute the gap diagnostic for the two asymmetric variants against two comparisons."""
    inputs = inject.load_inputs()
    rows = []
    for site in common.PRIMARIES:
        design = model.design_from_half_hourly(
            half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site
        )
        series_id = common.PRIMARY_SERIES_IDS[site]
        mine = final.filter(pl.col("time_series_id") == series_id)

        def raw(variant: str, series_id_rows: pl.DataFrame = mine) -> tuple[np.ndarray, np.ndarray]:
            part = series_id_rows.filter(pl.col("variant") == variant)
            index = np.array(
                [(d - common.FIRST_WEEK_START.date()).days // 7 for d in part["week_start"]]
            )
            path = np.full(common.N_WEEKS, np.nan)
            observed = np.zeros(common.N_WEEKS, dtype=bool)
            ratio = part["recovery_ratio"][0]
            path[index] = part["capacity_mw"].fill_null(np.nan).to_numpy() * ratio
            observed[index] = ~part["unobserved"].to_numpy()
            return path, observed

        free, observed = raw("free_weekly")
        for variant in real.VINTAGE_VARIANTS:
            asym, _ = raw(variant)
            spec = real.VARIANTS[variant]
            symmetric = model.fit_path(design=design, a=spec[1], r=1.0)
            for label, other in (("free_weekly", free), (f"symmetric_a{spec[1]:g}", symmetric)):
                use = observed & np.isfinite(asym) & np.isfinite(other)
                gap = float((asym[use] - other[use]).mean() / asym[use].mean())
                episodes = fall_episodes(path=other, observed=observed)
                held = [
                    e
                    for e in episodes
                    if below_level(path=asym, observed=observed, episode=e) < 0.5
                ]
                falls = "; ".join(
                    f"{common.week_start(week=a).date()} to {common.week_start(week=b).date()}"
                    for a, b in held
                )
                rows.append(
                    GapRow(
                        primary=site,
                        asymmetric=variant,
                        comparison=label,
                        mean_gap_share=gap,
                        held_falls=len(held),
                        flagged=gap >= GAP_SHARE or bool(held),
                        falls=falls,
                    )
                )
    return rows


def summary_table(*, final: pl.DataFrame) -> pl.DataFrame:
    """Return capacity at the start and end, rises and falls, and interval widths per Primary."""
    out = []
    for site in common.PRIMARIES:
        for variant in real.VARIANTS:
            part = final.filter(
                (pl.col("time_series_id") == common.PRIMARY_SERIES_IDS[site])
                & (pl.col("variant") == variant)
            )
            capacity = part["capacity_mw"].fill_null(np.nan).to_numpy()
            start, end = first_last(capacity=capacity)
            rises, falls = rises_and_falls(capacity=capacity)
            observed = part.filter(~pl.col("unobserved"))
            out.append(
                {
                    "primary": site,
                    "name": common.PRIMARY_NAMES[site],
                    "status": real.STATUS[site],
                    "variant": variant,
                    "weeks": part.height,
                    "unobserved_weeks": int(part["unobserved"].sum()),
                    "start_mw": start,
                    "end_mw": end,
                    "rises": rises,
                    "falls": falls,
                    "recovery_ratio": part["recovery_ratio"][0],
                    "median_half_width_mw": float(
                        np.nanmedian(
                            ((observed["high_mw"] - observed["low_mw"]) / 2)
                            .fill_null(np.nan)
                            .to_numpy()
                        )
                    ),
                }
            )
    return pl.DataFrame(out)


def vintage_ratios(*, final: pl.DataFrame, vintages: pl.DataFrame) -> pl.DataFrame:
    """Return the first estimate of each origin's latest week over the same week in the final fit.

    Args:
        final: `capacity_weekly.parquet`.
        vintages: `capacity_weekly_vintages.parquet`.

    Returns:
        One row per Primary and variant: the median ratio, its 10th and 90th percentiles, the
        median absolute revision as a share of the final value, and the number of origins.
    """
    latest = vintages.filter(
        (pl.col("week_start") == pl.col("origin_monday") - pl.duration(days=7))
        & ~pl.col("unobserved")
    ).select("time_series_id", "variant", "origin_monday", "week_start", first="capacity_mw")
    joined = latest.join(
        final.filter(~pl.col("unobserved")).select(
            "time_series_id", "variant", "week_start", final="capacity_mw"
        ),
        on=["time_series_id", "variant", "week_start"],
    ).filter(pl.col("final") > 0)
    return (
        joined.with_columns(ratio=pl.col("first") / pl.col("final"))
        .group_by("time_series_id", "variant")
        .agg(
            median_ratio=pl.col("ratio").median(),
            p10=pl.col("ratio").quantile(0.1),
            p90=pl.col("ratio").quantile(0.9),
            median_abs_revision=(pl.col("ratio") - 1).abs().median(),
            n_origins=pl.len(),
        )
        .sort("time_series_id", "variant")
    )


def notes_section(*, final: pl.DataFrame) -> str:
    """Return the closing section of the report: the interval notes and the clipped-week counts."""
    return (
        NOTES_MARKER
        + "\n"
        + interval_notes(final=final)
        + "\n\nObserved weeks whose bootstrap interval was clipped to include the point "
        "(the interval ends of unobserved weeks are widened and not counted):\n"
        + score.table_text(clipped_table(final=final))
    )


def rewrite_notes_only() -> None:
    """Rewrite the two files' metadata and the report's notes, without refitting anything."""
    final = pl.read_parquet(CAPACITY_PATH)
    vintages = pl.read_parquet(VINTAGES_PATH)
    final.write_parquet(CAPACITY_PATH, metadata=metadata(final=final))
    vintages.write_parquet(VINTAGES_PATH, metadata=metadata(final=final))
    text = REPORT_PATH.read_text().split("\n\n" + NOTES_MARKER)[0].rstrip("\n")
    REPORT_PATH.write_text(text + "\n\n" + notes_section(final=final) + "\n")
    print(notes_section(final=final))


def main() -> None:
    """Write the two parquet files and the report (`--notes-only`: only the metadata and notes)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="rewrite the metadata and the report's notes from the existing files",
    )
    if parser.parse_args().notes_only:
        rewrite_notes_only()
        return
    final = read_parts(prefix="final")
    vintages = read_parts(prefix="vintage")
    check_final(final=final)
    final.write_parquet(CAPACITY_PATH, metadata=metadata(final=final))
    vintages.write_parquet(VINTAGES_PATH, metadata=metadata(final=final))
    placebo_tables = pl.concat(
        [pl.read_parquet(f) for f in sorted(placebo.PLACEBO_DIR.glob("placebo_*.parquet"))]
    )
    gaps = pl.DataFrame([g.__dict__ for g in gap_rows(final=final)])
    all_vintage = vintage_ratios(final=final, vintages=vintages)
    pooled = (
        all_vintage.group_by("variant")
        .agg(
            median_of_median_ratio=pl.col("median_ratio").median(),
            median_abs_revision=pl.col("median_abs_revision").median(),
        )
        .sort("variant")
    )
    ratios_text = (
        "Recovery ratio per Primary and variant (median over tuning farms A to C; the frozen "
        "config's asymmetric ratios for comparison are in frozen_config.json):\n"
        + score.table_text(
            final.group_by("time_series_id", "variant")
            .agg(ratio=pl.col("recovery_ratio").first())
            .sort("time_series_id", "variant")
        )
    )
    text = "\n\n".join(
        [
            f"Real-series fits: {final.height} rows in {CAPACITY_PATH.name}, "
            f"{vintages.height} rows in {VINTAGES_PATH.name}; bootstrap {real.N_DRAWS} draws per "
            "Primary and variant; recommended variant " + real.RECOMMENDED + ".",
            "Per Primary and variant (start and end are means of the first and last "
            f"{EDGE_WEEKS} observed weeks; a rise or fall is a weekly change of at least "
            f"{MOVE_SHARE:.0%} of the mean capacity):\n"
            + score.table_text(summary_table(final=final)),
            ratios_text,
            "False steps on the placebo sky (200 permutations per Primary; a false step is a rise "
            "of 10% of mean capacity over 4 weeks, not within 4 weeks of a real rise of 10%):\n"
            + score.table_text(placebo.summarise(tables=placebo_tables)),
            "Gap diagnostic (flagged when the mean gap is "
            f"{GAP_SHARE:.0%} of capacity or more, or the comparison path falls "
            f"{FALL_DEPTH:.0%} for {FALL_WEEKS} weeks and the asymmetric path does not):\n"
            + score.table_text(gaps.drop("falls")),
            "Held falls listed for SW2-0b:\n"
            + "\n".join(
                f"{r['primary']} {r['asymmetric']} vs {r['comparison']}: {r['falls']}"
                for r in gaps.filter(pl.col("held_falls") > 0).iter_rows(named=True)
            ),
            "Vintage ratio (first estimate of the latest week over the same week in the final "
            "fit, per Primary):\n"
            + score.table_text(all_vintage)
            + "\nPooled:\n"
            + score.table_text(pooled),
            notes_section(final=final),
        ]
    )
    REPORT_PATH.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
