"""Score the eight recalibrations of experiment 97 on the stored BG2 forecasts.

For each scored month the script fits every arm on the rows whose valid time is before the month and
applies it to the runs initialised in the month, then prints coverage, pinball loss, and the node
curtailment cost. It writes `report.md` and the tables behind the figures into the private data
folder.

    uv run python ml-experiments/00097-*/e00097_report.py
"""

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import e00097_recal as recal
import numpy as np
import polars as pl
from nged_data.storage import scan_cleaned_power
from studies.bootstrap import bootstrap_month_totals
from studies.experiment_store import read_predictions
from studies.node_curtailment import (
    SCALED_LIMIT_FRACTIONS,
    NodeCurtailmentAssumptions,
    NodeTable,
    avoidable_cost,
    event_census,
    month_total_arrays,
    node_curtailment,
    scale_to_megawatts,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 97
SLUG: Final[str] = "cal1-post-hoc-recalibration-of-bg2-quantiles-tails"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
RESULTS_DIR: Final[Path] = DATA_DIR / "results"
SOURCE_EXPERIMENT_ID: Final[int] = 87
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
SET_NAMES: Final[Mapping[str, str]] = {
    "primary": "p_s4_weekly",
    "supply": "b_weekly",
}
"""The stored sets read: Primary substations, and bulk and grid supply points."""

CLASSES: Final[tuple[str, ...]] = ("Primary", "BSP/GSP")
BAND_EDGES_H: Final[tuple[float, ...]] = (12.0, 36.0, 96.0)
BANDS: Final[tuple[str, ...]] = ("<12 h", "12-36 h", "36-96 h", ">=96 h")
DECISION_BAND: Final[int] = 1
FIRST_SCORED_MONTH: Final[dt.datetime] = dt.datetime(2025, 10, 1, tzinfo=dt.UTC)
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
"""Scaled curtailment limits use only flows before this instant, as experiment 84 does."""
SCORED_END: Final[dt.datetime] = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
TAIL_TOLERANCE: Final[float] = 0.01
MIN_EXCEEDANCE_DAYS: Final[int] = 30
CENTRE_TOLERANCE: Final[float] = 0.01
EXCEEDANCE_LEVELS: Final[tuple[float, ...]] = (0.001, 0.005, 0.01, 0.05, 0.95, 0.99, 0.995, 0.999)
REPORTED_LEVELS: Final[tuple[float, ...]] = (0.005, 0.01, 0.05, 0.95, 0.99, 0.995)
SCORED_NODES: Final[tuple[str, ...]] = ("BOSTON 33kV S STN", "SKEGNESS 33kV S STN")
CLEANED_POWER_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "cleaned_power_time_series.delta"
)
NODE_TABLE_DIR: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00084-*")) / "private"
"""The private node, limit, and generator tables that experiment 84 wrote."""
QUANTILE_NAMES: Final[tuple[str, ...]] = tuple(
    f"q{round(p * 100):02d}" for p in recal.DELIVERED_LEVELS
)
N_COLUMNS: Final[int] = len(recal.OUT_LEVELS)


def read_rows() -> pl.DataFrame:
    """Read both stored sets into one frame with a class and a lead-band code per row."""
    frames = []
    for class_code, name in enumerate(SET_NAMES.values()):
        frame = read_predictions(
            experiment_id=SOURCE_EXPERIMENT_ID, name=name, row_set_id=ROW_SET_ID
        ).with_columns(pl.col("time_series_id").cast(pl.Int64), class_code=pl.lit(class_code))
        frames.append(frame)
    both = pl.concat(frames)
    finite = pl.all_horizontal(pl.col("actual", *QUANTILE_NAMES).is_finite())
    both = both.filter(finite)
    band = pl.lit(0)
    for edge in BAND_EDGES_H:
        band = band + (pl.col("lead_hours") >= edge).cast(pl.Int64)
    return both.with_columns(band_code=band).sort(
        "power_fcst_init_time", "time_series_id", "valid_time"
    )


def stored_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the sorted (n, 15) float32 quantile matrix with the extrapolated extra levels."""
    delivered = frame.select(QUANTILE_NAMES).to_numpy().astype(np.float64)
    return recal.extend_quantiles(delivered=delivered).astype(np.float32)


def rows_of(*, actual: np.ndarray, matrix: np.ndarray, index: np.ndarray) -> recal.Rows:
    """Slice rows into a float64 `Rows`."""
    return recal.Rows(
        actual=actual[index].astype(np.float64), quantiles=matrix[index].astype(np.float64)
    )


def pinball_and_cover(
    *, actual: np.ndarray, quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row pinball loss and the at-or-below indicator for each output level."""
    levels = np.array(recal.OUT_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return np.maximum(levels * error, (levels - 1.0) * error), (error <= 0).astype(np.float64)


def main() -> None:  # noqa: PLR0915
    """Run every arm over every scored month and write the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = read_rows()
    actual = frame["actual"].to_numpy().astype(np.float64)
    matrix = stored_matrix(frame=frame)
    valid = frame["valid_time"].to_numpy()
    init = frame["power_fcst_init_time"]
    month_label = init.dt.strftime("%Y-%m").to_numpy()
    class_code = frame["class_code"].to_numpy()
    band_code = frame["band_code"].to_numpy()
    series = frame["time_series_id"].to_numpy()
    group = class_code * len(BANDS) + band_code
    n_groups = len(CLASSES) * len(BANDS)
    months = sorted(m for m in set(month_label) if m >= FIRST_SCORED_MONTH.strftime("%Y-%m"))
    arms = recal.ARMS
    sums = {a: np.zeros((len(months), n_groups, N_COLUMNS, 2)) for a in arms}
    counts = np.zeros((len(months), n_groups))
    exceed_days: dict[tuple[str, int, int], set[int]] = {}
    node_ids = frame.filter(pl.col("time_series_id").is_in([1, 2]) & (pl.col("class_code") == 1))[
        "time_series_id"
    ].unique()
    node_mask = np.isin(series, node_ids.to_numpy()) & (class_code == 1)
    node_position = np.full(len(frame), -1)
    node_rows_all = np.flatnonzero(node_mask & np.isin(month_label, months))
    node_position[node_rows_all] = np.arange(len(node_rows_all))
    node_q = {a: np.zeros((len(node_rows_all), len(recal.DELIVERED_COLUMNS))) for a in arms}
    day_key = (valid.astype("datetime64[D]").astype(np.int64)) * 1000 + series
    print(f"rows {len(frame):,}; scored months {months}")

    for m_idx, month in enumerate(months):
        month_start = dt.datetime.strptime(month, "%Y-%m").replace(tzinfo=dt.UTC)
        in_month = month_label == month
        calibration_all = valid < np.datetime64(month_start.replace(tzinfo=None))
        recent_start = np.datetime64(
            (month_start - dt.timedelta(days=recal.RECENT_DAYS)).replace(tzinfo=None)
        )
        first_init = init.filter(pl.Series(in_month)).min()
        last_calibration = frame["valid_time"].filter(pl.Series(calibration_all)).max()
        if not (last_calibration is not None and last_calibration < first_init):  # ty: ignore[unsupported-operator]
            msg = f"leak: calibration reaches {last_calibration}, first scored run is {first_init}"
            raise AssertionError(msg)
        counts[m_idx] = np.bincount(group[in_month], minlength=n_groups)
        for cls in range(len(CLASSES)):
            for band in range(len(BANDS)):
                g = cls * len(BANDS) + band
                test_idx = np.flatnonzero(in_month & (group == g))
                if not len(test_idx):
                    continue
                cls_cal = np.flatnonzero(calibration_all & (class_code == cls))
                band_cal = np.flatnonzero(calibration_all & (group == g))
                recent_cal = np.flatnonzero(
                    calibration_all & (group == g) & (valid >= recent_start)
                )
                test = rows_of(actual=actual, matrix=matrix, index=test_idx)
                for arm, (function, grouping) in arms.items():
                    cal_idx = {
                        "none": band_cal,
                        "class": cls_cal,
                        "class_band": band_cal,
                        "class_band_recent": recent_cal,
                    }[grouping]
                    calibration = rows_of(actual=actual, matrix=matrix, index=cal_idx)
                    out = function(calibration=calibration, test=test)
                    pin, cover = pinball_and_cover(actual=test.actual, quantiles=out)
                    sums[arm][m_idx, g, :, 0] += cover.sum(axis=0)
                    sums[arm][m_idx, g, :, 1] += pin.sum(axis=0)
                    for level in EXCEEDANCE_LEVELS:
                        j = recal.OUT_LEVELS.index(level)
                        hit = cover[:, j] > 0 if level < 0.5 else cover[:, j] == 0
                        exceed_days.setdefault((arm, g, j), set()).update(
                            np.unique(day_key[test_idx[hit]]).tolist()
                        )
                    position = node_position[test_idx]
                    keep = position >= 0
                    if keep.any():
                        node_q[arm][position[keep]] = out[keep][:, list(recal.DELIVERED_COLUMNS)]
        print(f"month {month} done", flush=True)
    print(f"scale factors clipped: {recal.CLIPS.clipped} of {recal.CLIPS.fits} fits")

    results = build_tables(sums=sums, counts=counts, exceed_days=exceed_days, months=months)
    for name, table in results.items():
        table.write_parquet(RESULTS_DIR / f"{name}.parquet")
    node_frame = frame[node_rows_all.tolist()]
    curtailment = score_curtailment(node_frame=node_frame, node_q=node_q)
    verdict_lines = decision(tables=results)
    report = render(tables=results, curtailment=curtailment, decision_lines=verdict_lines)
    (DATA_DIR / "report.md").write_text(report)
    print(report)


def interval(*, month_sums: np.ndarray, month_counts: np.ndarray) -> tuple[float, float, float]:
    """Return the mean and its 95% interval from resampling whole months."""
    mean = float(month_sums.sum() / month_counts.sum())
    low, high = bootstrap_month_totals(month_sums=month_sums[None, :], month_counts=month_counts)
    return mean, low, high


def build_tables(
    *,
    sums: Mapping[str, np.ndarray],
    counts: np.ndarray,
    exceed_days: Mapping[tuple[str, int, int], set[int]],
    months: list[str],
) -> dict[str, pl.DataFrame]:
    """Build the coverage, centre, quantile-score, and by-month tables for every arm and cell."""
    cells: list[tuple[str, str, list[int]]] = []
    for c, cls in enumerate(CLASSES):
        for b, band in enumerate(BANDS):
            cells.append((cls, band, [c * len(BANDS) + b]))
        cells.append((cls, "all", [c * len(BANDS) + b for b in range(len(BANDS))]))
    centre_cols = [recal.OUT_LEVELS.index(p) for p in recal.CENTRE_LEVELS]
    coverage_rows, score_rows, month_rows = [], [], []
    baseline = "a0_as_stored"
    for arm, arr in sums.items():
        for cls, band, gs in cells:
            n_m = counts[:, gs].sum(axis=1)
            if n_m.sum() == 0:
                continue
            for level in recal.OUT_LEVELS:
                j = recal.OUT_LEVELS.index(level)
                cov_m = arr[:, gs, j, 0].sum(axis=1)
                pin_m = arr[:, gs, j, 1].sum(axis=1)
                cov, cov_lo, cov_hi = interval(month_sums=cov_m, month_counts=n_m)
                pin, pin_lo, pin_hi = interval(month_sums=pin_m, month_counts=n_m)
                days = len(set().union(*(exceed_days.get((arm, g, j), set()) for g in gs)))
                coverage_rows.append(
                    {
                        "arm": arm,
                        "class": cls,
                        "band": band,
                        "level": level,
                        "n_rows": int(n_m.sum()),
                        "coverage": cov,
                        "coverage_low": cov_lo,
                        "coverage_high": cov_hi,
                        "pinball": pin,
                        "pinball_low": pin_lo,
                        "pinball_high": pin_hi,
                        "exceedance_rows": round(
                            cov_m.sum() if level < 0.5 else n_m.sum() - cov_m.sum()
                        ),
                        "exceedance_days": days if level in EXCEEDANCE_LEVELS else None,
                    }
                )
            for name, cols in (
                ("centre", centre_cols),
                ("quantile_score", list(recal.DELIVERED_COLUMNS)),
            ):
                arm_m = arr[:, gs][:, :, cols, 1].sum(axis=(1, 2)) / len(cols)
                base_m = sums[baseline][:, gs][:, :, cols, 1].sum(axis=(1, 2)) / len(cols)
                score, lo, hi = interval(month_sums=arm_m, month_counts=n_m)
                diff, dlo, dhi = interval(month_sums=arm_m - base_m, month_counts=n_m)
                score_rows.append(
                    {
                        "arm": arm,
                        "class": cls,
                        "band": band,
                        "score": name,
                        "value": score,
                        "low": lo,
                        "high": hi,
                        "diff_vs_a0": diff,
                        "diff_low": dlo,
                        "diff_high": dhi,
                    }
                )
        for m_idx, month in enumerate(months):
            for c, cls in enumerate(CLASSES):
                g = c * len(BANDS) + DECISION_BAND
                if counts[m_idx, g] == 0:
                    continue
                month_rows.append(
                    {
                        "arm": arm,
                        "class": cls,
                        "month": month,
                        **{
                            f"coverage_{level}": float(
                                arr[m_idx, g, recal.OUT_LEVELS.index(level), 0] / counts[m_idx, g]
                            )
                            for level in (0.05, 0.95, 0.01, 0.99)
                        },
                    }
                )
    return {
        "coverage": pl.DataFrame(coverage_rows),
        "scores": pl.DataFrame(score_rows),
        "by_month": pl.DataFrame(month_rows),
    }


def decision(*, tables: Mapping[str, pl.DataFrame]) -> list[str]:
    """Apply the pre-registered decision rule to every arm and return printable lines."""
    cov, scores = tables["coverage"], tables["scores"]
    lines = []
    for arm in (a for a in recal.ARMS if a not in recal.POST_HOC_ARMS):
        tail_ok, centre_ok, testable = True, True, True
        notes = []
        for cls in CLASSES:
            for level in (0.01, 0.05, 0.95, 0.99):
                row = cov.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("class") == cls)
                    & (pl.col("band") == BANDS[DECISION_BAND])
                    & (pl.col("level") == level)
                ).row(0, named=True)
                if (row["exceedance_days"] or 0) < MIN_EXCEEDANCE_DAYS:
                    testable = False
                    notes.append(f"{cls} q{level}: not testable ({row['exceedance_days']} days)")
                elif abs(row["coverage"] - level) > TAIL_TOLERANCE:
                    tail_ok = False
                    notes.append(f"{cls} q{level}: coverage {row['coverage']:.3f}")
            centre = scores.filter(
                (pl.col("arm") == arm)
                & (pl.col("class") == cls)
                & (pl.col("band") == "all")
                & (pl.col("score") == "centre")
            ).row(0, named=True)
            base = scores.filter(
                (pl.col("arm") == "a0_as_stored")
                & (pl.col("class") == cls)
                & (pl.col("band") == "all")
                & (pl.col("score") == "centre")
            ).row(0, named=True)
            if centre["diff_high"] > CENTRE_TOLERANCE * base["value"]:
                centre_ok = False
                notes.append(
                    f"{cls} centre worse: upper {centre['diff_high']:.5f} "
                    f"vs {CENTRE_TOLERANCE * base['value']:.5f}"
                )
        lines.append(
            f"{arm}: tail_pass={tail_ok} centre_pass={centre_ok} all_testable={testable}; "
            + "; ".join(notes)
        )
    return lines


def read_node_table() -> NodeTable:
    """Read experiment 84's private tables for the two scored bulk supply points."""
    nodes = (
        pl.read_parquet(NODE_TABLE_DIR / "nodes.parquet")
        .filter(pl.col("node_id").is_in(SCORED_NODES))
        .with_columns(parent_node_id=pl.lit(None, dtype=pl.String))
    )
    limits = pl.read_parquet(NODE_TABLE_DIR / "limits.parquet").filter(
        pl.col("node_id").is_in(SCORED_NODES)
    )
    generators = pl.read_parquet(NODE_TABLE_DIR / "generators.parquet").filter(
        pl.col("node_id").is_in(SCORED_NODES)
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators)


def read_flows(*, table: NodeTable) -> pl.DataFrame:
    """Return the metered flows of the table's nodes from cleaned power before the holdout."""
    unflagged = scan_cleaned_power(str(CLEANED_POWER_PATH))
    plain = pl.LazyFrame._from_pyldf(unflagged._ldf)  # strip the Patito model, zero-copy
    return (
        plain.filter(
            pl.col("time_series_id").is_in(table.nodes["time_series_id"].to_list()),
            pl.col("time") < SCORED_END,
        )
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            pl.col("time").alias("valid_time"),
            flow_mw=pl.col("power").cast(pl.Float64),
        )
        .collect()
    )


def score_curtailment(
    *, node_frame: pl.DataFrame, node_q: Mapping[str, np.ndarray]
) -> dict[str, dict[str, pl.DataFrame]]:
    """Score every arm's node forecasts, node-only, at the real and scaled limits."""
    table = read_node_table()
    flows = read_flows(table=table)
    capacity = {
        int(s): float(c)
        for s, c in node_frame.group_by("time_series_id")
        .agg(pl.col("capacity_mw").first())
        .iter_rows()
    }
    base = node_frame.select(
        "time_series_id", "power_fcst_init_time", "valid_time", "seed", "actual"
    )
    arm_frames: dict[str, pl.DataFrame] = {}
    for arm, q in node_q.items():
        arm_frames[arm] = scale_to_megawatts(
            frame=base.with_columns(*(pl.Series(n, q[:, i]) for i, n in enumerate(QUANTILE_NAMES))),
            capacity_mw=capacity,
        )
    perfect = scale_to_megawatts(
        frame=base.with_columns(*(pl.col("actual").alias(n) for n in QUANTILE_NAMES)),
        capacity_mw=capacity,
    )
    arm_frames["perfect"] = perfect
    assumptions = NodeCurtailmentAssumptions(use_generators=False)
    scored_flows = flows.filter(pl.col("valid_time") >= FIRST_SCORED_MONTH)
    out: dict[str, dict[str, pl.DataFrame]] = {}
    for fraction in SCALED_LIMIT_FRACTIONS:
        scaled = scaled_limits(table=table, flows=flows, cutoff=SCORED_START, fraction=fraction)
        scaled_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction)
        result = node_curtailment(
            table=scaled_table,
            node_forecasts=arm_frames,
            generator_forecasts={},
            assumptions=assumptions,
        )
        census = event_census(table=table, flows=scored_flows, scaled={fraction: scaled}).filter(
            pl.col("limit_kind").str.starts_with("scaled")
        )
        out[f"{fraction}"] = {
            "per_arm": result.per_arm,
            "avoidable": avoidable_cost(per_arm=result.per_arm, perfect_arm="perfect"),
            "census": census,
            "intervals": curtailment_intervals(result=result),
        }
    return out


def curtailment_intervals(*, result) -> pl.DataFrame:  # noqa: ANN001
    """Return each arm's avoidable cost at the headline price with a month-resampled interval."""
    column = "energy_proxy_mwh_equivalent"
    price = 2000.0
    perfect, counts = month_total_arrays(
        month_totals=result.month_totals, arm="perfect", column=column
    )
    rows = []
    for arm in sorted(result.month_totals["arm"].unique()):
        sums, _ = month_total_arrays(month_totals=result.month_totals, arm=arm, column=column)
        diff = sums - perfect
        low, high = bootstrap_month_totals(month_sums=diff, month_counts=counts)
        total = float(counts.sum()) * price
        rows.append(
            {
                "arm": arm,
                "avoidable_gbp": float(diff.sum(axis=1).mean()) * price,
                "low_gbp": low * total,
                "high_gbp": high * total,
            }
        )
    return pl.DataFrame(rows)


def render(
    *,
    tables: Mapping[str, pl.DataFrame],
    curtailment: dict[str, dict[str, pl.DataFrame]],
    decision_lines: list[str],
) -> str:
    """Render every number the page quotes as markdown."""
    cov = tables["coverage"]
    lines = [
        "# Experiment 97 report",
        "",
        "## Decision-rule lines",
        *[f"- {s}" for s in decision_lines],
        "",
    ]
    for cls in CLASSES:
        lines += [
            f"## Coverage, {cls}, 12-36 h",
            "",
            "| arm | level | rows | coverage (95% interval) | exceedance days | pinball |",
            "|---|---|---|---|---|---|",
        ]
        sub = cov.filter(
            (pl.col("class") == cls)
            & (pl.col("band") == BANDS[DECISION_BAND])
            & pl.col("level").is_in(REPORTED_LEVELS)
        )
        for r in sub.sort("level", "arm").iter_rows(named=True):
            lines.append(
                f"| {r['arm']} | {r['level']} | {r['n_rows']} | {r['coverage']:.4f} "
                f"({r['coverage_low']:.4f} to {r['coverage_high']:.4f}) "
                f"| {r['exceedance_days']} | {r['pinball']:.5f} |"
            )
        lines.append("")
    scores = tables["scores"]
    lines += ["## Scores (all leads and by band)", ""]
    for r in scores.sort("class", "band", "score", "arm").iter_rows(named=True):
        lines.append(
            f"- {r['class']} {r['band']} {r['score']} {r['arm']}: {r['value']:.5f} "
            f"diff {r['diff_vs_a0']:+.5f} ({r['diff_low']:+.5f} to {r['diff_high']:+.5f})"
        )
    lines += ["", "## Curtailment (node-only, headline price GBP 2,000/MWh)", ""]
    for fraction, block in curtailment.items():
        lines += [
            f"### scaled limit f={fraction}",
            "",
            str(block["census"]),
            "",
            str(block["intervals"]),
            "",
        ]
        lines += [
            str(
                block["per_arm"]
                .filter(pl.col("is_headline_price"))
                .select(
                    "arm",
                    "n_node_half_hours",
                    "needed_mwh",
                    "planned_mwh",
                    "curtailed_mwh",
                    "breach_mwh",
                    "energy_proxy_gbp",
                )
            ),
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
