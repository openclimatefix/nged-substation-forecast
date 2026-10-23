# ruff: noqa: E501
"""Score the bulk supply points (part B) and the load-shift row groups (part C).

Part B scores D13 and D17 with climatology and the manual heuristic (experiment 87's stored
references) and the top-down and bottom-up GAM forecasts (experiment 204). Part C re-scores experiment 163's rows,
which include the logged transfers, for climatology, the manual heuristic, BG2 and the unmasked
GAM-K stack (K0), split into the decision program's row groups. K0 stores five quantiles
(q01, q05, q50, q95, q99), so its CRPS is the stored 11-quantile score and its tail measures are
the pinball losses at q95 and q99; the curtailment metric reads only q95.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00277-*/e00277_score_shifts.py
"""

import time
from typing import Final

import e00277_common as common
import numpy as np
import polars as pl
from e00277_score import (
    LEAD_BANDS,
    METRICS,
    WINTER_MONTHS,
    block_bootstrap,
    cost_summary,
    mean_with_interval,
    run_cost,
)
from studies.node_curtailment import node_curtailment, scaled_limits, table_with_scaled_limits

FIVE: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")
SHIFT_ARMS: Final[tuple[str, ...]] = ("K0", "bg2", "manual_heuristic", "climatology")
SHIFT_METRICS: Final[tuple[str, ...]] = ("qs", "mae", "pin95", "pin99")
SHIFT_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("bg2", "K0"),
    ("manual_heuristic", "K0"),
    ("climatology", "K0"),
    ("climatology", "bg2"),
)
FRACTIONS_C: Final[tuple[float, ...]] = (0.5, 0.75)
DURATION_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 6 h", 0.0, 6.0),
    ("6-24 h", 6.0, 24.0),
    ("over 1 day", 24.0, 1.0e9),
)


def pinball_at(*, actual: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Return the pinball loss at one level, per row."""
    error = actual - q
    return np.maximum(level * error, (level - 1.0) * error)


def part_b(*, rng: np.random.Generator, capacity: dict[int, float]) -> dict:
    """Score the two bulk supply points: climatology, manual heuristic, top-down, bottom-up."""
    out: dict = {}
    table = common.node_table(sites=common.BSPS)
    flows = common.read_flows(series=list(common.BSPS.values()))
    edges_by_series = common.thresholds(sites=common.BSPS, capacity=capacity)
    frames_all: dict[str, list[pl.DataFrame]] = {
        a: [] for a in ("td", "bu", "climatology", "manual_heuristic")
    }
    for bsp, ts in common.BSPS.items():
        td = pl.read_parquet(common.E161_PREDS / f"td_scored_{bsp}.parquet").filter(
            pl.col("seed") == 0
        )
        bu = pl.read_parquet(common.E204_PREDS / f"bu_d_scored_{bsp}.parquet").filter(
            pl.col("seed") == 0
        )
        refs = {
            name: pl.read_parquet(
                common.E87_PREDICTIONS / f"ref_{name}" / "predictions.parquet",
                columns=[*common.KEYS, "seed", "actual", *common.QUANTILES],
            )
            .filter(pl.col("time_series_id") == ts, pl.col("seed") == 0)
            .with_columns([(pl.col(c) * 100.0).alias(c) for c in ("actual", *common.QUANTILES)])
            for name in ("climatology", "manual_heuristic")
        }
        key_frames = {"td": td, "bu": bu, **refs}
        shared = td.select(common.KEYS)
        for frame in key_frames.values():
            shared = shared.join(
                frame.select(common.KEYS).with_columns(
                    pl.col("time_series_id").cast(shared.schema["time_series_id"])
                ),
                on=common.KEYS,
                how="inner",
            )
        shared = shared.unique().sort(common.KEYS)
        for name, frame in key_frames.items():
            aligned = (
                shared.join(
                    frame.with_columns(
                        pl.col("time_series_id").cast(shared.schema["time_series_id"])
                    ),
                    on=common.KEYS,
                    how="left",
                )
                .sort(common.KEYS)
                .with_columns(site=pl.lit(bsp), seed=pl.lit(0, dtype=pl.Int8))
            )
            frames_all[name].append(aligned)
    arrays = {a: pl.concat(fs) for a, fs in frames_all.items()}
    first = arrays["td"]
    actual = first["actual"].to_numpy().astype(np.float64)
    series = first["time_series_id"].to_numpy()
    sites = first["site"].to_numpy()
    edges = {
        label: np.array([edges_by_series[int(s)][label] for s in series]) for label in ("95", "99")
    }
    init_month = first["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    per_row = {}
    for arm, frame in arrays.items():
        q = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
        np.testing.assert_allclose(
            frame["actual"].to_numpy(),
            actual,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"{arm} actual differs",
        )
        mean = frame["mean"].to_numpy().astype(np.float64) if "mean" in frame.columns else None
        per_row[arm] = common.per_row_scores(actual=actual, q=q, mean=mean, edges=edges)
    groups = {"both BSPs": np.ones(len(actual), dtype=bool), **{b: sites == b for b in common.BSPS}}
    metrics = {
        arm: {
            g: {
                m: mean_with_interval(
                    values=per_row[arm][m][mask], labels=init_month[mask], rng=rng
                )
                for m in METRICS
            }
            for g, mask in groups.items()
        }
        for arm in arrays
    }
    out["metrics"] = metrics
    out["n_rows"] = {b: int((sites == b).sum()) for b in common.BSPS}
    out["children"] = {k: list(v) for k, v in common.BSP_CHILDREN.items()}
    frames = {
        arm: common.quantile_frame(
            rows=frame.select(*common.KEYS, "seed", "actual"),
            quantiles=frame.select(list(common.QUANTILES)).to_numpy(),
            capacity=capacity,
        )
        for arm, frame in arrays.items()
    }
    frames["perfect"] = frames["td"].with_columns(
        *[pl.col("actual").alias(name) for name in common.QUANTILES]
    )
    cost = {}
    for fraction in common.FRACTIONS:
        arrays_c, _, scaled = run_cost(table=table, flows=flows, frames=frames, fraction=fraction)
        cost[str(fraction)] = {
            **cost_summary_b(arrays=arrays_c, rng=rng),
            "limits_mw": scaled.rename({"export_limit_mw": "limit_mw"})
            .select("node_id", "limit_mw")
            .to_dicts(),
        }
        names = {v: k for k, v in common.BSPS.items()}
        perfect = (
            frames["perfect"]
            .with_columns(
                lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
            )
            .filter(pl.col("lead") >= 12, pl.col("lead") < 36)
            .with_columns(
                node_id=pl.col("time_series_id").replace_strict(names, return_dtype=pl.String)
            )
        )
        joined = perfect.join(scaled, on="node_id")
        cost[str(fraction)]["census"] = (
            joined.group_by("node_id")
            .agg(
                n_half_hours=pl.len(),
                n_above_limit=(pl.col("actual") > pl.col("export_limit_mw")).sum(),
            )
            .sort("node_id")
            .to_dicts()
        )
    out["cost"] = cost
    return out


def cost_summary_b(*, arrays: dict, rng: np.random.Generator) -> dict:
    """Return the cost summary with the bulk-supply-point contrast pairs."""
    summary = cost_summary(arrays=arrays, rng=rng)
    counts = arrays["perfect"]["counts"]
    for worse, better in (
        ("td", "bu"),
        ("climatology", "td"),
        ("manual_heuristic", "td"),
        ("climatology", "manual_heuristic"),
    ):
        diff = arrays[worse]["sums"] - arrays[better]["sums"]
        summary["contrasts"][f"{worse}-{better}"] = block_bootstrap(
            sums=diff, counts=counts, rng=rng, ratio=False
        )
    return summary


def read_k0_rows() -> pl.DataFrame:
    """Return experiment 163's K0 rows with their group memberships (scored year only)."""
    pieces = []
    for site, ts in common.PRIMARIES.items():
        k0 = (
            pl.read_parquet(common.E163_DIR / f"rows_{site}.parquet")
            .filter(pl.col("arm") == "K0")
            .with_columns(
                pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
                pl.col("valid_time").dt.replace_time_zone("UTC"),
                time_series_id=pl.lit(ts, dtype=pl.Int32),
            )
            .filter(pl.col("valid_time") >= common.SCORED_START)
        )
        quiet = pl.read_parquet(common.E163_DIR / f"quiet_{site}.parquet").with_columns(
            set=pl.lit("QUIET"), card=pl.lit("none")
        )
        sets = pl.read_parquet(common.E163_DIR / f"rowsets_{site}.parquet")
        members = pl.concat(
            [quiet.select("set", "card", "site", "row"), sets.select("set", "card", "site", "row")]
        )
        events = pl.read_parquet(common.E163_DIR / f"events_{site}.parquet").select(
            "card", duration_h=(pl.col("end") - pl.col("onset")) / 2.0
        )
        pieces.append(
            members.join(k0, on=["site", "row"], how="inner").join(events, on="card", how="left")
        )
    return pl.concat(pieces)


def add_other_arms(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Join BG2, the manual heuristic and climatology onto the K0 rows; drop rows any arm lacks."""
    ids = list(common.PRIMARIES.values())
    keys = common.KEYS
    sources = {
        "bg2": common.E87_PREDICTIONS / "p_s6_bg2" / "predictions.parquet",
        "manual_heuristic": common.E87_PREDICTIONS / "ref_manual_heuristic" / "predictions.parquet",
        "climatology": common.E87_PREDICTIONS / "ref_climatology" / "predictions.parquet",
    }
    out = rows
    for name, path in sources.items():
        stored = (
            pl.scan_parquet(path)
            .filter(pl.col("time_series_id").is_in(ids), pl.col("seed") == 0)
            .select(
                pl.col("time_series_id").cast(pl.Int32),
                "power_fcst_init_time",
                "valid_time",
                *[(pl.col(q) * 100.0).alias(f"{name}_{q}") for q in common.QUANTILES],
            )
            .collect()
        )
        out = out.join(stored, on=keys, how="inner")
    return out


def row_scores(*, rows: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Return per-row `qs`, `mae`, `pin95`, `pin99` for each arm on the joined rows (points)."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    out: dict[str, dict[str, np.ndarray]] = {}
    q50 = rows["q50"].to_numpy().astype(np.float64)
    out["K0"] = {
        "qs": rows["qs"].to_numpy().astype(np.float64),
        "mae": np.abs(actual - q50),
        "pin95": pinball_at(actual=actual, q=rows["q95"].to_numpy().astype(np.float64), level=0.95),
        "pin99": pinball_at(actual=actual, q=rows["q99"].to_numpy().astype(np.float64), level=0.99),
    }
    for arm in ("bg2", "manual_heuristic", "climatology"):
        q = rows.select([f"{arm}_{c}" for c in common.QUANTILES]).to_numpy().astype(np.float64)
        out[arm] = {
            "qs": common.quantile_score(actual=actual, q=q),
            "mae": np.abs(actual - q[:, 5]),
            "pin95": pinball_at(actual=actual, q=q[:, 9], level=0.95),
            "pin99": pinball_at(actual=actual, q=q[:, 10], level=0.99),
        }
    return out


def five_quantile_frame(
    *, rows: pl.DataFrame, arm: str, capacity: dict[int, float]
) -> pl.DataFrame:
    """Return an arm's five-quantile node frame in MW (the curtailment metric reads q95)."""
    prefix = "" if arm == "K0" else f"{arm}_"
    frame = rows.select(
        pl.col("time_series_id").cast(pl.Int64),
        "power_fcst_init_time",
        "valid_time",
        pl.lit(0, dtype=pl.Int64).alias("seed"),
        (pl.col("actual").cast(pl.Float64) / 100.0).alias("actual"),
        *[(pl.col(f"{prefix}{q}").cast(pl.Float64) / 100.0).alias(q) for q in FIVE],
    )
    from studies.node_curtailment import scale_to_megawatts

    return scale_to_megawatts(frame=frame, capacity_mw=capacity)


def group_masks(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the row-group masks (G1 to G5 and the splits of G5) over the joined, deduplicated rows."""
    kind = rows["set"].to_numpy()
    valid_month = rows["valid_time"].dt.month().to_numpy()
    lead = rows["lead_hours"].to_numpy()
    duration = rows["duration_h"].to_numpy()
    site = rows["site"].to_numpy()
    near = np.isin(kind, ["BEFORE", "INSIDE", "AFTER"])
    winter = np.isin(valid_month, list(WINTER_MONTHS))
    masks: dict[str, np.ndarray] = {
        "G1 ordinary rows": kind == "QUIET",
        "G2 before the alarm": kind == "BEFORE",
        "G3 during": kind == "INSIDE",
        "G4 after the end": kind == "AFTER",
        "G5 near (G2 + G3 + G4)": near,
    }
    for name, lo, hi in LEAD_BANDS:
        masks[f"G5 lead {name}"] = near & (lead >= lo) & (lead < hi)
        masks[f"G1 lead {name}"] = (kind == "QUIET") & (lead >= lo) & (lead < hi)
    masks["G5 November-March"] = near & winter
    masks["G5 April-October"] = near & ~winter
    masks["G1 November-March"] = (kind == "QUIET") & winter
    masks["G1 April-October"] = (kind == "QUIET") & ~winter
    for name, lo, hi in DURATION_BINS:
        masks[f"G5 duration {name}"] = near & (duration >= lo) & (duration < hi)
    for s in common.PRIMARIES:
        masks[f"G5 substation {s}"] = near & (site == s)
    return masks


def part_c(*, rng: np.random.Generator, capacity: dict[int, float]) -> dict:  # noqa: PLR0915
    """Score experiment 163's rows by row group."""
    started = time.time()
    rows = add_other_arms(rows=read_k0_rows())
    # a row in two sets (or two transfers) is kept once per set, with its first transfer as block
    rows = rows.sort("set", "card").unique(
        subset=["set", "site", "power_fcst_init_time", "valid_time"],
        keep="first",
        maintain_order=True,
    )
    print(f"part C rows {rows.height} at {time.time() - started:.0f} s", flush=True)
    scores = row_scores(rows=rows)
    masks = group_masks(rows=rows)
    init_month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    card = rows["card"].to_numpy()
    out: dict = {"n_rows": {g: int(m.sum()) for g, m in masks.items()}}
    out["n_cards"] = {g: len(np.unique(card[m])) if g[:2] != "G1" else 0 for g, m in masks.items()}
    # actual of the other sources equals K0's (sanity)
    actual = rows["actual"].to_numpy()
    np.testing.assert_allclose(
        rows["actual"].to_numpy(), actual, err_msg="actual differs between sources"
    )
    out["n_rows_after_join"] = rows.height
    metrics: dict = {}
    contrasts: dict = {}
    for g, mask in masks.items():
        labels = init_month if g.startswith("G1") else card
        metrics[g] = {
            arm: {
                m: mean_with_interval(values=scores[arm][m][mask], labels=labels[mask], rng=rng)
                for m in SHIFT_METRICS
            }
            for arm in SHIFT_ARMS
        }
        contrasts[g] = {
            f"{a}-{b}": {
                m: mean_with_interval(
                    values=scores[a][m][mask] - scores[b][m][mask], labels=labels[mask], rng=rng
                )
                for m in SHIFT_METRICS
            }
            for a, b in SHIFT_PAIRS
        }
        metrics[g]["K0_mean_forecast_mae"] = mean_with_interval(
            values=np.abs(actual - rows["mean_forecast"].to_numpy())[mask],
            labels=labels[mask],
            rng=rng,
        )
    out["metrics"] = metrics
    out["contrasts"] = contrasts
    # curtailment cost by group: one call per transfer (or per month for G1), then resample blocks
    table = common.node_table(sites=common.PRIMARIES)
    flows = common.read_flows(series=list(common.PRIMARIES.values()))
    frames_all = {a: five_quantile_frame(rows=rows, arm=a, capacity=capacity) for a in SHIFT_ARMS}
    set_col = rows["set"].to_numpy()
    cost: dict = {}
    for fraction in FRACTIONS_C:
        scaled = scaled_limits(
            table=table, flows=flows, cutoff=common.SCORED_START, fraction=fraction
        )
        full_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction)
        by_group: dict = {}
        for g in (
            "G1 ordinary rows",
            "G2 before the alarm",
            "G3 during",
            "G4 after the end",
            "G5 near (G2 + G3 + G4)",
        ):
            mask = masks[g]
            labels = init_month[mask] if g.startswith("G1") else card[mask]
            part_rows = rows.filter(pl.Series(mask))
            blocks = np.unique(labels)
            sums = {a: [] for a in (*SHIFT_ARMS, "perfect")}
            counts = []
            events = 0
            for block in blocks:
                sel = labels == block
                sub = {
                    a: f.filter(pl.Series(mask)).filter(pl.Series(sel))
                    for a, f in frames_all.items()
                }
                sub["perfect"] = sub["K0"].with_columns(*[pl.col("actual").alias(q) for q in FIVE])
                result = node_curtailment(
                    table=full_table,
                    node_forecasts=sub,
                    generator_forecasts={},
                    assumptions=_assumptions(),
                )
                per_arm = result.per_arm.filter(pl.col("is_headline_price"))
                for a, totals in sums.items():
                    value = per_arm.filter(pl.col("arm") == a)["energy_proxy_gbp"]
                    totals.append(float(value[0]) if len(value) else 0.0)
                n_half_hours = per_arm["n_node_half_hours"]
                counts.append(float(n_half_hours[0]) if len(n_half_hours) else 0.0)
                events += (
                    int(result.event_census["n_exceedances"].sum())
                    if result.event_census.height
                    else 0
                )
            del part_rows
            arrays = {a: {"sums": np.array(v), "counts": np.array(counts)} for a, v in sums.items()}
            summary = cost_summary(arrays=arrays, rng=rng)
            for worse, better in SHIFT_PAIRS:
                diff = arrays[worse]["sums"] - arrays[better]["sums"]
                summary["contrasts"][f"{worse}-{better}"] = block_bootstrap(
                    sums=diff, counts=arrays["perfect"]["counts"], rng=rng, ratio=False
                )
            summary["n_blocks"] = len(blocks)
            by_group[g] = summary
        cost[str(fraction)] = by_group
        print(f"part C cost f={fraction} at {time.time() - started:.0f} s", flush=True)
    del set_col
    out["cost"] = cost
    return out


def _assumptions():  # noqa: ANN202
    """Return the node-only assumptions at the headline price."""
    from dataclasses import replace

    from studies.node_curtailment import NodeCurtailmentAssumptions

    return replace(
        NodeCurtailmentAssumptions(use_generators=False),
        headline_price_gbp_per_mwh=common.PRICE_GBP_PER_MWH,
    )


def main() -> None:
    """Run parts B and C and write the report."""
    rng = np.random.default_rng(common.EXPERIMENT_ID + 1)
    capacity = common.capacity_mw()
    started = time.time()
    b = part_b(rng=rng, capacity=capacity)
    print(f"part B done at {time.time() - started:.0f} s", flush=True)
    common.write_json_atomic(path=common.OUT_DIR / "report_b.json", content=b)
    c = part_c(rng=rng, capacity=capacity)
    common.write_json_atomic(path=common.OUT_DIR / "report_c.json", content=c)
    print(f"done at {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
