"""Score experiment 197 and print every number the write-up quotes.

Reads the redesigned gate's `work/gate/cells.parquet` (written by `e00197_gate.py`) and, for the
real rows, experiment 190's stored scored-year forecasts of arm D (`d0`) and the shared-level arm
(`d3`, m 5, k 8, z 0.5). Writes `work/report/report.json`.

    uv run --no-sync python ml-experiments/00197-*/e00197_report.py
"""

import json
from typing import Final

import e00197_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
E190_PREDS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00190-dp-kalman-level-adaptation-gated-boost" / "work" / "dp_preds"
)
N_RESAMPLES: Final[int] = 20_000
N_GATE_RESAMPLES: Final[int] = 4_000
PASS_RECOVERY: Final[float] = 0.5
LOWER_BOUND_MIN: Final[float] = 0.25
HARM_LIMIT: Final[float] = 0.005
SELECTION_SIZES: Final[tuple[float, ...]] = (0.05, 0.20)
TEST_SIZE: Final[float] = 0.10
OUT_DIR: Final = common.WORK_DIR / "report"
CELL_KEYS: Final[list[str]] = ["site", "size", "hours", "case"]


def cell_table(*, cells: pl.DataFrame, variant: str, **filters: object) -> pl.DataFrame:
    """Return the per-case summed loss of a variant, restricted by equality filters."""
    frame = cells.filter(pl.col("variant") == variant)
    for column, value in filters.items():
        frame = frame.filter(pl.col(column) == value)
    return (
        frame.group_by(CELL_KEYS)
        .agg(loss=pl.col("loss_sum").sum(), n=pl.col("n").sum())
        .sort(CELL_KEYS)
    )


def recovery(
    *, cells: pl.DataFrame, arm: str, resample: bool = False, seed: int = 197, **filters: object
) -> dict:
    """Return the recovery of an arm's excess loss, pooled over the selected pairs.

    Recovery is one minus the arm's excess over the clean A0 loss, divided by A0's excess over the
    clean A0 loss. The interval resamples whole cases (a case is one injected window).

    Args:
        cells: The aggregated gate rows.
        arm: The variant's name.
        resample: Whether to add a 95% interval.
        seed: The bootstrap's seed.
        **filters: Equality filters on `size`, `hours`, `bin`, and `read_class`.

    Returns:
        The recovery, its interval, the pairs and cases, and A0's mean excess per pair.
    """
    clean = cell_table(cells=cells, variant="clean_a0", **filters)
    zero = cell_table(cells=cells, variant="a0", **filters)
    other = cell_table(cells=cells, variant=arm, **filters)
    keys = (
        zero.select(CELL_KEYS)
        .join(clean.select(CELL_KEYS), on=CELL_KEYS)
        .join(other.select(CELL_KEYS), on=CELL_KEYS)
    )
    if keys.is_empty():
        return {"recovery": None}
    lc, l0, la = (t.join(keys, on=CELL_KEYS)["loss"].to_numpy() for t in (clean, zero, other))
    n = zero.join(keys, on=CELL_KEYS)["n"].to_numpy()
    excess0 = float(l0.sum() - lc.sum())
    if excess0 <= 0.0:
        return {"recovery": None, "pairs": int(n.sum()), "cases": len(n)}
    point = 1.0 - (la.sum() - lc.sum()) / excess0
    out = {
        "recovery": float(point),
        "pairs": int(n.sum()),
        "cases": len(n),
        "excess_a0_per_pair": excess0 / float(n.sum()),
    }
    if resample:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(N_GATE_RESAMPLES):
            pick = rng.integers(0, len(n), len(n))
            e0 = l0[pick].sum() - lc[pick].sum()
            if e0 > 0.0:
                draws.append(1.0 - (la[pick].sum() - lc[pick].sum()) / e0)
        out["lower"], out["upper"] = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return out


def arms_of(*, cells: pl.DataFrame, prefix: str) -> list[str]:
    """Return the grid variants of A1 (`a1_m...`) or A3 (`a3_m...`)."""
    return sorted(v for v in cells["variant"].unique().to_list() if v.startswith(f"{prefix}_m"))


def select_setting(*, cells: pl.DataFrame, prefix: str, read_class: str) -> dict:
    """Return the grid cell with the highest recovery at the selection sizes, and the whole grid.

    Recovery is pooled over sizes 0.05 and 0.20, every duration, and the pairs of `read_class`.
    """
    selection = cells.filter(pl.col("size").is_in(list(SELECTION_SIZES)))
    scores = {
        arm: recovery(cells=selection, arm=arm, read_class=read_class)["recovery"]
        for arm in arms_of(cells=cells, prefix=prefix)
    }
    best = max(scores, key=lambda k: scores[k])
    return {"selected": best, "scores": scores}


def harm(*, cells: pl.DataFrame, arm: str) -> dict:
    """Return an arm's mean quantile score minus A0's on the left-alone days, with an interval."""
    alone = cells.filter(pl.col("kind") == "alone")
    keys = ["site", "case"]

    def per_case(variant: str) -> pl.DataFrame:
        return (
            alone.filter(pl.col("variant") == variant)
            .group_by(keys)
            .agg(loss=pl.col("loss_sum").sum() / pl.col("n").sum())
            .sort(keys)
        )

    diff = per_case(arm)["loss"].to_numpy() - per_case("a0")["loss"].to_numpy()
    rng = np.random.default_rng(197)
    draws = rng.choice(diff, size=(N_RESAMPLES, len(diff)), replace=True).mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(diff.mean()), "lower": float(lower), "upper": float(upper)}


def gate_report() -> dict:
    """Return the redesigned gate: selection, recovery by read class and hours, and harm."""
    cells = pl.read_parquet(common.WORK_DIR / "gate" / "cells.parquet")
    injected = cells.filter(pl.col("kind") == "injected")
    out: dict = {}
    # Share of pairs by read class, by hours since onset (size does not change the pairs).
    shares = {}
    for hours in sorted(injected["hours"].unique().to_list()):
        part = injected.filter((pl.col("variant") == "a0") & (pl.col("hours") == hours))
        for label in part["bin"].unique().to_list():
            sub = part.filter(pl.col("bin") == label)
            total = int(sub["n"].sum())
            shares[f"{hours}|{label}"] = {
                c: float(sub.filter(pl.col("read_class") == c)["n"].sum()) / total
                for c in ("band_read", "other_read", "unread")
            } | {"pairs": total}
    out["read_class_shares"] = shares
    # Selection (sizes 0.05 and 0.20).
    out["selection_a1"] = select_setting(cells=injected, prefix="a1", read_class="band_read")
    out["selection_a3"] = select_setting(cells=injected, prefix="a3", read_class="other_read")
    a1, a3 = out["selection_a1"]["selected"], out["selection_a3"]["selected"]
    chosen = {"a1": a1, "a3": a3, "a1_e190": "a1_e190", "a3_e190": "a3_e190"}
    test: dict = {}
    for label, arm in chosen.items():
        for hours in (8, 16, 36):
            for read_class in ("band_read", "other_read", "unread"):
                test[f"{label}|{hours}|{read_class}"] = recovery(
                    cells=injected, arm=arm, resample=True,
                    size=TEST_SIZE, hours=hours, read_class=read_class,
                )  # fmt: skip
            test[f"{label}|{hours}|all"] = {
                "recovery_original_gate": recovery(
                    cells=injected, arm=arm, resample=True, size=TEST_SIZE, hours=hours
                )
            }
    out["test"] = test
    out["chosen"] = chosen
    # Curve by hours since onset, per read class, for the chosen arms at 0.10 and 16 hours.
    curve = {}
    for label, arm in chosen.items():
        for hours in (16, 36):
            for read_class in ("band_read", "other_read"):
                for bin_label in ("0-3.5 h", "3.5-6 h", "6-12 h", "12-24 h", "24-36 h"):
                    curve[f"{label}|{hours}|{read_class}|{bin_label}"] = recovery(
                        cells=injected, arm=arm, size=TEST_SIZE, hours=hours,
                        read_class=read_class, bin=bin_label,
                    )  # fmt: skip
    out["curve"] = curve
    # Whole grid at the test size, band-read and other-read pairs, 16 hours.
    grid = {}
    for arm in [*arms_of(cells=cells, prefix="a1"), *arms_of(cells=cells, prefix="a3")]:
        for hours in (16, 36):
            for read_class in ("band_read", "other_read"):
                grid[f"{arm}|{hours}|{read_class}"] = recovery(
                    cells=injected, arm=arm, size=TEST_SIZE, hours=hours, read_class=read_class
                )
    out["grid"] = grid
    # Harm on the left-alone days.
    arms = [v for v in cells["variant"].unique().to_list() if v not in ("a0", "clean_a0")]
    out["harm"] = {arm: harm(cells=cells, arm=arm) for arm in arms}
    return out


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return an arm's scored-year rows at the five decision Primaries (experiment 190's files)."""
    return pl.concat(
        [
            pl.read_parquet(E190_PREDS / f"{arm}_scored_{site}.parquet").with_columns(
                site=pl.lit(site)
            )
            for site in SITES
        ]
    )


def real_rows() -> dict:
    """Return A3 and A1 minus A0 on the scored year (experiment 190's stored arms)."""
    cache = common.load_cache()
    labels = {cache.sites[s].table.time_series_id: s for s in SITES}
    pieces = {}
    for name, arm in (("d0", "d0"), ("d1", "d1"), ("d3", "d3")):
        frame = load_arm(arm=arm).filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= 168.0)
            & (pl.col("lead_hours") >= 0.5)
        )
        pieces[name] = frame
    shared = pieces["d0"].select(KEYS).unique()
    for frame in pieces.values():
        shared = shared.join(frame.select(KEYS).unique(), on=KEYS, how="inner")
    losses = {}
    for name, piece in pieces.items():
        frame = piece.join(shared, on=KEYS, how="inner").sort(KEYS)
        actual = frame["actual"].to_numpy().astype(np.float64)
        quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
        error = actual[:, None] - quantiles
        qs = 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)
        losses[name] = frame.select(*KEYS, "seed", "lead_hours").with_columns(qs=pl.Series(qs))
    stacked = losses_for_bootstrap(arm_losses=losses).with_columns(
        site=pl.col("site").replace_strict(labels, return_dtype=pl.String)
    )
    out: dict = {"rows": losses["d0"].height}
    out["scores"] = {
        name: float(stacked.filter(pl.col("arm") == name)["qs"].to_numpy().mean())
        for name in losses
    }
    for label, treatment in (("A3 minus A0", "d3"), ("A1 minus A0", "d1")):
        lower, upper = bootstrap_difference_at_level(
            losses=stacked,
            treatment=treatment,
            reference="d0",
            metric="qs",
            level=95.0,
            n_resamples=N_RESAMPLES,
        )
        t = stacked.filter(pl.col("arm") == treatment).group_by("site").agg(t=pl.col("qs").mean())
        r = stacked.filter(pl.col("arm") == "d0").group_by("site").agg(r=pl.col("qs").mean())
        joined = t.join(r, on="site")
        out[label] = {
            "estimate": out["scores"][treatment] - out["scores"]["d0"],
            "lower": float(lower),
            "upper": float(upper),
            "sites_better": int((joined["t"] < joined["r"]).sum()),
            "n_sites": joined.height,
        }
    return out


def main() -> None:
    """Score everything and write the report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"gate": gate_report(), "real_rows": real_rows()}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["real_rows"], indent=1))
    print(json.dumps({k: v for k, v in report["gate"].items() if k != "grid"}, indent=1))


if __name__ == "__main__":
    main()
