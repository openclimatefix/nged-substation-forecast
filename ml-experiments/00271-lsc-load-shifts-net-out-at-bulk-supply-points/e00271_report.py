# ruff: noqa: E501
"""Collect the per-world files into the tables the write-up quotes, and print them all.

Effect of a world on an arm = (avoidable cost in the world - avoidable cost on the real record) /
avoidable cost on the real record, over all scored rows (G0), at 0.5 times each BSP's earlier
maximum export. The interval resamples whole worlds (20,000 draws).

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= uv run --no-sync python e00271_report.py
```
"""

from __future__ import annotations

from typing import Final

import e00271_inputs as inputs
import e00271_score as sc
import numpy as np
import polars as pl

WORLD_DIR = inputs.RESULTS_DIR / "worlds"
N_DRAWS: Final[int] = 20_000
LEVEL: Final[float] = 97.5
"""Bonferroni over the two planned contrasts: each interval is 97.5%."""
SEED: Final[int] = 271
THRESHOLDS: Final[dict[str, float]] = {"within": 1.0, "cross": 5.0}


def load(*, name: str) -> pl.DataFrame:
    """Concatenate one table kind over all worlds, with the world's kind, placement, and index."""
    frames = []
    for path in sorted(WORLD_DIR.glob(f"*_{name}.parquet")):
        stem = path.name.removesuffix(f"_{name}.parquet")
        if stem == "real":
            continue
        kind, placement, index = stem.split("_")
        frames.append(
            pl.read_parquet(path).with_columns(
                kind=pl.lit(kind), placement=pl.lit(placement), world=pl.lit(int(index))
            )
        )
    return pl.concat(frames)


def interval(*, values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and percentile interval of the mean over worlds."""
    draws = rng.choice(values, size=(N_DRAWS, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return float(values.mean()), float(low), float(high)


def effects(*, cost: pl.DataFrame, real: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """The planned effect: avoidable cost in the world against the real record, per arm."""
    reference = real.filter(pl.col("group") == "G0", pl.col("arm") != sc.PERFECT).select(
        "target", "arm", real_gbp="avoidable_gbp"
    )
    g0 = cost.filter(pl.col("group") == "G0", pl.col("arm") != sc.PERFECT).join(
        reference, on=["target", "arm"]
    )
    g0 = g0.with_columns(
        effect_pct=100.0 * (pl.col("avoidable_gbp") - pl.col("real_gbp")) / pl.col("real_gbp")
    )
    rows = []
    for (kind, placement, target, arm), part in g0.group_by("kind", "placement", "target", "arm"):
        mean, low, high = interval(values=part["effect_pct"].to_numpy(), rng=rng)
        rows.append(
            {
                "kind": kind,
                "placement": placement,
                "target": target,
                "arm": arm,
                "n_worlds": part.height,
                "mean_pct": mean,
                "low_pct": low,
                "high_pct": high,
                "max_abs_pct": float(np.abs(part["effect_pct"].to_numpy()).max()),
                "mean_gbp": float((part["avoidable_gbp"] - part["real_gbp"]).mean()),  # ty: ignore[invalid-argument-type]
            }
        )
    return pl.DataFrame(rows).sort("kind", "placement", "target", "arm")


def arm_contrasts(*, cost: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Each arm's avoidable cost minus K0's in the same world and row group (GBP, and % of K0's)."""
    k0 = cost.filter(pl.col("arm") == "K0").select(
        "kind", "placement", "world", "target", "group", k0_gbp="avoidable_gbp"
    )
    joined = cost.filter(~pl.col("arm").is_in(["K0", sc.PERFECT])).join(
        k0, on=["kind", "placement", "world", "target", "group"]
    )
    joined = joined.with_columns(difference_gbp=pl.col("avoidable_gbp") - pl.col("k0_gbp"))
    rows = []
    for (kind, placement, target, group, arm), part in joined.group_by(
        "kind", "placement", "target", "group", "arm"
    ):
        mean, low, high = interval(values=part["difference_gbp"].to_numpy(), rng=rng)
        scale = float(part["k0_gbp"].mean())  # ty: ignore[invalid-argument-type]
        rows.append(
            {
                "kind": kind,
                "placement": placement,
                "target": target,
                "group": group,
                "arm": arm,
                "n_worlds": part.height,
                "difference_gbp": mean,
                "low_gbp": low,
                "high_gbp": high,
                "k0_gbp": scale,
                "difference_pct_of_k0": 100.0 * mean / scale if scale else float("nan"),
            }
        )
    return pl.DataFrame(rows).sort("kind", "placement", "target", "group", "arm")


def quality_contrasts(*, quality: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Mean per-row MAE of the median, quantile score, and twCRPS: each arm minus K0, by group."""
    per_row = quality.filter(pl.col("n") > 0).with_columns(
        mae=pl.col("abs_median") / pl.col("n"),
        qs=pl.col("qs") / pl.col("n"),
        tw=pl.col("tw") / pl.col("n"),
        bias_median=pl.col("err_median") / pl.col("n"),
        bias_mean=pl.col("err_mean") / pl.col("n"),
    )
    k0 = per_row.filter(pl.col("arm") == "K0").select(
        "kind",
        "placement",
        "world",
        "target",
        "group",
        k0_mae="mae",
        k0_qs="qs",
        k0_tw="tw",
    )
    joined = per_row.filter(pl.col("arm") != "K0").join(
        k0, on=["kind", "placement", "world", "target", "group"]
    )
    rows = []
    for (kind, placement, target, group, arm), part in joined.group_by(
        "kind", "placement", "target", "group", "arm"
    ):
        row = {
            "kind": kind,
            "placement": placement,
            "target": target,
            "group": group,
            "arm": arm,
            "n_worlds": part.height,
            "mean_rows": float(part["n"].mean()),  # ty: ignore[invalid-argument-type]
        }
        for metric in ("mae", "qs", "tw"):
            diff = (part[metric] - part[f"k0_{metric}"]).to_numpy()
            mean, low, high = interval(values=diff, rng=rng)
            row |= {
                f"{metric}_diff": mean,
                f"{metric}_low": low,
                f"{metric}_high": high,
                f"k0_{metric}": float(part[f"k0_{metric}"].mean()),  # ty: ignore[invalid-argument-type]
            }
        rows.append(row)
    return pl.DataFrame(rows).sort("kind", "placement", "target", "group", "arm")


def main() -> None:
    """Write the tables to the results folder and print them."""
    rng = np.random.default_rng(SEED)
    cost = load(name="cost")
    quality = load(name="quality")
    shifts = load(name="shifts")
    real_cost = pl.read_parquet(WORLD_DIR / "real_cost.parquet")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    out = inputs.RESULTS_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "effects": effects(cost=cost, real=real_cost, rng=rng),
        "arm_contrasts": arm_contrasts(cost=cost, rng=rng),
        "quality_contrasts": quality_contrasts(quality=quality, rng=rng),
        "shift_summary": shifts.group_by("kind", "placement").agg(
            pl.len().alias("n_shifts"),
            pl.col("crosses").mean().alias("share_crossing"),
            pl.col("moved_mw").mean().alias("mean_moved_mw"),
            pl.col("moved_mw").max().alias("max_moved_mw"),
            ((pl.col("end") - pl.col("onset")) / 48).mean().alias("mean_days"),
            pl.col("world").n_unique().alias("n_worlds"),
        ),
        "real_record_cost": real_cost.filter(pl.col("group") == "G0"),
        "group_rows": cost.filter(pl.col("arm") == "K0", pl.col("target") == "observed")
        .group_by("kind", "placement", "group")
        .agg(pl.col("rows").mean().alias("mean_rows")),
    }
    for name, frame in tables.items():
        frame.write_parquet(out / f"{name}.parquet")
        print(f"## {name}\n{frame}\n")
    effects_table = tables["effects"].filter(pl.col("arm") == "K0", pl.col("target") == "observed")
    for row in effects_table.iter_rows(named=True):
        limit = THRESHOLDS[row["kind"]]
        holds = max(abs(row["low_pct"]), abs(row["high_pct"])) < limit
        print(
            f"C {row['kind']} {row['placement']}: {row['mean_pct']:+.3f}% ({row['low_pct']:+.3f} to {row['high_pct']:+.3f}) threshold {limit}% holds={holds}"
        )


if __name__ == "__main__":
    main()
