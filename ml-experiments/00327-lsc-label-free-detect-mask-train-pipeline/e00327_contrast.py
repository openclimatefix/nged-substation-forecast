# ruff: noqa: E501
"""Absolute scores and paired contrasts on the real record, from `real_units.parquet`.

Intervals resample whole calendar months (ordinary rows) or whole labelled shifts (after-shift
rows), 20,000 draws. Contrasts are arm minus arm C; positive means the arm is worse.

    uv run --no-sync python e00327_contrast.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
N_DRAWS: Final[int] = 20_000
SEED: Final[int] = 327
METRICS: Final[tuple[str, ...]] = ("qs", "pin95", "pin99", "pin01", "mae", "below01", "above99")
LEVEL: Final[float] = 0.0125
"""Two-sided tail for the 98.75% interval (Bonferroni over four contrasts)."""


def per_unit(*, table: pl.DataFrame, arm: str, group: str, winter: bool | None) -> pl.DataFrame:
    """Sums of n and each metric per unit, pooled over series."""
    t = table.filter((pl.col("arm") == arm) & (pl.col("group") == group))
    if winter is not None:
        t = t.filter(pl.col("winter") == winter)
    return (
        t.group_by("unit").agg(pl.col("n").sum(), *[pl.col(m).sum() for m in METRICS]).sort("unit")
    )


def main() -> None:
    """Print and store absolute scores and contrasts, with and without the short-history series 7."""
    full = pl.read_parquet(WORK / "results" / "real_units.parquet")
    for label, table in (("all15", full), ("without_D9", full.filter(pl.col("series") != 7))):
        print(f"=== {label} ===")
        run(table=table, label=label)
    per = (
        full.filter(pl.col("group") == "ordinary")
        .group_by("series", "arm")
        .agg(pl.col("n").sum(), pl.col("qs").sum(), pl.col("pin95").sum())
        .with_columns(qs=pl.col("qs") / pl.col("n"), pin95=pl.col("pin95") / pl.col("n"))
        .select("series", "arm", "qs")
        .pivot(on="arm", index="series", values="qs")
        .sort("series")
    )
    print("ordinary rows, quantile score per series")
    print(per)


def run(*, table: pl.DataFrame, label: str) -> None:
    """Contrasts for one set of series."""
    rng = np.random.default_rng(SEED)
    arms = sorted(table["arm"].unique().to_list())
    out: dict = {"absolute": {}, "contrast": {}}
    for group in ("ordinary", "after48", "after7d", "all_rows"):
        for winter in (None, True, False):
            key = f"{group}|winter={winter}"
            base = per_unit(table=table, arm="C", group=group, winter=winter)
            if base.height == 0:
                continue
            n = base["n"].to_numpy()
            idx = rng.integers(0, base.height, size=(N_DRAWS, base.height))
            out["absolute"][key] = {"units": base.height, "rows": float(n.sum())}
            for arm in arms:
                cur = per_unit(table=table, arm=arm, group=group, winter=winter)
                assert np.array_equal(cur["unit"].to_numpy(), base["unit"].to_numpy()), (arm, key)
                out["absolute"][key][arm] = {m: float(cur[m].sum() / n.sum()) for m in METRICS}
                if arm == "C":
                    continue
                res = {}
                for m in METRICS:
                    d = cur[m].to_numpy() - base[m].to_numpy()
                    draws = d[idx].sum(axis=1) / n[idx].sum(axis=1)
                    res[m] = {
                        "diff": float(d.sum() / n.sum()),
                        "lo95": float(np.quantile(draws, 0.025)),
                        "hi95": float(np.quantile(draws, 0.975)),
                        "lo98": float(np.quantile(draws, LEVEL / 2 * 1.0)),
                        "hi98": float(np.quantile(draws, 1 - LEVEL / 2)),
                    }
                out["contrast"][f"{arm}-C|{key}"] = res
    (WORK / "results" / f"real_contrasts_{label}.json").write_text(json.dumps(out, indent=1))
    for key, v in out["absolute"].items():
        print(key, f"units {v['units']} rows {v['rows']:.0f}")
        for arm in arms:
            a = v[arm]
            print(
                f"   {arm:3s} qs {a['qs']:.4f} pin95 {a['pin95']:.4f} pin99 {a['pin99']:.4f} mae {a['mae']:.4f} below01 {a['below01']:.4f} above99 {a['above99']:.4f}"
            )
    for key, res in out["contrast"].items():
        c = res["qs"]
        p = res["pin95"]
        print(
            f"{key}: qs {c['diff']:+.4f} ({c['lo95']:+.4f} to {c['hi95']:+.4f}); pin95 {p['diff']:+.4f} ({p['lo95']:+.4f} to {p['hi95']:+.4f}) [98.75%: {p['lo98']:+.4f} to {p['hi98']:+.4f}]"
        )


if __name__ == "__main__":
    main()
