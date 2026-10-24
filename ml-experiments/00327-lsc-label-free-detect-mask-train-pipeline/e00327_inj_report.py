# ruff: noqa: E501
"""Contrasts on the injected worlds against the true NRA series, by row group.

Groups follow experiment 297: G1 ordinary rows, G2 BEFORE_ALARM, G3 DURING, G4 AFTER, G5 every row
near an injected event (those three plus MARGIN), G0 all rows, and DA, DURING and AFTER pooled (the
pre-registered headline). Headline rows have an onset month (events) or target month (ordinary)
that is even. Metrics are in points of effective capacity: the 95% quantile pinball loss against
the true NRA series (`pin95`), experiment 297's threshold-weighted score (`tw95`), the mean quantile
score (`qs`), q01 and q99 pinball and coverage. Intervals resample whole injected events (world,
site, event) for event groups and whole months for ordinary rows, 20,000 draws.

    uv run --no-sync python e00327_inj_report.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
N_DRAWS: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("K0", "PERFECT", "A", "B")
METRICS: Final[tuple[str, ...]] = (
    "pin95",
    "pin99",
    "pin01",
    "tw95",
    "qs",
    "below01",
    "above99",
    "mae",
)
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "G1_ordinary": ("ORDINARY",),
    "G2_before_alarm": ("BEFORE_ALARM",),
    "G3_during": ("DURING",),
    "G4_after": ("AFTER",),
    "G5_near": ("BEFORE_ALARM", "DURING", "AFTER", "MARGIN"),
    "DA_during_after": ("DURING", "AFTER"),
    "G0_all": ("ORDINARY", "BEFORE_ALARM", "DURING", "AFTER", "MARGIN"),
}


def load() -> pl.DataFrame:
    """All scored rows with the metrics added and the resampling unit."""
    t = pl.concat([pl.read_parquet(p) for p in sorted((WORK / "inj").glob("D*_*.parquet"))])
    e95, e99, e01 = (pl.col("actual_nra") - pl.col(c) for c in ("q95", "q99", "q01"))
    return t.with_columns(
        pin95=pl.max_horizontal(0.95 * e95, -0.05 * e95),
        pin99=pl.max_horizontal(0.99 * e99, -0.01 * e99),
        pin01=pl.max_horizontal(0.01 * e01, -0.99 * e01),
        qs=pl.col("qs_nra"),
        tw95=pl.col("tw95_nra"),
        mae=(pl.col("q50") - pl.col("actual_nra")).abs(),
        below01=(pl.col("actual_nra") < pl.col("q01")).cast(pl.Float64),
        above99=(pl.col("actual_nra") > pl.col("q99")).cast(pl.Float64),
        unit=pl.when(pl.col("event") >= 0)
        .then(pl.concat_str(["site", "world", "event"], separator="/"))
        .otherwise(pl.concat_str([pl.lit("m"), "period", "month"], separator="/")),
        parity_month=pl.when(pl.col("event") >= 0)
        .then(pl.col("onset_month") % 100)
        .otherwise(pl.col("month")),
    )


def main() -> None:
    """Print and store absolute scores and contrasts."""
    t = load()
    rng = np.random.default_rng(3271)
    out: dict = {"counts": {}, "absolute": {}, "contrast": {}}
    for parity_name, parity in (("even", 0), ("all", None)):
        sub = t if parity is None else t.filter(pl.col("parity_month") % 2 == parity)
        for gname, phases in GROUPS.items():
            g = sub.filter(pl.col("phase").is_in(phases))
            if gname == "G0_all":
                g = sub
            agg = {}
            for arm in ARMS:
                a = (
                    g.filter(pl.col("arm") == arm)
                    .group_by("unit")
                    .agg(pl.len().alias("n"), *[pl.col(m).sum() for m in METRICS])
                    .sort("unit")
                )
                agg[arm] = a
            n = agg["K0"]["n"].to_numpy().astype(float)
            idx = rng.integers(0, len(n), size=(N_DRAWS, len(n)))
            key = f"{gname}|{parity_name}"
            out["counts"][key] = {"units": len(n), "rows": float(n.sum())}
            out["absolute"][key] = {
                arm: {m: float(agg[arm][m].sum() / n.sum()) for m in METRICS} for arm in ARMS
            }
            for x, y in (("A", "K0"), ("B", "K0"), ("PERFECT", "K0"), ("A", "B")):
                res = {}
                for m in METRICS:
                    d = agg[x][m].to_numpy() - agg[y][m].to_numpy()
                    draws = d[idx].sum(axis=1) / n[idx].sum(axis=1)
                    res[m] = {
                        "diff": float(d.sum() / n.sum()),
                        "lo95": float(np.quantile(draws, 0.025)),
                        "hi95": float(np.quantile(draws, 0.975)),
                        "lo98": float(np.quantile(draws, 0.00625)),
                        "hi98": float(np.quantile(draws, 0.99375)),
                    }
                out["contrast"][f"{x}-{y}|{key}"] = res
    (WORK / "results" / "inj_contrasts.json").write_text(json.dumps(out, indent=1))
    for key, v in out["counts"].items():
        print(key, v)
        for arm in ARMS:
            a = out["absolute"][key][arm]
            print(
                f"   {arm:8s} pin95 {a['pin95']:.4f} tw95 {a['tw95']:.4f} qs {a['qs']:.4f} mae {a['mae']:.4f} below01 {a['below01']:.4f} above99 {a['above99']:.4f}"
            )
    for key, res in out["contrast"].items():
        p, q = res["pin95"], res["qs"]
        print(
            f"{key}: pin95 {p['diff']:+.4f} ({p['lo95']:+.4f} to {p['hi95']:+.4f}) [98.75%: {p['lo98']:+.4f} to {p['hi98']:+.4f}]; qs {q['diff']:+.4f} ({q['lo95']:+.4f} to {q['hi95']:+.4f})"
        )


if __name__ == "__main__":
    main()
