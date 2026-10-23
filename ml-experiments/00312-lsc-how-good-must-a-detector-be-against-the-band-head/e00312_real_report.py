# ruff: noqa: E501
"""The real-record scores of K0, GAM_ONLY, and the cross-fitted heads: the numbers the page quotes.

Groups: DURING a logged transfer touching the substation's series, AFTER (48 hours after its end),
and ORDINARY (outside every logged transfer by 7 days; every fourth run, weight 4). Intervals
resample whole logged transfers for DURING and AFTER and calendar months for ORDINARY, 95%,
20,000 draws. The ORDINARY rows overlap the injected worlds' training rows; every head scored a
row it was fitted away from by month parity.

    PYTHONPATH=... uv run --no-sync python e00312_real_report.py
"""

from __future__ import annotations

import json

import e00297_report as r297
import numpy as np
import polars as pl
from e00312_common import REAL_DIR, RESULTS_DIR

ARMS = ("K0", "GAM_ONLY", "HEAD_BASE", "HEAD_PIT")


def main() -> None:
    """Print and store the real-record numbers."""
    frames = []
    for path in sorted(REAL_DIR.glob("D*.parquet")):
        site = path.stem
        long = pl.read_parquet(path)
        wide = long.filter(pl.col("arm") == "K0").select(
            "row", "group", "event", "month", "lead_hours_f", "actual_f", "site"
        ) if "lead_hours_f" in long.columns else long.filter(pl.col("arm") == "K0").select("row", "group", "event", "month", "lead_hours", "actual", "site")
        wide = wide.rename({c: c.removesuffix("_f") for c in wide.columns if c.endswith("_f")})
        for arm in ARMS:
            part = long.filter(pl.col("arm") == arm).select(
                "row", *[pl.col(c).alias(f"{c}__{arm}") for c in ("qs_obs", "tw95_obs", "q01", "q05", "q95", "q99")]
            )
            wide = wide.join(part, on="row", how="left", validate="1:1")
        frames.append(wide.with_columns(site=pl.lit(site)))
    wide = pl.concat(frames)
    group = wide["group"].to_numpy()
    weight = np.where(group == "ORDINARY", r297.ORDINARY_WEIGHT, 1.0)
    unit = np.where(
        group == "ORDINARY",
        -(wide["month"].to_numpy().astype(np.int64) + 1),
        (pl.Series([hash(s) % 1000 for s in wide["site"]]).to_numpy() * 1000 + wide["event"].to_numpy() + 1).astype(np.int64),
    )
    rng = np.random.default_rng(312)
    out: dict = {}
    n_events = (
        wide.filter(pl.col("group").is_in(["DURING", "AFTER"]))
        .select(pl.struct("site", "event").n_unique())
        .item()
    )
    print(f"rows {wide.height}; logged transfers with rows: {n_events}")
    out["n_logged_transfers"] = n_events
    actual = wide["actual"].to_numpy().astype(np.float64)
    table = []
    for arm in ARMS:
        for g in ("DURING", "AFTER", "ORDINARY"):
            m = group == g
            qs = wide[f"qs_obs__{arm}"].to_numpy().astype(np.float64)
            res = {"arm": arm, "group": g, "mean_qs": float(np.sum(qs[m] * weight[m]) / np.sum(weight[m]))}
            for side, ind in (
                ("below_q01", actual < wide[f"q01__{arm}"].to_numpy()),
                ("below_q05", actual < wide[f"q05__{arm}"].to_numpy()),
                ("above_q95", actual > wide[f"q95__{arm}"].to_numpy()),
                ("above_q99", actual > wide[f"q99__{arm}"].to_numpy()),
            ):
                res[side] = float(np.sum(ind[m] * weight[m]) / np.sum(weight[m]))
            table.append(res)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    out["table"] = table
    contrasts = []
    for base, arm in (("K0", "HEAD_BASE"), ("K0", "HEAD_PIT"), ("HEAD_BASE", "HEAD_PIT"), ("K0", "GAM_ONLY")):
        for g in ("DURING", "AFTER", "ORDINARY"):
            m = group == g
            b = wide[f"qs_obs__{base}"].to_numpy().astype(np.float64)
            a = wide[f"qs_obs__{arm}"].to_numpy().astype(np.float64)
            res = r297.interval(gain=(b - a)[m], base=b[m], weight=weight[m], units=unit[m], rng=rng)
            contrasts.append({"base": base, "arm": arm, "group": g, **res})
            r297.show(f"{arm} vs {base} qs {g}", res)
    out["contrasts"] = contrasts
    per_site = []
    for site in sorted(set(wide["site"].to_list())):
        for arm in ("K0", "HEAD_BASE"):
            for g in ("AFTER", "ORDINARY"):
                m = (group == g) & (wide["site"].to_numpy() == site)
                if not m.any():
                    continue
                row = {"site": site, "arm": arm, "group": g}
                for side, ind in (
                    ("below_q01", actual < wide[f"q01__{arm}"].to_numpy()),
                    ("above_q99", actual > wide[f"q99__{arm}"].to_numpy()),
                ):
                    row[side] = float(np.sum(ind[m] * weight[m]) / np.sum(weight[m]))
                per_site.append(row)
                print(row)
    out["per_site"] = per_site
    (RESULTS_DIR / "real.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
