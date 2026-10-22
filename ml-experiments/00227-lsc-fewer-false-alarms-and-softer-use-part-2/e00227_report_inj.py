# ruff: noqa: PTH207
"""Part (b) on the injected worlds: share of the perfect-knowledge gain and harm per false alarm.

Reads the `injected/*.parquet` files of `e00227_injected.py`. Writes `part_b_injected.parquet`.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_report_inj.py
"""

from __future__ import annotations

import glob
import json
from typing import Final

import numpy as np
import polars as pl
from e00227_common import WORK_DIR

N_DRAWS: Final[int] = 20_000
PHASES: Final[tuple[str, ...]] = ("BEFORE_ALARM", "DURING", "AFTER")
DESIGNS: Final[tuple[str, ...]] = ("MASK", "SOFT_2", "SOFT_4", "SOFT_16", "HALF", "SUB")
SOURCES: Final[tuple[str, ...]] = ("PERFECT", "DET")


def ratio(
    *, num: np.ndarray, den: np.ndarray, units: np.ndarray, seed: int = 227
) -> tuple[float, float, float]:
    """Ratio of sums with a 95% interval from resampling whole units."""
    labels, inverse = np.unique(units, return_inverse=True)
    n = np.bincount(inverse, weights=np.nan_to_num(num), minlength=len(labels))
    d = np.bincount(inverse, weights=den, minlength=len(labels))
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(N_DRAWS, len(labels)))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = n[picks].sum(axis=1) / d[picks].sum(axis=1)
    return (
        float(n.sum() / d.sum()),
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )


def load() -> pl.DataFrame:
    """Wide per-row scores: one column per arm."""
    key = ["site", "world", "row", "period", "event", "phase", "event_hours", "fraction", "n_false"]
    wides = []
    for path in sorted(glob.glob(str(WORK_DIR / "injected" / "*.parquet"))):
        f = pl.read_parquet(path)
        wides.append(
            f.pivot(on="arm", index=key, values="qs").with_columns(
                pl.col(pl.Float64).cast(pl.Float32)
            )
        )
    return pl.concat(wides, how="diagonal")


def main() -> None:
    """Print the injected-world tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    wide = load()
    print("rows", wide.height, "worlds", wide["world"].n_unique(), "sites", wide["site"].n_unique())
    rows = []
    headline = (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24)
    strata = {"headline (25% or more, over 1 day)": headline, "all transfers": pl.col("event") >= 0}
    for sname, cond in strata.items():
        for phase in PHASES:
            f = wide.filter((pl.col("phase") == phase) & cond)
            if not f.height:
                continue
            units = (f["world"].cast(pl.Int64) * 1_000_000 + f["event"]).to_numpy()
            k0 = f["K0|NONE"].to_numpy().astype(float)
            perfect = k0 - f["PERFECT|MASK"].to_numpy().astype(float)
            for source in SOURCES:
                for design in DESIGNS:
                    arm = f"{source}|{design}"
                    gain = k0 - f[arm].to_numpy().astype(float)
                    g = ratio(num=gain, den=np.ones(len(gain)), units=units)
                    n_units = len(np.unique(units))
                    rows.append(
                        {
                            "stratum": sname,
                            "phase": phase,
                            "source": source,
                            "design": design,
                            "kind": "gain",
                            "est": g[0],
                            "lo": g[1],
                            "hi": g[2],
                            "n_rows": f.height,
                            "n_units": n_units,
                        }
                    )
            pg = ratio(num=perfect, den=np.ones(len(perfect)), units=units)
            rows.append(
                {
                    "stratum": sname,
                    "phase": phase,
                    "source": "PERFECT",
                    "design": "MASK",
                    "kind": "reference_gain",
                    "est": pg[0],
                    "lo": pg[1],
                    "hi": pg[2],
                    "n_rows": f.height,
                    "n_units": len(np.unique(units)),
                }
            )
    # share of DET|MASK's AFTER gain
    # ordinary rows near false alarms: harm per false alarm
    o = wide.filter(pl.col("phase") == "ORDINARY")
    units = (
        o["world"].cast(pl.Int64) * 100 + o["site"].str.replace("D", "").cast(pl.Int64)
    ).to_numpy()
    first = (
        o.group_by("world", "site")
        .agg(pl.col("n_false").first())
        .with_columns(
            unit=pl.col("world").cast(pl.Int64) * 100
            + pl.col("site").str.replace("D", "").cast(pl.Int64)
        )
    )
    n_false_of = dict(zip(first["unit"].to_list(), first["n_false"].to_list(), strict=True))
    labels, inverse = np.unique(units, return_inverse=True)
    false_per_unit = np.array([n_false_of[int(u)] for u in labels], dtype=float)
    k0 = o["K0|NONE"].to_numpy().astype(float)
    for source in ("DET",):
        for design in DESIGNS:
            diff = o[f"{source}|{design}"].to_numpy().astype(float) - k0
            num = np.bincount(inverse, weights=np.nan_to_num(diff), minlength=len(labels))
            picks = np.random.default_rng(227).integers(0, len(labels), size=(N_DRAWS, len(labels)))
            with np.errstate(divide="ignore", invalid="ignore"):
                draws = num[picks].sum(axis=1) / false_per_unit[picks].sum(axis=1)
            est = float(num.sum() / false_per_unit.sum())
            rows.append(
                {
                    "stratum": "ordinary near false alarms",
                    "phase": "ORDINARY",
                    "source": source,
                    "design": design,
                    "kind": "harm_per_false_alarm",
                    "est": est,
                    "lo": float(np.nanpercentile(draws, 2.5)),
                    "hi": float(np.nanpercentile(draws, 97.5)),
                    "n_rows": o.height,
                    "n_units": len(labels),
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(WORK_DIR / "part_b_injected.parquet")
    (WORK_DIR / "part_b_injected.json").write_text(json.dumps(table.to_dicts(), indent=1))
    print(table.with_columns(pl.col(pl.Float64).round(4)))


if __name__ == "__main__":
    main()
