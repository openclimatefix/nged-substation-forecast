# ruff: noqa: E501
"""Assemble the stored forecasts of Primary demand research default 3 for the scored and replicate years.

Default 3 is experiment 251's A+H+S below 168 hours and experiment 322's D1 above. The replicate year
holds forecasts at 168 to 240 hours only. Writes `work/default3_scored.parquet` and
`work/default3_replicate.parquet` under this experiment's data folder.

    PYTHONPATH=<folders of experiments 184, 202, 316, 317, 319, 320, 322> \
        uv run --no-sync python ml-experiments/00335-*/e00335_assemble.py
"""

import json
from pathlib import Path
from typing import Final

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00320_common as c320  # ty: ignore[unresolved-import]
import e00322_common as c322  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 335
SLUG: Final[str] = "primary-demand-default-3-tail-calibration-at-0-5-to-240-hours"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
QUANTILES = c320.QUANTILES
KEYS = ["site", "power_fcst_init_time", "valid_time"]


def main() -> None:
    """Join default 3's quantiles, the actuals, the references, and the group flags, then check the gate."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    stored = (
        pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet").select(KEYS).with_row_index("i251")
    )
    rows = rows.join(stored, on=KEYS, how="left", validate="1:1")
    long_arms = pl.read_parquet(
        c322.ARMS_DIR / "scored_long_lead.parquet",
        columns=[*KEYS, "p_share", "ens_mean_c", *[f"d1_{q}" for q in QUANTILES]],
    )
    rows = rows.join(long_arms, on=KEYS, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    rows = rows.filter(pl.Series(long | rows["i251"].is_not_null().to_numpy()))
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    arm1 = np.load(c320.c319.E251_ARMS / "arm1.npy")
    idx = rows["i251"].fill_null(0).to_numpy()
    short_q = arm1[idx].astype(np.float64)
    long_q = rows.select([f"d1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    d3 = np.where(long[:, None], long_q, short_q)
    actual = rows["actual"].to_numpy().astype(np.float64)
    loss = 2.0 * s317.pin(actual=actual, q=d3).mean(axis=1)
    gate = {
        "default3_0_240": float(loss.mean()),
        "default3_168_240": float(loss[long].mean()),
        "expected_0_240": 2.727,
        "expected_168_240": 3.077,
        "n_rows": rows.height,
    }
    print(json.dumps(gate), flush=True)
    out = rows.select(
        *KEYS,
        "lead_hours",
        "actual",
        "cold_winter_night",
        "sunny_daylight",
        "daylight",
        "p_share",
        "ens_mean_c",
        *[f"mh_{q}" for q in QUANTILES],
        *[f"cl_{q}" for q in QUANTILES],
    ).with_columns(
        *[pl.Series(f"d3_{q}", d3[:, i].astype(np.float32)) for i, q in enumerate(QUANTILES)]
    )
    out.write_parquet(WORK_DIR / "default3_scored.parquet")
    (WORK_DIR / "gate.json").write_text(json.dumps(gate, indent=2))

    rep_chain = s317.s319.load_chain_period(period="replicate") if hasattr(s317, "s319") else None
    if rep_chain is None:
        import e00319_score as s319  # ty: ignore[unresolved-import]

        rep_chain = s319.load_chain_period(period="replicate")
    rep = pl.read_parquet(
        c322.ARMS_DIR / "replicate_long_lead.parquet",
        columns=[*KEYS, "lead_hours", *[f"d1_{q}" for q in QUANTILES]],
    )
    rep = rep_chain.select(*KEYS, "actual").join(rep, on=KEYS, how="inner", validate="1:1")
    rep = rep.rename({f"d1_{q}": f"d3_{q}" for q in QUANTILES})
    rep.write_parquet(WORK_DIR / "default3_replicate.parquet")
    print("replicate rows", rep.height, flush=True)


if __name__ == "__main__":
    main()
