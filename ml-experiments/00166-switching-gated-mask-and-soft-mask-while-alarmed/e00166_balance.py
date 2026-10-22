"""The balance score of every real pair alarm of experiment 156, at its chosen window and k.

Experiment 156's stored pair-alarm tables do not keep the partner steps, so this script re-runs
the pair alarm's partner search with the same inputs (window 8 half-hours, k 3.75 noise units) and
stores, for each alarm, the alarm's step, the summed opposite step at the metered neighbours, and
the balance score. The balance score is the smaller of `r` and `1 / r`, where `r` is minus the
summed opposite step over the alarm's own step, so 1 means the two steps cancel exactly and values
near 0 mean one step dwarfs the other. Experiment 156 kept alarms with `r` from 0.4 to 2.5, so the
score here lies in 0.4 to 1. Nothing here reads a forecast score.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00166_balance.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00166_common import E156_DIR, WORK_DIR
from e00166_d_common import ADJACENCY
from e00166_d_detectors import DetectorInputs, local_peaks
from e00166_d_variants import load_both, scales_for, variant_inputs

WINDOW: Final[int] = 8
K: Final[float] = 3.75
VETO_MIN_NEIGHBOURS: Final[int] = 2


def balance_table(*, inputs: DetectorInputs, k: float) -> pl.DataFrame:
    """Experiment 156's pair alarms with the summed opposite step and the balance score added."""
    half = inputs.window // 2
    rows: list[dict] = []
    for series, shift in inputs.shift.items():
        z = shift / inputs.scale[series]
        neighbours = [n for n in ADJACENCY.get(series, ()) if n in inputs.shift]
        for stamp in local_peaks(z=z, threshold=k, window=half):
            own = shift[stamp]
            sign = np.sign(own)
            partners: dict[int, float] = {}
            same = 0
            for n in neighbours:
                segment = inputs.shift[n][max(stamp - 1, 0) : stamp + 2] / inputs.scale[n]
                if not np.isfinite(segment).any():
                    continue
                if (-sign * segment).max() >= k / 2:
                    j = int(np.nanargmax(-sign * segment))
                    partners[n] = float(inputs.shift[n][max(stamp - 1, 0) + j])
                elif (sign * segment).max() >= k / 2:
                    same += 1
            if not partners or same >= min(VETO_MIN_NEIGHBOURS, len(neighbours)):
                continue
            ratio = -float(sum(partners.values())) / own
            if not 0.4 <= ratio <= 2.5:
                continue
            rows.append(
                {
                    "series": series,
                    "stamp": int(stamp),
                    "step_mw": float(own),
                    "partner_sum_mw": float(sum(partners.values())),
                    "ratio": float(ratio),
                    "balance_score": float(min(ratio, 1.0 / ratio)),
                    "n_partners": len(partners),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "series": pl.Int64,
            "stamp": pl.Int64,
            "step_mw": pl.Float64,
            "partner_sum_mw": pl.Float64,
            "ratio": pl.Float64,
            "balance_score": pl.Float64,
            "n_partners": pl.Int64,
        },
    )


def main() -> None:
    """Write `balance.parquet` and check it against experiment 156's stored alarm table."""
    window = int(json.loads((E156_DIR / "detect" / "headline.json").read_text())["window"])
    k = float(json.loads((E156_DIR / "detect" / "chosen_k.json").read_text())[str(window)])
    assert (window, k) == (WINDOW, K), (window, k)
    panel, offset_panel = load_both()
    scale = scales_for(
        variant="X", window=window, residual=panel["residual"], offset=offset_panel["residual"]
    )
    inputs = variant_inputs(
        variant="X",
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scale=scale,
        window=window,
    )
    table = balance_table(inputs=inputs, k=k)
    stored = pl.read_parquet(E156_DIR / "detect" / f"real_W{window}_alarms.parquet")
    keys = ["series", "stamp"]
    assert table.height == stored.height, (table.height, stored.height)
    assert table.join(stored, on=keys, how="anti").is_empty(), "alarms differ from experiment 156"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(WORK_DIR / "balance.parquet")
    print(table.height, "alarms")
    print(table["balance_score"].quantile(0.1), table["balance_score"].quantile(0.5))
    for cut in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(cut, int((table["balance_score"] >= cut).sum()))


if __name__ == "__main__":
    main()
