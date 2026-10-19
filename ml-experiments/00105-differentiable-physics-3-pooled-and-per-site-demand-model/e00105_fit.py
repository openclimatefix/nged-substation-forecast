"""Fit the pooled and per-site DP arms on all rows and write their quantiles.

Every arm is screening tier: one frozen fit on data before 2025-07-01 (a 2-year window, or all
years for the `_all` arms), no refits, one seed, and a replica fit 12 months earlier for the spread.
The settings come from `work/tuned.json` (`e00105_tune.py`).

Arms: `p` (pooled, variational, network on), `p_map` (pooled, MAP), `p_nonet` (pooled, no network),
`s` (the same model one MW substation at a time), `p_all` and `s_all` (the same on every usable
year).

    uv run --no-sync python ml-experiments/00105-*/e00105_fit.py [--arms p p_map ...]
"""

import argparse
import json
from typing import Final

import e00105_arms as arms
import e00105_common as common
import torch

ARMS: Final[dict[str, tuple[str, int | None, bool]]] = {
    "p": ("p", common.WINDOW_MONTHS, True),
    "p_map": ("p_map", common.WINDOW_MONTHS, True),
    "p_nonet": ("p_nonet", common.WINDOW_MONTHS, True),
    "s": ("s", common.WINDOW_MONTHS, False),
    "p_all": ("p", None, True),
    "s_all": ("s", None, False),
}
"""For each arm: the model kind, the window in months (None for all years), and whether pooled."""


def run_arm(*, name: str, cache) -> dict[str, float]:  # noqa: ANN001
    """Fit one arm and store its predictions; return its fit record."""
    kind, months, pooled = ARMS[name]
    settings = arms.tuned()
    config = arms.config_for(kind=kind, settings=settings)
    if pooled:
        forecasts, record = arms.arm_forecast(
            cache=cache,
            config=config,
            fit_sites=common.POOL,
            score_sites=common.REPORT_SITES,
            months=months,
            cut=common.FINAL_CUT,
            through=common.SCORING_END,
            save_as=name,
        )
    else:
        forecasts, record = arms.per_site_forecasts(
            cache=cache,
            config=config,
            sites=common.REPORT_SITES,
            months=months,
            cut=common.FINAL_CUT,
            through=common.SCORING_END,
        )
    frame = arms.frames_from(cache=cache, forecasts=forecasts)
    common.store_arm(name=name, frame=frame)
    return record


def main() -> None:
    """Fit the requested arms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    cache = common.load_cache()
    records_path = common.WORK_DIR / "fit_records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    for name in arguments.arms:
        if (common.DATA_DIR / "predictions" / name).exists():
            print(f"{name}: stored already, skipped", flush=True)
            continue
        records[name] = run_arm(name=name, cache=cache)
        print(f"{name}: {records[name]}", flush=True)
        common.write_json(name="fit_records.json", payload=records)


if __name__ == "__main__":
    main()
