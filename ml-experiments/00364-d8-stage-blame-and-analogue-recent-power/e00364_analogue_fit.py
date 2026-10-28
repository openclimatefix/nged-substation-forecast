# ruff: noqa: E501
"""Refit experiment 87's analogue member (AN) at D8 with and without recent power.

AN is the monthly XGBoost member with experiment 87's origin lags (the newest usable value, the
value one day earlier and the value one week earlier than the newest usable half-hour) and the mean
and median of the six weekly analogues. This script calls experiment 87's fitter and changes only
which of those columns carry power from just before the forecast origin:

| Arm | Change |
|---|---|
| `control` | none (reproduces the stored AN) |
| `nolag` | the three origin lags are set to NaN in training and prediction |
| `norecent` | as `nolag`, and a weekly analogue counts only when it is at least 7 days before the origin |

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00364-*/e00364_analogue_fit.py --arm nolag
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fit  # ty: ignore[unresolved-import]
import polars as pl

ARMS: Final[tuple[str, ...]] = ("control", "nolag", "norecent")
SITE: Final[str] = "D8"
OUT_ROOT: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/"
    "00364-d8-stage-blame-and-analogue-recent-power/work/analogue_preds"
)
NO_RECENT_AGE: Final[int] = 7 * common.STEPS_PER_DAY
"""Half-hours: with `norecent`, power newer than this before the origin is never used."""


def install(*, arm: str) -> None:
    """Patch experiment 87's fitter so the arm's recent-power columns are removed."""
    if arm == "control":
        return
    original = fit.add_lags

    def add_lags_without_origin_lags(
        *, frame: pl.DataFrame, power: object, a7: bool
    ) -> pl.DataFrame:
        out = original(frame=frame, power=power, a7=a7)
        return out.with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(name) for name in common.LAG_COLUMNS]
        )

    fit.add_lags = add_lags_without_origin_lags
    if arm == "norecent":
        common.NEWEST_AGE = NO_RECENT_AGE


def main() -> None:
    """Fit the arm's 12 monthly refits at D8 and write one file per refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cores", type=int, default=6)
    args = parser.parse_args()
    install(arm=args.arm)
    config = fit.CONFIGS["an"]
    jobs = fit.selected_jobs(config=config, regime="monthly", origins=common.all_origins())
    started = time.monotonic()
    for refit, origins in sorted(jobs.items()):
        lines = fit.run_job(
            config_name="an",
            refit=refit,
            origins=origins,
            sites=[SITE],
            offset=False,
            out_root=OUT_ROOT / args.arm,
            rounds=config.rounds,
            nthread=args.cores,
            seed=common.SEED,
            workers=1,
            log_hours=False,
        )
        print(*lines, f"{time.monotonic() - started:.0f} s elapsed", flush=True)


if __name__ == "__main__":
    main()
