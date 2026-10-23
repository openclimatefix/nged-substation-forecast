"""Write the arm specs of experiment 280 and run one arm's pipeline.

An arm's spec gives, per substation, the solar term of experiment 180's pipeline: `empirical`
(read the shape from the substation's averaged learned table), `level` (a constant that
multiplies the shape), `april_june` (a further factor in April, May, and June), and the export
`clip` at `clip_share` of the held capacity. A substation missing from an arm's spec has the
fleet shape at the pinned capacity with no clip.

| Arm | Shape | Level | April to June factor |
|---|---|---|---|
| `c0` | fleet | 1 | 1 |
| `e0` | empirical | 1 | 1 |
| `e1` | empirical | `shape_swap` | 1 |
| `e2` | empirical | `fitted` | 1 |
| `f1` | fleet | `fleet_fitted` | 1 |
| `s1` | fleet | 1 | calibrated on April to June 2025 |
| `s2` | fleet | 1 | oracle values of experiment 266 (not a forecast) |

An arm name with a suffix `_k5` or `_k9` is the same arm with the 60% clip at D5 or D9 only; it is
run at that substation alone, because the other substations are unchanged.

    uv run --no-sync python ml-experiments/00280-*/e00280_arms.py --write-specs
    uv run --no-sync python ml-experiments/00280-*/e00280_arms.py --run e2_k5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import e00280_common as common

BASE_ARMS: Final[tuple[str, ...]] = ("c0", "e0", "e1", "e2", "f1", "s1", "s2")
CLIP_SITES: Final[dict[str, str]] = {"k5": "D5", "k9": "D9"}
CLIP_SHARE: Final[float] = 0.6
ORACLE_APRIL_JUNE: Final[dict[str, float]] = {"D5": 0.59, "D9": 0.86, "D16": 0.90}
"""Experiment 266's April to June ratios of the capacity the scored year supports to the pinned
capacity (minus 41%, 14%, and 10%). These read the scored year, so arm `s2` is a bound."""
HERE: Final[Path] = Path(__file__).resolve().parent


def base_spec(*, arm: str, site: str, levels: dict) -> dict:
    """Return a base arm's solar-term spec at a substation, empty for the control."""
    if arm == "e0":
        return {"empirical": True}
    if arm == "e1":
        return {"empirical": True, "level": levels[site]["shape_swap"]}
    if arm == "e2":
        return {"empirical": True, "level": levels[site]["fitted"]}
    if arm == "f1":
        return {"level": levels[site]["fleet_fitted"]}
    if arm == "s1":
        return {"april_june": levels[site]["aj_april_june"]}
    if arm == "s2":
        factor = ORACLE_APRIL_JUNE.get(site)
        return {} if factor is None else {"april_june": factor}
    return {}


def all_specs() -> dict[str, dict[str, dict]]:
    """Return every arm's spec by arm name and substation."""
    levels = json.loads((common.WORK_DIR / "levels.json").read_text())
    specs: dict[str, dict[str, dict]] = {}
    for arm in BASE_ARMS:
        specs[arm] = {
            site: spec
            for site in common.SCORED_SITES
            if (spec := base_spec(arm=arm, site=site, levels=levels))
        }
        for suffix, site in CLIP_SITES.items():
            specs[f"{arm}_{suffix}"] = {
                site: {
                    **base_spec(arm=arm, site=site, levels=levels),
                    "clip": True,
                    "clip_share": CLIP_SHARE,
                }
            }
    return specs


def sites_of(*, arm: str) -> list[str]:
    """Return the substations an arm is run at."""
    suffix = arm.rsplit("_", 1)[-1]
    return [CLIP_SITES[suffix]] if suffix in CLIP_SITES else list(common.SCORED_SITES)


def run(*, arm: str, workers: int) -> None:
    """Run the three steps of one arm: the monthly fits, the forecasts, and the corrections."""
    env = {**os.environ, "E00280_ARM": arm}
    sites = sites_of(arm=arm)
    steps = (
        ["e00280_fit.py", "--cadences", "monthly", "--workers", str(workers)],
        ["e00280_forecast.py", "--arms", "a2", "--workers", str(workers)],
        ["e00280s2_run.py", "--workers", str(workers)],
    )
    for step in steps:
        subprocess.run(
            [sys.executable, str(HERE / step[0]), *step[1:], "--sites", *sites],
            env=env,
            check=True,
        )


def main() -> None:
    """Write the specs or run an arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-specs", action="store_true")
    parser.add_argument("--run", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.write_specs:
        common.write_json_atomic(path=common.SPEC_PATH, content=all_specs())
        print(json.dumps(all_specs(), indent=1)[:1500])
    if args.run:
        run(arm=args.run, workers=args.workers)


if __name__ == "__main__":
    main()
