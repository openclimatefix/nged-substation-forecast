# ruff: noqa: E501
"""Audit checks for experiment 278: why a retrained head is not better, leave-one-site-out, median minus mean.

(1) `own_once`: the head fitted at 2025-07 with this experiment's pipeline, applied to every row of
the scored year with no retrain; its difference from the quarterly head isolates the effect of
retraining from the effect of this experiment's split rules. (2) The spread of the stack score
across the three seeds of one head. (3) Leave-one-substation-out for C1 and C3. (4) Median minus
mean forecast per substation for arm D at each GAM cadence.

    uv run --no-sync python ml-experiments/00278-*/e00278_checks.py
"""

import json

import numpy as np
import polars as pl

import e00278_common as common  # isort: skip
import e00278_head_score as hs  # isort: skip
import e00278_gam_score as gs  # isort: skip

SEED = 278


def main() -> None:
    """Run the checks and write `work/report/checks.json`."""
    rows = pl.read_parquet(common.E220_ARMS / "rows.parquet")
    base = {
        n: np.load(common.E220_ARMS / f"{n}.npy").astype(np.float64) for n in ("a", "a_s", "a_h")
    }
    edges = common.edges_of(rows=rows)
    actual = rows["actual"].to_numpy().astype(np.float64)
    months = rows["month"].to_numpy()
    sites = rows["site"].to_numpy()
    everything = np.ones(rows.height, dtype=bool)
    rng = np.random.default_rng(SEED)
    out: dict = {}

    def qs(q: np.ndarray) -> np.ndarray:
        return common.per_row(actual=actual, q=np.sort(q, axis=1), edges=edges)["qs"]

    # (1) own_once and (2) seed spread.
    def delta_from(label_fn, seeds=(0, 1, 2)) -> np.ndarray:  # noqa: ANN001
        out_delta = np.full(rows.height, np.nan)
        for s in seeds:
            frame = pl.read_parquet(
                common.WORK_DIR / "head" / f"cut{label_fn}_seed{s}.parquet",
                columns=["site", "run", "valid_time", "delta"],
            )
            joined = (
                rows.select(common.KEYS)
                .with_row_index("i")
                .join(frame, on=common.KEYS, how="left")
                .sort("i")
            )
            d = joined["delta"].to_numpy()
            out_delta = d if s == seeds[0] else out_delta + d
        return out_delta / len(seeds)

    own = delta_from("2025-07")
    quarterly = hs.head_delta(rows=rows, cadence="quarterly")
    monthly = hs.head_delta(rows=rows, cadence="monthly")
    once = (base["a_h"] - base["a"])[:, 5]
    qs_arms = {
        name: qs(base["a_s"] + d[:, None])
        for name, d in {
            "once": once,
            "own_once": own,
            "quarterly": quarterly,
            "monthly": monthly,
        }.items()
    }
    out["qs_stack_by_head"] = {k: float(v.mean()) for k, v in qs_arms.items()}
    for label, (x, y) in {
        "own_once minus quarterly": ("own_once", "quarterly"),
        "own_once minus once(201)": ("own_once", "once"),
    }.items():
        d = qs_arms[x] - qs_arms[y]
        out[label] = {
            "all_95": common.mean_interval(
                values=d, mask=everything, blocks=months, level=95.0, rng=rng
            )
        }
    seed_qs = []
    for s in (0, 1, 2):
        d = delta_from("2025-07", seeds=(s,))
        seed_qs.append(float(qs(base["a_s"] + d[:, None]).mean()))
    out["own_once_single_seed_qs"] = seed_qs

    # (3) leave one substation out.
    c1 = None
    gam = {}
    for name, arm in gs.CADENCES.items():
        q, _ = gs.aligned(rows=rows, frame=gs.read_arm(arm=arm, directory=common.PREDS_DIR))
        gam[name] = q
    c1 = qs(gam["quarterly"]) - qs(gam["monthly"])
    c3 = qs_arms["once"] - qs_arms["quarterly"]
    out["leave_one_out"] = {}
    for site in common.SCORED_SITES:
        keep = sites != site
        out["leave_one_out"][site] = {
            "C1_95": common.mean_interval(values=c1, mask=keep, blocks=months, level=95.0, rng=rng),
            "C3_95": common.mean_interval(values=c3, mask=keep, blocks=months, level=95.0, rng=rng),
        }
    # (4) median minus mean forecast, arm D, per site.
    out["median_minus_mean"] = {}
    for name, arm in gs.CADENCES.items():
        q, mean = gs.aligned(rows=rows, frame=gs.read_arm(arm=arm, directory=common.PREDS_DIR))
        out["median_minus_mean"][name] = {
            s: float((q[:, 5] - mean)[sites == s].mean()) for s in common.SCORED_SITES
        }
    common.write_json_atomic(path=common.REPORT_DIR / "checks.json", content=out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
