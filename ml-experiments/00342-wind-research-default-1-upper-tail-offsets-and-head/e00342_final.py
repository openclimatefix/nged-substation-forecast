"""Experiment 00342: build the upper-tail arms from the default's hindcast and scored quantiles.

The default is rebuilt exactly as in experiment 00340 (experiment 00336's stored mixtures, then
experiment 33's spread factor and experiment 259's offset table). Every arm below is then fitted on
the usable hindcast runs only and applied to the default's scored quantiles:

- `x1lit`: the brief's literal x1, an offset table for q90, q95, and q99 per farm and lead day,
  plus the farm-scale cap. The default's table already does this, so it should equal the default.
- `x1`: the same offsets per farm, lead day, and bin of the ensemble-mean speed, plus the cap.
- `x2`: boosted trees that add offsets to q90, q95, and q99 (inputs: the default's quantiles, the
  members' speed mean, standard deviation, and share above the rated speed, the lead, the farm).
  `x2base`: the same without the member summaries. `x2plc`: member summaries from another case of
  the same farm and lead day. `x2lofo`: trained on the other two farms, no farm input.
- `x3`: a probability mass at the farm scale, with weight lambda x (share of members above the
  rated speed); lambda is fitted per farm.
- `x1cap`: only the cap on the default (descriptive); `x1capstrong`: the cap as the plan first
  defined it, from the rows at ERA5 speeds above the rated speed (descriptive).

Outputs, in `work/final/`: `<arm>.npy` (3 seeds, scored cases, 11), `<arm>_rep.npy`,
`x2_5seeds.npy` (5 head seeds on default seed 0), `default_notable.npy` and
`replicate_cases.parquet` (copied), `fits.json`, and `reproduction.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue env OMP_NUM_THREADS=1 \
      uv run --no-sync python ml-experiments/00342-*/e00342_final.py   (PYTHONPATH: see common)
"""

import json
import os
import shutil
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_final as final336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00342_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

c303 = common.c303
c336 = common.c336
POINTS = c303.POINTS_PER_FRACTION
THREADS = 14
XGB_PARAMS = {
    "objective": "reg:quantileerror",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.05,
    "min_child_weight": 50,
    "subsample": 0.8,
    "nthread": THREADS,
}
ROUNDS = 150


def member_stats(*, set_name: str) -> np.ndarray:
    """Return the member speed summaries (cases, 3) of the three farms in order."""
    return np.concatenate(
        [np.load(common.stats_path(set_name=set_name, farm=f)) for f in common.FARMS]
    )


def head_features(
    *, base: np.ndarray, members: np.ndarray, lead: np.ndarray, sites: np.ndarray, kind: str
) -> np.ndarray:
    """Return the x2 inputs: the 11 base quantiles, optional member summaries, lead, farm."""
    parts = [base.astype(np.float32), (lead / 100.0)[:, None].astype(np.float32)]
    if kind != "lofo":
        parts.append(np.stack([(sites == f).astype(np.float32) for f in common.FARMS], axis=1))
    if kind in ("full", "lofo", "placebo"):
        parts.append(members.astype(np.float32))
    return np.concatenate(parts, axis=1)


def placebo_members(*, members: np.ndarray, sites: np.ndarray, lead_day: np.ndarray) -> np.ndarray:
    """Return `members` with each row replaced by another row of the same farm and lead day."""
    rng = np.random.default_rng(342)
    out = members.copy()
    keys = np.char.add(sites.astype(str), lead_day.astype(int).astype(str))
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        out[rows] = members[rng.permutation(rows)]
    return out


def head_predict(
    *,
    train_x: np.ndarray,
    train_base: np.ndarray,
    train_actual: np.ndarray,
    score_x: np.ndarray,
    score_base: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit one tree model per upper level on the hindcast and return `score_base` with q90/95/99."""
    out = score_base.astype(np.float64).copy()
    for k in common.UPPER_INDEX:
        params = XGB_PARAMS | {"quantile_alpha": float(c303.LEVELS[k]), "seed": seed}
        train = xgb.DMatrix(
            train_x, label=train_actual.astype(np.float32), base_margin=train_base[:, k]
        )
        booster = xgb.train(params, train, num_boost_round=ROUNDS)
        out[:, k] = booster.predict(xgb.DMatrix(score_x, base_margin=score_base[:, k]))
    return out


def finish(*, quantiles: np.ndarray, sites: np.ndarray, caps: dict[str, float]) -> np.ndarray:
    """Clip to 0 to 1, sort along the levels, and apply the farm-scale cap."""
    ordered = np.sort(np.clip(quantiles, 0.0, 1.0), axis=1)
    return common.cap_quantiles(quantiles=ordered, sites=sites, caps=caps)


def main() -> None:  # noqa: PLR0915
    """Write the quantiles of every arm."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sites_table = c303.site_table()
    rated = score326.rated_speeds(sites=sites_table)
    hind = ens_final.hindcast_cases(sites=sites_table)
    scored = ens_final.scored_cases(sites=sites_table)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    hind_actual = hind_usable["actual_norm"].to_numpy().astype(np.float64)
    in_replicate = (hind_usable["power_fcst_init_time"] >= common.REPLICATE_START).to_numpy()
    fit_rows = (hind_usable["valid_time"] < common.REPLICATE_START).to_numpy()
    replicate_cases = pl.read_parquet(c336.FINAL_DIR / "replicate_cases.parquet")
    expected_rep = hind_usable.filter(pl.Series(in_replicate))
    if not (
        replicate_cases.height == expected_rep.height
        and (replicate_cases["site"] == expected_rep["site"]).all()
        and (replicate_cases["valid_time"] == expected_rep["valid_time"]).all()
        and (replicate_cases["power_fcst_init_time"] == expected_rep["power_fcst_init_time"]).all()
    ):
        msg = "experiment 336's replicate cases differ from the hindcast cases from 2025-01"
        raise ValueError(msg)
    shutil.copy(c336.FINAL_DIR / "replicate_cases.parquet", common.FINAL_DIR)
    shutil.copy(c336.FINAL_DIR / "default_notable.npy", common.FINAL_DIR)
    all_cases = pl.concat(
        [
            hind_usable.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    era5_hind = final336.stats(kind="era5", set_name="hindcast")[usable]
    sites_hind = hind_usable["site"].to_numpy()
    sites_scored = scored["site"].to_numpy()
    lead_day_hind = hind_usable["lead_day"].to_numpy()
    lead_day_scored = scored["lead_day"].to_numpy()
    lead_hind = hind_usable["lead_hours"].to_numpy().astype(np.float64)
    lead_scored = scored["lead_hours"].to_numpy().astype(np.float64)
    members_hind = member_stats(set_name="hindcast")[usable]
    members_scored = member_stats(set_name="scored")
    if members_scored.shape[0] != scored.height:
        msg = "member statistics and scored cases differ in length"
        raise ValueError(msg)
    caps = common.farm_caps(actual=hind_actual, sites=sites_hind)
    caps_strong = common.farm_caps_strong(
        actual=hind_actual, era5=era5_hind, sites=sites_hind, rated=rated
    )
    caps_rep = common.farm_caps(actual=hind_actual[fit_rows], sites=sites_hind[fit_rows])
    caps_rep_strong = common.farm_caps_strong(
        actual=hind_actual[fit_rows],
        era5=era5_hind[fit_rows],
        sites=sites_hind[fit_rows],
        rated=rated,
    )
    top_of = {
        "hind": np.array([caps[s] for s in sites_hind]),
        "scored": np.array([caps[s] for s in sites_scored]),
    }
    # Case subsets for the replicate period.
    sub = {
        "rep_score": {
            "sites": sites_hind[in_replicate],
            "lead_day": lead_day_hind[in_replicate],
            "lead": lead_hind[in_replicate],
            "members": members_hind[in_replicate],
        },
        "rep_train": {
            "sites": sites_hind[fit_rows],
            "lead_day": lead_day_hind[fit_rows],
            "lead": lead_hind[fit_rows],
            "members": members_hind[fit_rows],
        },
    }
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    fits: dict = {
        "caps_points": {f: v * POINTS for f, v in caps.items()},
        "caps_strong_points": {f: v * POINTS for f, v in caps_strong.items()},
        "rated": rated,
    }

    def put(*, arm: str, main: np.ndarray, rep: np.ndarray) -> None:
        out.setdefault(arm, {"main": [], "rep": []})
        out[arm]["main"].append(main.astype(np.float32))
        out[arm]["rep"].append(rep.astype(np.float32))

    five_seed_outputs = []
    for seed in common.SEEDS:
        raw_h = final336.load(arm="default", seed=seed, set_name="hindcast")[usable]
        raw_s = final336.load(arm="default", seed=seed, set_name="scored")
        adjusted = c303.apply_spread(
            cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
        )
        adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
        table, keys = ens_final.offsets_table(hindcast=hind_usable, quantiles=adj_h)
        scored_q = ens_final.apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
        hind_q = ens_final.apply_offsets(
            cases=hind_usable, quantiles=adj_h, table=table, keys=keys
        ).astype(np.float64)
        table_r, keys_r = ens_final.offsets_table(
            hindcast=hind_usable.filter(pl.Series(fit_rows)), quantiles=adj_h[fit_rows]
        )
        rep_q = ens_final.apply_offsets(
            cases=replicate_cases, quantiles=adj_h[in_replicate], table=table_r, keys=keys_r
        )
        hind_q_rep = ens_final.apply_offsets(
            cases=hind_usable.filter(pl.Series(fit_rows)),
            quantiles=adj_h[fit_rows],
            table=table_r,
            keys=keys_r,
        ).astype(np.float64)
        put(arm="default", main=scored_q, rep=rep_q)
        # Each period: (train quantiles, train actual, train sub-key, scored quantiles, scored key).
        periods: dict[str, dict[str, Any]] = {
            "main": {
                "tq": hind_q,
                "ta": hind_actual,
                "t_sites": sites_hind,
                "t_day": lead_day_hind,
                "t_lead": lead_hind,
                "t_members": members_hind,
                "sq": scored_q.astype(np.float64),
                "s_sites": sites_scored,
                "s_day": lead_day_scored,
                "s_lead": lead_scored,
                "s_members": members_scored,
                "t_top": top_of["hind"],
                "caps": caps,
                "caps_strong": caps_strong,
            },
            "rep": {
                "tq": hind_q_rep,
                "ta": hind_actual[fit_rows],
                "t_sites": sub["rep_train"]["sites"],
                "t_day": sub["rep_train"]["lead_day"],
                "t_lead": sub["rep_train"]["lead"],
                "t_members": sub["rep_train"]["members"],
                "sq": rep_q.astype(np.float64),
                "s_sites": sub["rep_score"]["sites"],
                "s_day": sub["rep_score"]["lead_day"],
                "s_lead": sub["rep_score"]["lead"],
                "s_members": sub["rep_score"]["members"],
                "t_top": np.array([caps_rep[x] for x in sub["rep_train"]["sites"]]),
                "caps": caps_rep,
                "caps_strong": caps_rep_strong,
            },
        }
        results: dict[str, dict[str, np.ndarray]] = {}
        for period, p in periods.items():
            res: dict[str, np.ndarray] = {}
            t_cells = common.cell_ids(
                sites=p["t_sites"], lead_day=p["t_day"], mean_speed=p["t_members"][:, 0]
            )
            s_cells = common.cell_ids(
                sites=p["s_sites"], lead_day=p["s_day"], mean_speed=p["s_members"][:, 0]
            )
            res["x1cap"] = finish(quantiles=p["sq"], sites=p["s_sites"], caps=p["caps"])
            res["x1capstrong"] = finish(
                quantiles=p["sq"], sites=p["s_sites"], caps=p["caps_strong"]
            )
            literal_t = common.cell_ids(
                sites=p["t_sites"], lead_day=p["t_day"], mean_speed=np.zeros(p["t_sites"].size)
            )
            literal_s = common.cell_ids(
                sites=p["s_sites"], lead_day=p["s_day"], mean_speed=np.zeros(p["s_sites"].size)
            )
            lit_table = common.fit_x1_table(quantiles=p["tq"], actual=p["ta"], cells=literal_t)
            res["x1lit"] = finish(
                quantiles=common.apply_x1_table(
                    quantiles=p["sq"], cells=literal_s, table=lit_table
                ),
                sites=p["s_sites"],
                caps=p["caps"],
            )
            x1_table = common.fit_x1_table(quantiles=p["tq"], actual=p["ta"], cells=t_cells)
            res["x1"] = finish(
                quantiles=common.apply_x1_table(quantiles=p["sq"], cells=s_cells, table=x1_table),
                sites=p["s_sites"],
                caps=p["caps"],
            )
            if period == "main":
                fits.setdefault("x1_offsets_points", {})[f"seed{seed}"] = {
                    str(c): (v * POINTS).round(3).tolist() for c, v in x1_table.items()
                }
                fits.setdefault("x1lit_max_abs_offset_points", {})[f"seed{seed}"] = float(
                    max((np.abs(v).max() for v in lit_table.values()), default=0.0) * POINTS
                )
            # x2 and its controls.
            base_t, base_s = p["tq"], p["sq"]
            full_t = head_features(
                base=base_t,
                members=p["t_members"],
                lead=p["t_lead"],
                sites=p["t_sites"],
                kind="full",
            )
            full_s = head_features(
                base=base_s,
                members=p["s_members"],
                lead=p["s_lead"],
                sites=p["s_sites"],
                kind="full",
            )
            res["x2"] = finish(
                quantiles=head_predict(
                    train_x=full_t,
                    train_base=base_t,
                    train_actual=p["ta"],
                    score_x=full_s,
                    score_base=base_s,
                    seed=seed,
                ),
                sites=p["s_sites"],
                caps=p["caps"],
            )
            if period == "main" and seed == 0:
                five_seed_outputs.extend(
                    finish(
                        quantiles=head_predict(
                            train_x=full_t,
                            train_base=base_t,
                            train_actual=p["ta"],
                            score_x=full_s,
                            score_base=base_s,
                            seed=head_seed,
                        ),
                        sites=p["s_sites"],
                        caps=p["caps"],
                    )
                    for head_seed in common.HEAD_SEEDS
                )
            only_t = head_features(
                base=base_t,
                members=p["t_members"],
                lead=p["t_lead"],
                sites=p["t_sites"],
                kind="base",
            )
            only_s = head_features(
                base=base_s,
                members=p["s_members"],
                lead=p["s_lead"],
                sites=p["s_sites"],
                kind="base",
            )
            res["x2base"] = finish(
                quantiles=head_predict(
                    train_x=only_t,
                    train_base=base_t,
                    train_actual=p["ta"],
                    score_x=only_s,
                    score_base=base_s,
                    seed=seed,
                ),
                sites=p["s_sites"],
                caps=p["caps"],
            )
            plc_t = head_features(
                base=base_t,
                members=placebo_members(
                    members=p["t_members"], sites=p["t_sites"], lead_day=p["t_day"]
                ),
                lead=p["t_lead"],
                sites=p["t_sites"],
                kind="placebo",
            )
            plc_s = head_features(
                base=base_s,
                members=placebo_members(
                    members=p["s_members"], sites=p["s_sites"], lead_day=p["s_day"]
                ),
                lead=p["s_lead"],
                sites=p["s_sites"],
                kind="placebo",
            )
            res["x2plc"] = finish(
                quantiles=head_predict(
                    train_x=plc_t,
                    train_base=base_t,
                    train_actual=p["ta"],
                    score_x=plc_s,
                    score_base=base_s,
                    seed=seed,
                ),
                sites=p["s_sites"],
                caps=p["caps"],
            )
            lofo = base_s.copy()
            lofo_t = head_features(
                base=base_t,
                members=p["t_members"],
                lead=p["t_lead"],
                sites=p["t_sites"],
                kind="lofo",
            )
            lofo_s = head_features(
                base=base_s,
                members=p["s_members"],
                lead=p["s_lead"],
                sites=p["s_sites"],
                kind="lofo",
            )
            for farm in common.FARMS:
                train_rows = p["t_sites"] != farm
                score_rows = p["s_sites"] == farm
                lofo[score_rows] = head_predict(
                    train_x=lofo_t[train_rows],
                    train_base=base_t[train_rows],
                    train_actual=p["ta"][train_rows],
                    score_x=lofo_s[score_rows],
                    score_base=base_s[score_rows],
                    seed=seed,
                )
            res["x2lofo"] = finish(quantiles=lofo, sites=p["s_sites"], caps=p["caps"])
            # x3: the saturation mass, lambda fitted per farm on the training quantiles.
            x3 = base_s.copy()
            lambdas = {}
            for farm in common.FARMS:
                rows_t = p["t_sites"] == farm
                lam = common.fit_lambda(
                    quantiles=base_t[rows_t],
                    actual=p["ta"][rows_t],
                    share=p["t_members"][rows_t, 2].astype(np.float64),
                    top=p["t_top"][rows_t],
                )
                lambdas[farm] = lam
                rows_s = p["s_sites"] == farm
                x3[rows_s] = common.saturation_mass(
                    quantiles=base_s[rows_s],
                    weight=lam * p["s_members"][rows_s, 2].astype(np.float64),
                    top=np.full(int(rows_s.sum()), p["caps"][farm]),
                )
            res["x3"] = finish(quantiles=x3, sites=p["s_sites"], caps=p["caps"])
            if period == "main":
                fits.setdefault("x3_lambda", {})[f"seed{seed}"] = lambdas
            results[period] = res
        for arm in results["main"]:
            put(arm=arm, main=results["main"][arm], rep=results["rep"][arm])
        print(f"seed {seed} done: lambdas {fits['x3_lambda'][f'seed{seed}']}", flush=True)
    for arm, parts in out.items():
        np.save(common.FINAL_DIR / f"{arm}.npy", np.stack(parts["main"]))
        np.save(common.FINAL_DIR / f"{arm}_rep.npy", np.stack(parts["rep"]))
    np.save(common.FINAL_DIR / "x2_5seeds.npy", np.stack(five_seed_outputs).astype(np.float32))
    # Reproduction gate against experiment 336's refit and experiment 303's P1.
    scored_actual = scored["actual_norm"].to_numpy().astype(np.float64)
    keep = (~scored["masked"]).to_numpy()
    stored = np.load(c336.FINAL_DIR / "default.npy")
    mine = np.stack(out["default"]["main"])
    report = {
        "known_answer_test": common.known_answer_test(),
        "leakage_check_fit_rows": {
            "latest_hindcast_valid_time": str(hind_usable["valid_time"].max()),
            "latest_replicate_fit_valid_time": str(
                hind_usable.filter(pl.Series(fit_rows))["valid_time"].max()
            ),
            "earliest_scored_valid_time": str(scored["valid_time"].min()),
        },
        "default_max_abs_difference_to_experiment_336": float(np.abs(mine - stored).max()),
        "default_kept_score": final336.score(quantiles=mine, actual=scored_actual, keep=keep),
        "kept_score_stored_p1": final336.score(
            quantiles=np.load(c336.c326.STORED_303_FINAL_DIR / "P1.npy"),
            actual=scored_actual,
            keep=keep,
        ),
    }
    report["gate_difference"] = abs(report["default_kept_score"] - report["kept_score_stored_p1"])
    report["gate_passed"] = bool(report["gate_difference"] <= final336.GATE_TOLERANCE)
    (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(report, indent=1))
    (common.FINAL_DIR / "fits.json").write_text(json.dumps(fits, indent=1))
    print("reproduction", report, flush=True)


if __name__ == "__main__":
    main()
