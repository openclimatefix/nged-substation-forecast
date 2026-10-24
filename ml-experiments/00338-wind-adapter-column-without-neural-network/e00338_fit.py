"""Fit experiment 00338's arms for one farm and store their member mixtures.

Each arm gives XGBoost one extra column, a non-neural effective hub-height speed, in the layout of
experiment 00326's arm (c): the six wind columns plus the effective speed, the binned-curve base
margin, column subsampling off. Mixtures go to `work/mix/<arm>_s<seed>_<hindcast|scored>_<farm>.npy`
(cases, 11), the equal-weight mixture of the 51 members before the spread factor.

Arms: `x1` (lookup table), `x2` (boosted scale), `x3` (table distilled from the neural adapter),
`x1_no_direction` and `x1_no_ratio` (drop-one tables), and the permuted-column placebos `x1_pl`
and `x2_pl`. Arms (c) and the default are read from experiments 303 and 326.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      env PYTHONPATH=$P303:$P326:$P338 \
      uv run --no-sync python ml-experiments/00338-*/e00338_fit.py --farms W1

where `$P303`, `$P326`, and `$P338` are the experiment folders under `ml-experiments/`.
"""

import argparse
import os
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("E00303_TORCH_THREADS", "4")

import e00303_ens as ens  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_fit as fit326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00338_adapters as adapters  # noqa: E402
import e00338_common as common  # noqa: E402
import numpy as np  # noqa: E402

XGB_THREADS = int(os.environ.get("E00338_XGB_THREADS", "4"))


def stored(*, arm: str, seed: int, set_name: str, farm: str, array: np.ndarray) -> None:
    """Write one mixture atomically."""
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    ens.write_atomic(path=common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy", array=array)


def present(*, arm: str, farm: str) -> bool:
    """Return whether every mixture of `arm` for `farm` is stored."""
    return all(
        (common.MIX_DIR / f"{arm}_s{seed}_{name}_{farm}.npy").exists()
        for seed in common.SEEDS
        for name in ("hindcast", "scored")
    )


def fit_and_store(
    *,
    data: fit303.Hindcast,
    arm: str,
    seed: int,
    farm: str,
    train: dict[str, np.ndarray],
    effective_train: np.ndarray,
    effective_sets: dict[str, np.ndarray],
) -> None:
    """Fit XGBoost with `effective_train` as the extra column and store both sets' mixtures."""
    booster, centres, curve = fit326.fit_booster(
        train=data.train,
        features=fit326.matrix(columns=train, effective=effective_train),
        seed=seed,
    )
    booster.set_param({"nthread": XGB_THREADS})
    for name, columns in (("hindcast", data.hind_columns), ("scored", data.scored_columns)):
        stored(
            arm=arm,
            seed=seed,
            set_name=name,
            farm=farm,
            array=fit326.mixture(
                booster=booster,
                centres=centres,
                curve=curve,
                columns=columns,
                effective=effective_sets[name],
            ),
        )


def run_farm(*, farm: str, arms: list[str]) -> None:
    """Fit and store the requested arms for one farm."""
    data = fit303.Hindcast(farm=farm)
    train = fit326.train_columns(data=data)
    target = data.train["target"].to_numpy()
    sets = {"hindcast": data.hind_columns, "scored": data.scored_columns}
    edges = adapters.make_edges(
        columns=train, ratio_bins=common.RATIO_BINS, temperature_bins=common.TEMPERATURE_BINS
    )
    pending = [a for a in arms if not present(arm=a, farm=farm)]
    common.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in common.SEEDS:
        started = time.time()
        table_cores = {}
        for arm, use_ratio, use_sector in (
            ("x1", True, True),
            ("x1_no_direction", True, False),
            ("x1_no_ratio", False, True),
        ):
            needs = {arm} | ({"x1_pl"} if arm == "x1" else set())
            if not needs & set(pending):
                continue
            arm_edges = (
                edges
                if use_ratio
                else adapters.Edges(
                    ratio=np.array([], dtype=np.float32), temperature=edges.temperature
                )
            )
            core = adapters.fit_table_core(
                train_columns=train,
                target=target,
                edges=arm_edges,
                with_idle_layer=data.with_idle_layer,
                seed=seed,
                use_sector=use_sector,
            )
            table_cores[arm] = core
            np.save(
                common.TABLE_DIR / f"{arm}_s{seed}_{farm}.npy",
                core.table_scale().detach().numpy(),
            )
            effective_train = adapters.hub_speed(core=core, columns=train)
            effective_sets = {n: adapters.hub_speed(core=core, columns=c) for n, c in sets.items()}
            if arm in pending:
                fit_and_store(
                    data=data,
                    arm=arm,
                    seed=seed,
                    farm=farm,
                    train=train,
                    effective_train=effective_train,
                    effective_sets=effective_sets,
                )
            if arm == "x1" and "x1_pl" in pending:
                rng = np.random.default_rng(common.SHUFFLE_SEED + seed)
                fit_and_store(
                    data=data,
                    arm="x1_pl",
                    seed=seed,
                    farm=farm,
                    train=train,
                    effective_train=effective_train[rng.permutation(effective_train.size)],
                    effective_sets={
                        n: v[rng.permutation(v.size)] for n, v in effective_sets.items()
                    },
                )
        if {"x2", "x2_pl"} & set(pending):
            core, _ = fit303.fit_era5_core(
                train=data.train, adapter="none", seed=seed, with_idle_layer=data.with_idle_layer
            )
            reference = table_cores.get("x1") or adapters.fit_table_core(
                train_columns=train,
                target=target,
                edges=edges,
                with_idle_layer=data.with_idle_layer,
                seed=seed,
            )
            scale = (
                reference.table_scale()
                .detach()
                .numpy()[adapters.cell_of(columns=train, edges=edges)]
            )
            constraints = adapters.monotone_constraints(scale=scale, columns=train)
            boosted = adapters.fit_boosted_scale(
                core=core, train_columns=train, target=target, constraints=constraints, seed=seed
            )
            (common.TABLE_DIR / f"x2_s{seed}_{farm}.txt").write_text(constraints)
            effective_train = boosted.effective_speed(columns=train)
            effective_sets = {n: boosted.effective_speed(columns=c) for n, c in sets.items()}
            if "x2" in pending:
                fit_and_store(
                    data=data,
                    arm="x2",
                    seed=seed,
                    farm=farm,
                    train=train,
                    effective_train=effective_train,
                    effective_sets=effective_sets,
                )
            if "x2_pl" in pending:
                rng = np.random.default_rng(common.SHUFFLE_SEED + seed)
                fit_and_store(
                    data=data,
                    arm="x2_pl",
                    seed=seed,
                    farm=farm,
                    train=train,
                    effective_train=effective_train[rng.permutation(effective_train.size)],
                    effective_sets={
                        n: v[rng.permutation(v.size)] for n, v in effective_sets.items()
                    },
                )
        if "x3" in pending:
            net_core, _ = fit303.fit_era5_core(
                train=data.train, adapter="net", seed=seed, with_idle_layer=data.with_idle_layer
            )
            table, counts = adapters.distilled_table(
                net_core=net_core,
                train_columns=train,
                edges=edges,
                with_idle_layer=data.with_idle_layer,
            )
            np.save(
                common.TABLE_DIR / f"x3_s{seed}_{farm}.npy", table.table_scale().detach().numpy()
            )
            np.save(common.TABLE_DIR / f"x3_counts_s{seed}_{farm}.npy", counts)
            fit_and_store(
                data=data,
                arm="x3",
                seed=seed,
                farm=farm,
                train=train,
                effective_train=adapters.hub_speed(core=table, columns=train),
                effective_sets={
                    n: adapters.hub_speed(core=table, columns=c) for n, c in sets.items()
                },
            )
        print(f"{farm} seed {seed}: {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Fit the requested arms for the requested farms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default=",".join(common.FARMS))
    parser.add_argument("--arms", default=",".join(common.FIT_ARMS))
    args = parser.parse_args()
    for farm in args.farms.split(","):
        run_farm(farm=farm, arms=args.arms.split(","))


if __name__ == "__main__":
    main()
