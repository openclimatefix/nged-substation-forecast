"""The known-answer gate of experiment 119, run on direction F1's training year only.

Test 1 injects a known function of the patch into `actual` and checks that ENC-c and ENC-r recover
it at a size that a real gain of 0.05 points would have, and that the centre-only encoder and the
two placebos do not. Test 2 checks that no arm moves the score on the clean data by more than 0.01
points; test 5 checks seed stability and over-fitting. Tests 3 and 4 (reproduction and causality)
are `e00119_checks.py`. The scored year is never opened.

    uv run python ml-experiments/00119-*/e00119_gate.py
"""

import json
import time
from dataclasses import dataclass

import e00119_bundle as bundles
import e00119_common as common
import e00119_cv as cv
import e00119_data as data
import e00119_model as model
import e00119_stats as stats
import numpy as np
import polars as pl
from xgboost import XGBRegressor

SEED = 11
DENSE_TARGETS = (0.10, 0.05, 0.025)
SPARSE_SHARE = 0.10
RECOVERY_PASS = 0.5
PLACEBO_RECOVERY_LIMIT = 0.2
CLEAN_BOUND = 0.01
SEED_SPREAD_BOUND = 0.02
TRAIN_GAP_BOUND = 0.25
PRIMARY_TARGET = 0.05


@dataclass(frozen=True)
class Injection:
    """One known-answer injection.

    Attributes:
        arm: The arm that must recover it.
        kind: `c` adds g to `actual`; `r` adds the temperature offset through the sensitivity.
        target: The score gain, in points, that exact recovery would give over all rows.
        sparse: Whether g is nonzero only on the 10% of days with the largest gradient.
    """

    arm: str
    kind: str
    target: float
    sparse: bool

    @property
    def label(self) -> str:
        """Return a short name for tables."""
        return f"{self.arm} {'sparse' if self.sparse else 'dense'} {self.target:g}"


INJECTIONS: tuple[Injection, ...] = (
    *(Injection("ENC-c", "c", t, False) for t in DENSE_TARGETS),
    Injection("ENC-c", "c", PRIMARY_TARGET, True),
    *(Injection("ENC-r", "r", t, False) for t in DENSE_TARGETS),
    Injection("ENC-r", "r", PRIMARY_TARGET, True),
    Injection("ENC-c0", "c", PRIMARY_TARGET, False),
    Injection("ENC-cps", "c", PRIMARY_TARGET, False),
    Injection("ENC-cpt", "c", PRIMARY_TARGET, False),
)
CLEAN_ARMS: tuple[str, ...] = ("ENC-c", "ENC-r", "ENC-c0", "ENC-cps", "ENC-cpt")


def gradients(*, dataset: data.Dataset, rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's east-minus-west temperature and north-minus-south shortwave gradient.

    The gradients use the member mean of the row's own patch step, over present cells east
    (west, north, south) of the centre cell.
    """
    patch = dataset.patch
    temperature, shortwave = patch[..., 0], patch[..., 2]
    ew = np.zeros(patch.shape[:3], dtype=np.float32)
    ns = np.zeros(patch.shape[:3], dtype=np.float32)
    for i in range(patch.shape[0]):
        dx, dy = dataset.geometry[i, :, 0], dataset.geometry[i, :, 1]
        east, west = dx > 0.01, dx < -0.01
        north, south = dy > 0.01, dy < -0.01
        with np.errstate(all="ignore"):
            ew[i] = np.nanmean(temperature[i][..., east], axis=-1) - np.nanmean(
                temperature[i][..., west], axis=-1
            )
            ns[i] = np.nanmean(shortwave[i][..., north], axis=-1) - np.nanmean(
                shortwave[i][..., south], axis=-1
            )
    site, run, step = (rows[c].to_numpy().astype(np.int64) for c in ("site_index", "run", "step"))
    return np.nan_to_num(ew[site, run, step]), np.nan_to_num(ns[site, run, step])


def standardise(values: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Return values divided by their standard deviation over the masked rows (zero mean)."""
    centre = values[mask].mean()
    return (values - centre) / values[mask].std()


def sparse_days(*, rows: pl.DataFrame, ew: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a mask of the rows on the 10% of days (site, run) with the largest |east-west|."""
    frame = pl.DataFrame(
        {
            "site": rows["site_index"].to_numpy(),
            "run": rows["run"].to_numpy(),
            "ew": np.abs(ew),
            "mask": mask,
        }
    )
    day = (
        frame.filter(pl.col("mask"))
        .group_by("site", "run")
        .agg(pl.col("ew").mean())
        .with_columns(rank=pl.col("ew").rank(descending=True))
    )
    cutoff = np.ceil(SPARSE_SHARE * day.height)
    picked = day.filter(pl.col("rank") <= cutoff).select("site", "run").with_columns(keep=True)
    joined = frame.join(picked, on=["site", "run"], how="left")
    return joined["keep"].fill_null(False).to_numpy()


def size_alpha(
    *,
    actual: np.ndarray,
    k0: np.ndarray,
    shift: np.ndarray,
    mask: np.ndarray,
    target: float,
) -> float:
    """Return alpha so that moving K0 by alpha * shift gains `target` points over the masked rows.

    The gain is the score of K0 on the injected actual minus the score of K0 shifted by exactly
    the injected amount, which is the best an encoder that recovered the injection could do.
    """
    base_actual, base_k0, base_shift = actual[mask], k0[mask], shift[mask]

    def gain(alpha: float) -> float:
        injected = base_actual + alpha * base_shift
        wrong = stats.quantile_score(actual=injected, quantiles=base_k0).mean()
        right = stats.quantile_score(
            actual=injected, quantiles=base_k0 + alpha * base_shift[:, None]
        ).mean()
        return float(wrong - right)

    low, high = 0.0, 1.0
    while gain(high) < target:
        high *= 2.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if gain(mid) < target else (low, mid)
    return 0.5 * (low + high)


def null_actual(*, rows: pl.DataFrame, foreign: np.ndarray) -> np.ndarray:
    """Return a target with no link to the weather: K0's q50 plus another run's residual.

    The partner is the placebo run of `bundles.foreign_runs` (the same year, 2 to 15 days away) and
    the residual is `actual` minus K0's `q50` of that run at the same lead. K0's quantiles stay
    calibrated to the residual distribution. A row whose partner has no row at that lead keeps its
    own actual.
    """
    frame = rows.select("site_index", "run", "lead_hours", "actual", "q50")
    partner = foreign[rows["site_index"].to_numpy().astype(np.int64), rows["run"].to_numpy()]
    keyed = frame.select(
        "site_index",
        partner=pl.col("run"),
        lead_hours=pl.col("lead_hours"),
        residual=pl.col("actual") - pl.col("q50"),
    )
    joined = frame.with_columns(partner=pl.Series(partner)).join(
        keyed, on=["site_index", "partner", "lead_hours"], how="left"
    )
    target = joined["q50"] + joined["residual"]
    return target.fill_null(joined["actual"]).to_numpy().astype(np.float64)


def non_patch_features(*, dataset: data.Dataset, rows: pl.DataFrame) -> np.ndarray:
    """Return the inputs the encoder has besides the patch's other cells, one row per row."""
    site, run, step = (rows[c].to_numpy().astype(np.int64) for c in ("site_index", "run", "step"))
    centre = np.nan_to_num(dataset.patch[site, run, step, 0, :])
    scalars = rows.select("lead_hours", "hod_sin", "hod_cos", "mean").to_numpy()
    return np.column_stack([centre, scalars, site]).astype(np.float32)


def orthogonalise(
    *, values: np.ndarray, features: np.ndarray, rows: pl.DataFrame, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Return values minus a cross-fitted XGBoost prediction from the non-patch inputs.

    Each of the five month blocks of the F1 training year is predicted by a model fitted on the
    other four. Rows outside the mask get 0.

    Args:
        values: The target values to orthogonalise.
        features: The non-patch input features, one row per target value.
        rows: The row table, with a `run_date` column.
        mask: Which rows are fitted and predicted.

    Returns:
        The residuals, and the share of the variance the non-patch inputs explained, out of fold.
    """
    residual = np.zeros_like(values)
    run_date = rows["run_date"]
    for block in cv.blocks_of(direction=cv.F1):
        held = mask & ((run_date >= block[0]) & (run_date < block[1])).to_numpy()
        fit_rows = mask & ~held
        booster = XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.5,
            n_jobs=6,
            tree_method="hist",
        )
        booster.fit(features[fit_rows], values[fit_rows])
        residual[held] = values[held] - booster.predict(features[held])
    explained = 1.0 - residual[mask].var() / values[mask].var()
    return residual, {"explained_variance_share": float(explained)}


def evaluate(
    *,
    rows: pl.DataFrame,
    oof: cv.OofResult,
    actual: np.ndarray,
    k0: np.ndarray,
    oracle: np.ndarray,
) -> dict:
    """Return the gate readings of one out-of-fold run on the (possibly injected) targets."""
    index = oof.index
    site = rows["site_index"].to_numpy()[index]
    decision = np.isin(site, [common.SCORED_SITES.index(s) for s in common.DECISION_SITES])
    act = actual[index]
    enc = stats.quantile_score(actual=act, quantiles=oof.quantiles)
    base = stats.quantile_score(actual=act, quantiles=k0[index])
    best = stats.quantile_score(actual=act, quantiles=oracle[index])
    month = stats.month_of_run(run_date=rows["run_date"])[index]
    interval = stats.month_interval(
        treatment=enc[decision],
        reference=base[decision],
        site=site[decision],
        run=rows["run"].to_numpy()[index][decision],
        month=month[decision],
        level=stats.LEVEL_99,
    )
    per_site = {}
    for name in common.DECISION_SITES:
        at = site == common.SCORED_SITES.index(name)
        gained = float((base[at] - enc[at]).mean())
        possible = float((base[at] - best[at]).mean())
        per_site[name] = {"gain": gained, "possible": possible}
    pooled_gain = float((base[decision] - enc[decision]).mean())
    pooled_possible = float((base[decision] - best[decision]).mean())
    recovered_sites = sum(
        1
        for v in per_site.values()
        if v["possible"] > 0 and v["gain"] >= RECOVERY_PASS * v["possible"]
    )
    return {
        "contrast": interval,
        "gain": pooled_gain,
        "possible": pooled_possible,
        "recovered_share": pooled_gain / pooled_possible if pooled_possible else float("nan"),
        "sites_recovering_half": recovered_sites,
        "per_site": per_site,
        "best_epochs": oof.best_epochs,
    }


def main() -> None:  # noqa: PLR0915
    """Run the redefined gate (one rerun only) and write `gate.json`."""
    started = time.monotonic()
    dataset = data.load_dataset()
    rows = bundles.rows_with_sensitivities(dataset=dataset)
    tensors = bundles.make_tensors(dataset=dataset)
    distant = bundles.distant_sites(dataset=dataset)
    foreign = bundles.foreign_runs(dataset=dataset)
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    direction = cv.F1
    in_year = cv.run_mask(rows=rows, period=direction.train)
    actual = rows["actual"].to_numpy().astype(np.float64)
    k0 = rows.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    ew, ns = gradients(dataset=dataset, rows=rows)
    features = non_patch_features(dataset=dataset, rows=rows)
    ew_resid, ew_share = orthogonalise(
        values=ew.astype(np.float64), features=features, rows=rows, mask=in_year
    )
    ns_resid, ns_share = orthogonalise(
        values=ns.astype(np.float64), features=features, rows=rows, mask=in_year
    )
    z_ew, z_ns = standardise(ew_resid, mask=in_year), standardise(ns_resid, mask=in_year)
    g_dense = 0.5 * z_ew + 0.5 * z_ns
    sparse_mask = sparse_days(rows=rows, ew=ew, mask=in_year) & in_year
    sens_t = rows["sens_t"].to_numpy().astype(np.float64)
    site = rows["site_index"].to_numpy()
    results: dict = {
        "injections": {},
        "clean": {},
        "orthogonalisation": {"east_west": ew_share, "north_south": ns_share},
    }
    for injection in INJECTIONS:
        c_value = decisions[f"F1.{'ENC-r' if injection.kind == 'r' else 'ENC-c'}"]["c"]
        if injection.kind == "c":
            per_unit = g_dense * (sparse_mask if injection.sparse else 1.0)
            alpha_rows = np.full(
                len(rows),
                size_alpha(
                    actual=actual, k0=k0, shift=per_unit, mask=in_year, target=injection.target
                ),
            )
        else:
            per_unit = sens_t * z_ew * (sparse_mask if injection.sparse else 1.0)
            alpha_rows = np.zeros(len(rows))
            for index in range(len(common.SCORED_SITES)):
                at = in_year & (site == index)
                alpha_rows[site == index] = size_alpha(
                    actual=actual, k0=k0, shift=per_unit, mask=at, target=injection.target
                )
        shift = alpha_rows * per_unit
        injected = actual + shift
        oracle = k0 + shift[:, None]
        arm = model.ARMS[injection.arm]
        bundle = bundles.make_bundle(
            rows=rows, tensors=tensors, arm=arm, distant=distant, foreign=foreign, actual=injected
        )
        oof = cv.run_oof(
            rows=rows, bundle=bundle, arm=arm, direction=direction, decay_c=c_value, seed=SEED
        )
        reading = evaluate(rows=rows, oof=oof, actual=injected, k0=k0, oracle=oracle)
        reading["alpha_mean"] = float(alpha_rows[in_year].mean())
        results["injections"][injection.label] = reading
        print(
            f"{injection.label}: recovered {reading['recovered_share']:.2f}, "
            f"sites {reading['sites_recovering_half']}/5, interval "
            f"[{reading['contrast']['lower']:.3f}, {reading['contrast']['upper']:.3f}]",
            flush=True,
        )
    null = null_actual(rows=rows, foreign=foreign)
    for label, target in (("clean", actual), ("null", null)):
        results[label] = {}
        for arm_name in CLEAN_ARMS:
            arm = model.ARMS[arm_name]
            c_value = decisions[f"F1.{'ENC-r' if arm.kind == 'r' else 'ENC-c'}"]["c"]
            bundle = bundles.make_bundle(
                rows=rows, tensors=tensors, arm=arm, distant=distant, foreign=foreign, actual=target
            )
            oof = cv.run_oof(
                rows=rows, bundle=bundle, arm=arm, direction=direction, decay_c=c_value, seed=SEED
            )
            reading = evaluate(rows=rows, oof=oof, actual=target, k0=k0, oracle=k0)
            results[label][arm_name] = reading
            print(f"{label} {arm_name}: gain {reading['gain']:+.4f}", flush=True)
    arm = model.ARMS["ENC-c"]
    bundle = bundles.make_bundle(
        rows=rows, tensors=tensors, arm=arm, distant=distant, foreign=foreign
    )
    per_seed: dict[str, dict] = {}
    differences: list[float] = []
    for seed in cv.SEEDS:
        oof = cv.run_oof(
            rows=rows,
            bundle=bundle,
            arm=arm,
            direction=direction,
            decay_c=decisions["F1.ENC-c"]["c"],
            seed=seed,
        )
        enc = stats.quantile_score(actual=actual[oof.index], quantiles=oof.quantiles)
        base = stats.quantile_score(actual=actual[oof.index], quantiles=k0[oof.index])
        differences.append(float((enc - base).mean()))
        per_seed[str(seed)] = {
            "enc_minus_k0": differences[-1],
            "train_gap": [
                abs(a - b) / b for a, b in zip(oof.train_loss, oof.oof_loss, strict=True)
            ],
        }
    values = differences
    results["seeds"] = {
        "per_seed": per_seed,
        "spread": float(max(values) - min(values)),
        "passes_spread": bool(max(values) - min(values) <= SEED_SPREAD_BOUND),
    }
    common.write_json_atomic(path=common.WORK_DIR / "gate.json", content=results)
    print(f"gate done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
