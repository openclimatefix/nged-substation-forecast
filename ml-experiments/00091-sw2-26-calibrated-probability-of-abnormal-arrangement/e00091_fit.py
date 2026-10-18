"""Calibrate the three arms on odd-month transfers, score them on even-month transfers.

Reads the rows of `e00091_rows.py` and prints every number the write-up quotes. The calibration
(a logistic regression, then isotonic regression) is fitted on rows whose reference transfer
starts in an odd calendar month. Every score is on even months. Intervals come from resampling
whole injection batches.

    uv run python ml-experiments/00091-*/e00091_fit.py
"""

import pickle
from dataclasses import dataclass
from typing import Final

import e00091_common as common
import e00091_grid as grid
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROWS_DIR: Final = common.WORK_DIR / "rows"
OUT_DIR: Final = common.WORK_DIR / "results"
HEADLINE_SHAPES: Final[tuple[str, ...]] = ("S2", "S3", "S4", "S6")
ALL_SHAPES: Final[tuple[str, ...]] = ("S1", *HEADLINE_SHAPES)
N_RESAMPLES: Final[int] = 10_000
LEVEL: Final[float] = 0.975
"""Two planned contrasts, so each is read at 97.5% (Bonferroni)."""
RELIABILITY_MARGIN: Final[float] = 0.1
BIN_SHARE: Final[float] = 0.05
SEED: Final[int] = 91
ARMS: Final[tuple[str, ...]] = ("flag", "latest", "latest_x1", "best")
"""`flag` is arm 1, `latest` arm 2, `latest_x1` arm 3, and `best` the post hoc variant of arm 2."""
FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "latest": ("score", "log_age", "log_since", "returned", "return_z"),
    "latest_x1": ("score", "log_age", "log_since", "returned", "return_z", "x1"),
    "best": ("score", "log_age", "log_since", "returned", "return_z"),
}


@dataclass
class Fitted:
    """One arm's calibration: logistic model, isotonic map, and the rate with no candidate."""

    arm: str
    logistic: LogisticRegression | None
    isotonic: IsotonicRegression | None
    base_rate: float
    flag_rates: tuple[float, float] | None = None


LARGE_STEP_Z: Final[float] = 4.0
"""Post hoc: a transfer is large when its donor's first-day step is this many noise units."""
VARIANTS: Final[tuple[str, ...]] = ("all", "clean", "clean_large")
"""`all` is pre-registered; `clean` drops rows the uninjected detector already flags (post hoc);
`clean_large` also drops rows inside injected transfers whose donor step is under `LARGE_STEP_Z`."""


def load_rows() -> pl.DataFrame:
    """Read every rows file and tag each row's calibration split and its background flag.

    A row inside an injected transfer belongs to the calendar month the transfer starts in; any
    other row belongs to the month of its lag stamp. Odd months fit, even months score. Rows inside
    a logged closure or a labelled event are dropped for injected shapes. `background` is true
    where the same detector, run on the uninjected residuals at the same cut, already flags the row.
    """
    frame = pl.concat(
        [pl.read_parquet(p) for p in sorted(ROWS_DIR.glob("rows_*.parquet"))],
        how="vertical_relaxed",
    )
    months = grid.stamp_months()
    real = frame.filter(pl.col("shape") == "REAL")
    background = real.select("cut", "series", "lag_stamp", pl.col("flag").alias("background"))
    injected = (
        frame.filter(pl.col("shape") != "REAL")
        .filter(~pl.col("blocked"))
        .join(background, on=["cut", "series", "lag_stamp"], how="left")
    )
    lag_month = months[injected["lag_stamp"].to_numpy()]
    onset = injected["ev_onset"].to_numpy()
    onset_month = months[np.nan_to_num(onset, nan=0).astype(int)]
    row_month = np.where(np.isfinite(onset), onset_month, lag_month)
    injected = injected.with_columns(
        row_month=pl.Series(row_month),
        fit=pl.Series(row_month % 2 == 1),
        age_hours=pl.coalesce("ev_age_hours", "cut_age_hours"),
    )
    return pl.concat([injected, real], how="diagonal_relaxed")


LONG_SEED_FROM: Final[int] = 10_000
"""Batch seeds from here on belong to the second pass of longer transfers."""
DURATION_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 24 h", 0.0, 24.0),
    ("1 to 3 d", 24.0, 72.0),
    ("3 to 14 d", 72.0, 336.0),
    ("over 14 d", 336.0, 1e9),
)
"""Duration bins of the newest active transfer at a cut, in hours."""


def add_reference_hours(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `ref_hours`: the duration of the newest-onset transfer active at each row's cut.

    Active means the transfer starts before the cut and ends within the 7-day lag window of it.
    """
    import e00091_harness as harness

    state = harness.make_state()
    main_events = harness.cached_events(power=state["panel"]["power"])
    batches = dict(harness.seeded_batches(events=main_events))
    long_path = common.WORK_DIR / "long_events.pkl"
    if long_path.exists():
        from e00091_inject import batches as split

        for i, batch in enumerate(split(events=pickle.loads(long_path.read_bytes()))):
            batches[9100 + 1000 + i] = batch
    pairs = frame.select("seed", "cut").unique()
    hours = []
    for seed, cut in pairs.iter_rows():
        active = [e for e in batches[seed] if e.onset < cut and e.end > cut - 336]
        hours.append(max(active, key=lambda e: e.onset) if active else None)
    pairs = pairs.with_columns(
        ref_hours=pl.Series([(e.end - e.onset) / 2 if e else None for e in hours], dtype=pl.Float64)
    )
    return frame.join(pairs, on=["seed", "cut"], how="left")


def variant_rows(*, frame: pl.DataFrame, variant: str) -> pl.DataFrame:
    """Return the injected rows of one variant (see `VARIANTS`), from the main draw only."""
    injected = frame.filter((pl.col("shape") != "REAL") & (pl.col("seed") < LONG_SEED_FROM))
    if variant == "all":
        return injected
    injected = injected.filter(~pl.col("background").fill_null(False))
    if variant == "clean":
        return injected
    return injected.filter(pl.col("ev_step_z").is_null() | (pl.col("ev_step_z") >= LARGE_STEP_Z))


def arm_columns(*, frame: pl.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the has-candidate mask and the feature matrix of an arm (candidate rows only)."""
    prefix = "b" if arm == "best" else "l"
    have = frame[f"has_{'best' if arm == 'best' else 'latest'}"].to_numpy()
    cols = {
        "score": frame[f"{prefix}_score"].to_numpy(),
        "log_age": np.log1p(frame[f"{prefix}_age_days"].to_numpy()),
        "log_since": np.log1p(frame[f"{prefix}_since_days"].to_numpy()),
        "returned": frame[f"{prefix}_returned"].to_numpy().astype(float),
        "return_z": frame[f"{prefix}_return_z"].to_numpy(),
        "x1": frame[f"{prefix}_x1"].to_numpy(),
    }
    matrix = (
        np.column_stack([cols[name] for name in FEATURES.get(arm, ())]) if arm in FEATURES else None
    )
    return have, matrix  # ty: ignore[invalid-return-type]


def fit_arm(*, frame: pl.DataFrame, arm: str) -> Fitted:
    """Fit one arm's calibration on the odd-month rows of `frame`."""
    y = frame["y"].to_numpy().astype(float)
    if arm == "flag":
        flag = frame["flag"].to_numpy()
        rates = (float(y[~flag].mean()), float(y[flag].mean()))
        return Fitted(arm, None, None, rates[0], rates)
    have, matrix = arm_columns(frame=frame, arm=arm)
    logistic = LogisticRegression(C=1.0, max_iter=1000).fit(matrix[have], y[have])
    raw = logistic.predict_proba(matrix[have])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y[have])
    return Fitted(arm, logistic, isotonic, float(y[~have].mean()))


def predict(*, fitted: Fitted, frame: pl.DataFrame) -> np.ndarray:
    """Return the calibrated probability of every row of `frame`."""
    if fitted.arm == "flag":
        flag = frame["flag"].to_numpy()
        return np.where(flag, fitted.flag_rates[1], fitted.flag_rates[0])  # ty: ignore[not-subscriptable]
    have, matrix = arm_columns(frame=frame, arm=fitted.arm)
    out = np.full(frame.height, fitted.base_rate)
    raw = fitted.logistic.predict_proba(matrix[have])[:, 1]  # ty: ignore[unresolved-attribute]
    out[have] = fitted.isotonic.predict(raw)  # ty: ignore[unresolved-attribute]
    return out


def brier(*, p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of a probability."""
    return float(np.mean((p - y) ** 2))


def auc(*, p: np.ndarray, y: np.ndarray) -> float:
    """Area under the receiver operating characteristic curve, NaN with one class."""
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def reliability(*, p: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """Reliability in tenths of probability, with each bin's share of rows."""
    edges = np.linspace(0, 1, 11)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    rows = []
    for b in range(10):
        m = index == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "share": float(m.mean()),
                    "mean_p": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                    "gap": float(abs(p[m].mean() - y[m].mean())),
                }
            )
    return pl.DataFrame(rows)


def cluster_sums(*, frame: pl.DataFrame, predictions: dict[str, np.ndarray]) -> pl.DataFrame:
    """Per (shape, seed) batch: row count and the sum of squared errors of every arm."""
    y = frame["y"].to_numpy().astype(float)
    extra = {f"sse_{arm}": (p - y) ** 2 for arm, p in predictions.items()}
    return (
        frame.select("shape", "seed")
        .with_columns(**{k: pl.Series(v) for k, v in extra.items()})
        .group_by("shape", "seed")
        .agg(pl.len().alias("n"), *[pl.col(k).sum() for k in extra])
    )


def bootstrap_contrast(*, sums: pl.DataFrame, arm: str, versus: str = "flag") -> dict[str, float]:
    """Relative change in Brier score of `arm` against `versus`, resampling batches.

    The Brier score of a shape is the sum over its drawn batches' squared errors over their rows,
    and the pooled score is the mean over `HEADLINE_SHAPES`.
    """
    rng = np.random.default_rng(SEED)
    seeds = np.array(sorted(sums["seed"].unique().to_list()))
    index = {s: i for i, s in enumerate(seeds)}
    table = np.zeros((len(HEADLINE_SHAPES), len(seeds), 3))
    for row in sums.filter(pl.col("shape").is_in(HEADLINE_SHAPES)).iter_rows(named=True):
        table[HEADLINE_SHAPES.index(row["shape"]), index[row["seed"]]] = (
            row["n"],
            row[f"sse_{arm}"],
            row[f"sse_{versus}"],
        )
    draws = rng.integers(0, len(seeds), size=(N_RESAMPLES, len(seeds)))
    picked = table[:, draws, :].sum(axis=2)  # (shape, resample, 3)
    score_arm = (picked[..., 1] / picked[..., 0]).mean(axis=0)
    score_versus = (picked[..., 2] / picked[..., 0]).mean(axis=0)
    change = score_arm / score_versus - 1
    total = table.sum(axis=1)
    point = float(((total[:, 1] / total[:, 0]).mean()) / ((total[:, 2] / total[:, 0]).mean()) - 1)
    low, high = np.quantile(change, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return {"change": point, "low": float(low), "high": float(high)}


def report_variant(*, injected: pl.DataFrame, name: str) -> dict[str, np.ndarray]:
    """Fit on odd months, score on even months, and print one variant's report."""
    fit, test = injected.filter(pl.col("fit")), injected.filter(~pl.col("fit"))
    print(f"\n######## variant {name}")
    print(
        f"rows: fit {fit.height:,} (positives {int(fit['y'].sum()):,}), "
        f"test {test.height:,} (positives {int(test['y'].sum()):,})"
    )
    fitted = {arm: fit_arm(frame=fit, arm=arm) for arm in ARMS}
    (OUT_DIR / f"fitted_{name}.pkl").write_bytes(pickle.dumps(fitted))
    for arm in ("latest", "latest_x1", "best"):
        coef = dict(zip(FEATURES[arm], fitted[arm].logistic.coef_[0].round(3), strict=True))  # ty: ignore[unresolved-attribute]
        print(arm, "logistic coefficients", coef)
    print("flag hit rates (no flag, flag):", fitted["flag"].flag_rates)
    predictions = {arm: predict(fitted=fitted[arm], frame=test) for arm in ARMS}
    predictions["flag_raw"] = test["flag"].to_numpy().astype(float)
    y = test["y"].to_numpy().astype(float)
    head = test["shape"].is_in(HEADLINE_SHAPES).to_numpy()
    print("\n-- even months, headline shapes S2 S3 S4 S6 pooled by rows")
    for arm, p in predictions.items():
        rel = reliability(p=p[head], y=y[head]).filter(pl.col("share") >= BIN_SHARE)
        print(
            f"{arm:10s} Brier {brier(p=p[head], y=y[head]):.5f}"
            f"  AUC {auc(p=p[head], y=y[head]):.4f}"
            f"  max gap in bins holding >=5% of rows {rel['gap'].max():.3f}"
        )
    for arm in ("flag", "latest", "latest_x1"):
        print(f"\nreliability {arm} (headline shapes)")
        print(reliability(p=predictions[arm][head], y=y[head]))
    sums = cluster_sums(frame=test, predictions=predictions)
    print("\n-- planned contrasts: relative change in Brier score against arm 1 (97.5%)")
    for arm in ("latest", "latest_x1", "best"):
        print(arm, bootstrap_contrast(sums=sums, arm=arm))
    print(
        "latest_x1 against latest", bootstrap_contrast(sums=sums, arm="latest_x1", versus="latest")
    )
    print("\n-- Brier by shape")
    for shape in ALL_SHAPES:
        idx = (test["shape"] == shape).to_numpy()
        print(shape, {a: round(brier(p=predictions[a][idx], y=y[idx]), 5) for a in predictions})
    print(
        "\n-- by hours since onset (covering transfer for positives, newest active transfer else)"
    )
    age = test["age_hours"].to_numpy()
    for label, low, high in (("<24h", 0, 24), ("1-3d", 24, 72), (">3d", 72, 1e9)):
        m = head & (age >= low) & (age < high)
        print(
            label,
            f"rows {m.sum():,} positives {int(y[m].sum()):,}",
            {
                a: (
                    round(brier(p=predictions[a][m], y=y[m]), 5),
                    round(auc(p=predictions[a][m], y=y[m]), 3),
                )
                for a in ("flag", "latest", "latest_x1")
            },
        )
    print("\n-- by month (Brier, arm 2 then arm 1, headline shapes)")
    months = test["row_month"].to_numpy()
    for month in sorted(set(months.tolist())):
        m = head & (months == month)
        print(
            month,
            f"{int(y[m].sum()):,} positives",
            round(brier(p=predictions["latest"][m], y=y[m]), 5),
            round(brier(p=predictions["flag"][m], y=y[m]), 5),
        )
    test.write_parquet(OUT_DIR / f"test_rows_{name}.parquet")
    np.savez(OUT_DIR / f"test_predictions_{name}.npz", **predictions)
    return predictions


def report_durations(*, frame: pl.DataFrame) -> None:
    """Post hoc: fit on odd months of both draws, score by the newest transfer's duration bin.

    Rows are the clean rows (no background flag) of the headline shapes. A row's bin is the
    duration of the newest-onset transfer active at its cut, so the bin holds that transfer's
    positive rows and the negative rows of the same cuts.
    """
    injected = frame.filter(
        pl.col("shape").is_in(HEADLINE_SHAPES) & ~pl.col("background").fill_null(False)
    )
    injected = injected.filter(
        pl.col("ev_step_z").is_null()
        | (pl.col("ev_step_z") >= LARGE_STEP_Z)
        | (pl.col("seed") >= LONG_SEED_FROM)
    )
    injected = add_reference_hours(frame=injected)
    fit, test = injected.filter(pl.col("fit")), injected.filter(~pl.col("fit"))
    print("\n######## post hoc: by duration of the newest active transfer (both draws, clean rows)")
    print(f"rows: fit {fit.height:,}, test {test.height:,}")
    fitted = {arm: fit_arm(frame=fit, arm=arm) for arm in ARMS}
    predictions = {arm: predict(fitted=fitted[arm], frame=test) for arm in ARMS}
    y = test["y"].to_numpy().astype(float)
    hours = test["ref_hours"].to_numpy()
    recall = test["flag"].to_numpy()
    for name, low, high in DURATION_BINS:
        m = np.isfinite(hours) & (hours >= low) & (hours < high)
        print(f"\n-- {name}: rows {m.sum():,}, positives {int(y[m].sum()):,}")
        if y[m].sum() == 0:
            continue
        print(
            f"   flag recall on positives {recall[m & (y > 0)].mean():.3f}, "
            f"flag rate on negatives {recall[m & (y == 0)].mean():.3f}"
        )
        for arm in ("flag", "latest", "latest_x1", "best"):
            p = predictions[arm][m]
            rel = reliability(p=p, y=y[m]).filter(pl.col("share") >= BIN_SHARE)
            print(
                f"   {arm:10s} Brier {brier(p=p, y=y[m]):.5f}  AUC {auc(p=p, y=y[m]):.4f}  "
                f"max gap {rel['gap'].max():.3f}"
            )
        print(reliability(p=predictions["latest"][m], y=y[m]))
    test.select(
        "seed", "shape", "cut", "series", "lag_stamp", "y", "ref_hours", "flag"
    ).write_parquet(OUT_DIR / "test_rows_duration.parquet")
    np.savez(OUT_DIR / "test_predictions_duration.npz", **predictions)


def main() -> None:
    """Score every variant, then the real half-hours with the `all` calibration."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    for name in VARIANTS:
        report_variant(injected=variant_rows(frame=frame, variant=name), name=name)
    report_durations(frame=frame)
    real = frame.filter(pl.col("shape") == "REAL")
    ry = real["y"].to_numpy().astype(float)
    print(
        f"\n######## real half-hours (logs are incomplete, so precision is a lower bound): "
        f"{real.height:,} rows, {int(ry.sum()):,} inside a logged closure or labelled event"
    )
    fitted = pickle.loads((OUT_DIR / "fitted_all.pkl").read_bytes())
    for arm in ARMS:
        p = predict(fitted=fitted[arm], frame=real)
        print(
            f"{arm:10s} Brier {brier(p=p, y=ry):.5f}  AUC {auc(p=p, y=ry):.4f}  "
            f"mean p {p.mean():.4f}  base rate {ry.mean():.4f}"
        )
    flag = real["flag"].to_numpy()
    print(
        f"flag marks {flag.sum():,} rows; {int((flag & (ry > 0)).sum()):,} inside the logs; "
        f"{int(((ry > 0) & ~flag).sum()):,} logged rows unflagged"
    )


if __name__ == "__main__":
    main()
