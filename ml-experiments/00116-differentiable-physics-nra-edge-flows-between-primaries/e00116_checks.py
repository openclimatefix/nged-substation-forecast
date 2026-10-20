"""The plumbing controls of experiment 116, run before any scored contrast is read.

- **P1, P1b, P2** (the path solver against a quadratic programme, block-coordinate descent against
  a joint program, and recovery of a noise-free transfer) are tests of `studies.dp_edge_flow`; this
  script runs them.
- **P3a causality:** changing every residual after the newest usable row leaves A5's corrections
  unchanged.
- **P3b zero flows:** with every flow forced to zero, A5 equals the plain filter exactly.
- **P3c labels:** emptying the switching-event labels leaves arm A0's stored fit unchanged.
- **P3d cut:** changing the power after a fit's cut leaves its forecasts unchanged.
- **P3e noise scales:** the residual and edge scales use rows at or before the cut only.
- **P3f out-of-sample windows:** no origin of the scored year reads an in-sample residual.
- **P4 events:** the hand-filtered events number 66 (35 conserved and 31 one-sided).

    uv run python ml-experiments/00116-*/e00116_checks.py
"""

import subprocess
import sys
from dataclasses import replace
from typing import Any, Final

import e00116_causal as causal
import e00116_common as common
import e00116_data as data_module
import e00116_flows as flows_module
import e00116_gam as gam
import e00116_inject as inject
import e00116_local as local
import e00116_score as score
import numpy as np
import polars as pl
from contracts.settings import PROJECT_ROOT
from studies.dp_edge_flow import fit_survival
from studies.dp_pinned_capacity import read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CHECKS_PATH: Final = common.WORK_DIR / "checks.json"
TOLERANCE: Final[float] = 1e-9


def module_tests() -> bool:
    """Run the module's tests (P1, P1b, P2, and the filter and survival checks)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "packages/studies/tests/test_dp_edge_flow.py",
            "--run-studies",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _transfer_and_context(*, cache: Any) -> tuple[dict, causal.Context, dict, dict, list[int]]:
    """Return one gate transfer, the tuning-year context, its streams, and three origins."""
    config = flows_module.chosen_config()
    survival = fit_survival(lengths=np.array([4, 10, 60, 200]), censored=np.array([False] * 4))
    context = causal.build_context(
        cache=cache, cut_name="tuning", arm="a0", config=config, survival=survival
    )
    transfer = inject.draw_set(set_name="G", year_name="tuning", cache=cache).row(0, named=True)
    streams, added = local.injected_streams(cache=cache, data=context.data, transfer=transfer)
    first = int(transfer["onset"]) - causal.DELAY_HALF_HOURS
    return transfer, context, streams, added, [first, first + 12, first + 24]


def causality(*, cache: Any) -> bool:
    """P3a: nothing after the newest usable row changes A5's corrections."""
    transfer, context, streams, _, cutoffs = _transfer_and_context(cache=cache)
    nodes = (transfer["node_a"], transfer["node_b"])
    before = causal.causal_forecasts(
        context=context, streams_mw=streams, nodes=nodes, cutoffs=cutoffs
    )
    rng = np.random.default_rng(0)
    changed = {}
    for node, stream in streams.items():
        copy = stream.copy()
        copy[cutoffs[-1] + 1 :] += rng.normal(0.0, 5.0, len(copy) - cutoffs[-1] - 1)
        changed[node] = copy
    after = causal.causal_forecasts(
        context=context, streams_mw=changed, nodes=nodes, cutoffs=cutoffs
    )
    return all(
        np.array_equal(b.correction, a.correction)
        for node in nodes
        for b, a in zip(before[node], after[node], strict=True)
    )


def zero_flows(*, cache: Any) -> float:
    """P3b: the largest difference between A5 with zero flows and the plain filter."""
    transfer, context, streams, _, cutoffs = _transfer_and_context(cache=cache)
    stiff = replace(
        context, paths=tuple(replace(p, step_penalty=1e9, level_penalty=1e9) for p in context.paths)
    )
    node = transfer["receiver"]
    a5 = causal.causal_forecasts(
        context=stiff,
        streams_mw=streams,
        nodes=(transfer["node_a"], transfer["node_b"]),
        cutoffs=cutoffs,
    )[node]
    plain = causal.plain_forecasts(context=context, streams_mw=streams, node=node, cutoffs=cutoffs)
    return max(
        float(np.max(np.abs(x.correction - y.correction))) for x, y in zip(a5, plain, strict=True)
    )


def labels_do_not_matter(*, cache: Any) -> float:
    """P3c: arm A0's fit is the same when every switching-event label is emptied."""
    vintages = read_vintages()
    emptied = replace(
        cache,
        sites={
            site: replace(features, switching_mask=np.zeros_like(features.switching_mask))
            for site, features in cache.sites.items()
        },
    )
    cut = common.CUTS["replicate"]
    one = gam.fit_job(cache=cache, vintages=vintages, arm="a0", site="D11", cut=cut)
    two = gam.fit_job(cache=emptied, vintages=vintages, arm="a0", site="D11", cut=cut)
    return float(np.max(np.abs(one["mean"].to_numpy() - two["mean"].to_numpy())))


def cut_is_respected(*, cache: Any) -> float:
    """P3d: changing the power after the cut leaves the fit's forecasts unchanged."""
    vintages = read_vintages()
    cut = common.CUTS["replicate"]
    year = common.year_slice(cache=cache, cut=cut)
    features = cache.sites["D11"]
    y = features.y.copy()
    y[year.start :] += 7.0
    shifted = replace(cache, sites={**cache.sites, "D11": replace(features, y=y)})
    one = gam.fit_job(cache=cache, vintages=vintages, arm="a0", site="D11", cut=cut)
    two = gam.fit_job(cache=shifted, vintages=vintages, arm="a0", site="D11", cut=cut)
    return float(np.max(np.abs(one["mean"].to_numpy() - two["mean"].to_numpy())))


def noise_uses_pre_cut_rows(*, cache: Any) -> float:
    """P3e: the largest change in a scale when every residual after the cut is changed."""
    data = data_module.load_year(cache=cache, cut_name="replicate", arm="a0")
    changed = replace(
        data,
        residual_mw={
            site: np.where(np.arange(len(r)) > data.cut_position, r + 3.0, r)
            for site, r in data.residual_mw.items()
        },
    )
    one, two = data.noise_mw(), changed.noise_mw()
    edge = abs(
        data.edge_scale_mw(node_a="D5", node_b="D16")
        - changed.edge_scale_mw(node_a="D5", node_b="D16")
    )
    return max(*(abs(one[s] - two[s]) for s in one), edge)


def windows_are_out_of_sample(*, cache: Any) -> bool:
    """P3f: the stream the pipelines read holds no value before the scored year."""
    data = data_module.load_year(cache=cache, cut_name="scored", arm="a0")
    return all(
        not np.isfinite(stream[: data.first]).any() for stream in data.out_of_sample().values()
    )


def event_counts() -> dict[str, int]:
    """P4: the hand-filtered events as experiment 101 filters them."""
    frame = (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
        )
        .filter(~pl.col("is_sentinel"), pl.col("kind") == "real_operation")
        .filter(pl.col("visibility").is_in(["conserved", "one_sided"]))
    )
    return {
        "events": frame.height,
        "conserved": int((frame["visibility"] == "conserved").sum()),
        "one_sided": int((frame["visibility"] == "one_sided").sum()),
    }


def main() -> None:
    """Run every control and write `checks.json`."""
    cache = common.load_cache()
    results = {
        "module_tests_pass": module_tests(),
        "P3a_causal": causality(cache=cache),
        "P3b_zero_flow_difference": zero_flows(cache=cache),
        "P3c_label_difference": labels_do_not_matter(cache=cache),
        "P3d_cut_difference": cut_is_respected(cache=cache),
        "P3e_scale_difference": noise_uses_pre_cut_rows(cache=cache),
        "P3f_out_of_sample": windows_are_out_of_sample(cache=cache),
        "P4_events": event_counts(),
    }
    results["all_pass"] = bool(
        results["module_tests_pass"]
        and results["P3a_causal"]
        and results["P3b_zero_flow_difference"] < TOLERANCE
        and results["P3c_label_difference"] < TOLERANCE
        and results["P3d_cut_difference"] < TOLERANCE
        and results["P3e_scale_difference"] < TOLERANCE
        and results["P3f_out_of_sample"]
        and results["P4_events"] == {"events": 66, "conserved": 35, "one_sided": 31}
    )
    score.write_json(path=CHECKS_PATH, payload=results)
    for name, value in results.items():
        print(name, value)


if __name__ == "__main__":
    main()
