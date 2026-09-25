"""Part C: refit the ENS-horizons page's arms under D0, a trimmed D0, an IFS 49r1 era cut, and a Dec-2024-onwards row set.

Usage: partC_fit.py {wind,solar} {D0,D0trim,I,II} [member_days...]
The frame is rebuilt in memory with the study script's own functions from the saved ENS members;
nothing is written under data/.
"""
import logging
import sys
import time
from datetime import UTC, datetime

from common import *  # noqa: F403
import ens_forecast_horizons as eh
from blend_products import SOLAR, WIND
from studies.cross_validation import (
    calendar_month_coverage, cut_eras, raise_on_uncovered_months, rotate_folds, search_fold_offsets,
)

domain_name, design = sys.argv[1], sys.argv[2]
extra_only = "--extra-only" in sys.argv
sens = "--sens" in sys.argv or extra_only
member_days = [int(x) for x in sys.argv[3:] if not x.startswith("--")]
if extra_only:
    member_days = []
domain = SOLAR if domain_name == "solar" else WIND
logging.basicConfig(level=logging.INFO)
eh.MAX_CONCURRENT_FITS = 4
import os
if os.environ.get("DEVICE") == "cuda":
    import studies.cross_validation as _cv
    _orig = _cv.booster_parameters
    _cv.booster_parameters = lambda **kw: {**_orig(**kw), "device": "cuda"}
    print("device cuda", flush=True)

IFS_START = datetime(2024, 11, 12, tzinfo=UTC)
FIRST_WHOLE = datetime(2024, 12, 1, tzinfo=UTC)
paths = eh._paths(domain=domain_name)
saved_losses = pl.read_parquet(paths["losses"])
method, _ = eh.choose_method(losses=saved_losses, domain=domain_name)
print("method", method, flush=True)

inputs = eh.build_inputs(domain=domain_name)
inputs = eh.main_frame(inputs=inputs, method=method, domain=domain_name)
frame = inputs.frame
base_columns = [c for c in frame.columns if c not in ("era", "era_code", "fold")]
print("rows D0", frame.height, flush=True)

offsets_info = None
if design == "D0":
    pass
elif design == "D0trim":
    kept = frame.select(base_columns).filter(~pl.col("time").is_between(IFS_START, FIRST_WHOLE, closed="left"))
    frame = with_eras(frame=kept.sort("site", "time")).with_columns(month_of_year=pl.col("time").dt.month())
elif design == "I":
    kept = frame.select(base_columns).filter(~pl.col("time").is_between(IFS_START, FIRST_WHOLE, closed="left")).sort("site", "time")
    firsts = ["2024-12", UKV_UPGRADE_MONTH]
    found = search_fold_offsets(frame=kept, first_months=firsts)
    print("offsets found", [dict(o) for o in found][:10], "n", len(found), flush=True)
    offsets = dict(found[0]) if found else {0: 0, 1: 0, 2: 0}
    offsets_info = {"n_found": len(found), "used": offsets}
    frame = cut_eras(frame=kept, first_months=firsts, fold_offsets=offsets).with_columns(
        month_of_year=pl.col("time").dt.month()
    )
elif design == "I2":
    kept = frame.select(base_columns).filter(~pl.col("time").is_between(IFS_START, FIRST_WHOLE, closed="left")).sort("site", "time")
    firsts = ["2024-12", UKV_UPGRADE_MONTH]
    found = search_fold_offsets(frame=kept, first_months=firsts)
    offsets = dict(found[1])
    offsets_info = {"n_found": len(found), "used": offsets}
    frame = cut_eras(frame=kept, first_months=firsts, fold_offsets=offsets).with_columns(
        month_of_year=pl.col("time").dt.month()
    )
elif design == "IIr":
    kept = frame.select(base_columns).filter(pl.col("time") >= FIRST_WHOLE).sort("site", "time")
    frame = rotate_folds(frame=with_eras(frame=kept), fold_offsets={0: 0, 1: 2}).with_columns(
        month_of_year=pl.col("time").dt.month()
    )
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=frame))
elif design == "II":
    kept = frame.select(base_columns).filter(pl.col("time") >= FIRST_WHOLE).sort("site", "time")
    frame = with_eras(frame=kept).with_columns(month_of_year=pl.col("time").dt.month())
else:
    raise SystemExit("bad design")
print("rows", design, frame.height, frame["fold"].value_counts().sort("fold").to_dicts(),
      frame.group_by("era_code").len().sort("era_code").to_dicts(), flush=True)
print("offsets", offsets_info, flush=True)

SENS_ARMS = {eh.ens_arm(way="mean", day=d) for d in (0, 1, 7)} | {eh.ens_arm(way="control", day=1), eh.ens_arm(way="control", day=0), "era5", "live_gb_rich_xgb"}
import json as _json
_extra = OUT / "sens_extra_contrasts.json"
if _extra.exists():
    for _d, _t, _r in _json.loads(_extra.read_text()):
        if _d == domain_name:
            SENS_ARMS |= {a for a in (_t, _r) if not a.startswith(("climatology", "persistence", "diurnal_persistence", "smart_persistence"))}
BASE_SENS = set(SENS_ARMS)
if extra_only:
    SENS_ARMS = set()
    for _d, _t, _r in _json.loads(_extra.read_text()):
        if _d == domain_name:
            SENS_ARMS |= {a for a in (_t, _r) if not a.startswith(("climatology", "persistence", "diurnal_persistence", "smart_persistence"))}
    _existing = OUT / f"losses_ens_{domain_name}_{design}_sens.parquet"
    if _existing.exists():
        SENS_ARMS -= set(pl.read_parquet(_existing, columns=["arm"])["arm"].unique().to_list())
    SENS_ARMS -= {eh.ens_arm(way="mean", day=d) for d in (0, 1, 7)} | {eh.ens_arm(way="control", day=1), eh.ens_arm(way="control", day=0), "era5", "live_gb_rich_xgb"}
if sens:
    feats = eh.fitted_features(domain=domain)
    pooled_jobs = [(a, "sensitivity", "power_mw", feats[a], eh.SETTINGS["sensitivity"], False) for a in sorted(SENS_ARMS) if a in feats]
else:
    pooled_jobs = [j for j in eh.jobs(domain=domain) if j[1] == "pooled"]
print("arms", len(pooled_jobs), flush=True)
if not pooled_jobs:
    raise SystemExit("nothing to fit")
t = time.time()
keep = ["site", "time", "month", "fold", "seed", "arm", "setting", eh.METRIC, "signed_error_capped_mw"]
parts = [eh.run_all(dataset=frame, jobs=pooled_jobs, max_workers=4).select(keep)]
print("fitted", time.time() - t, flush=True)
if not sens:
    baselines, _ = eh._baseline_losses(frame=frame, domain=domain_name)
    parts.append(baselines.filter(pl.col("setting") == "pooled").select(keep))
for day in member_days:
    t = time.time()
    losses, _ = eh._fit_members(frame=frame, members=inputs.members[day], day=day, domain=domain, setting="sensitivity" if sens else "pooled")
    parts.append(losses.select(keep))
    print("members day", day, time.time() - t, flush=True)
all_losses = pl.concat(parts).with_columns(design=pl.lit(design), domain=pl.lit(domain_name))
if not sens:
    eh.check_shared_rows(losses=all_losses)
all_losses.write_parquet(OUT / f"losses_ens_{domain_name}_{design}{'_sens_extra' if extra_only else '_sens' if sens else ''}.parquet")
if not sens:
    frame.select("site", "time", "month", "fold", "era_code").write_parquet(OUT / f"folds_ens_{domain_name}_{design}.parquet")
print("done", all_losses.height, flush=True)
