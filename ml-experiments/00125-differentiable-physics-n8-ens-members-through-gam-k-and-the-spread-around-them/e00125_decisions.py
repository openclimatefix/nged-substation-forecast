"""Fix the cold-snap rule from ERA5 alone and hash it, before any score is read.

A cold day is a local calendar day whose ERA5 daily-mean 2 m temperature at the site's grid box is
at or below the 10th percentile of that site's November to March daily means over 2019-09-01 to
2024-06-30. A cold snap is a run of two or more consecutive cold days. The file written here holds
each site's threshold, its cold days, and its cold-snap episodes in the replicate months and the
scored year, with a hash of the content.

    uv run python ml-experiments/00125-*/e00125_decisions.py
"""

import hashlib
import json
from datetime import UTC, datetime

import e00125_common as common
import numpy as np

RULE_START = np.datetime64("2019-09-01")
RULE_END = np.datetime64("2024-06-30")
WINTER_MONTHS = (11, 12, 1, 2, 3)
MIN_SNAP_DAYS = 2


def site_decisions(*, site: str) -> dict:
    """Return one site's threshold, cold days, and cold-snap episodes."""
    cache = common.load_cache()
    features = cache.sites[site]
    local_date = cache.calendar.local_date.astype("datetime64[D]")
    days = np.unique(local_date)
    daily = np.array([np.nanmean(features.temperature_c[local_date == day]) for day in days])
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    rule = (days >= RULE_START) & (days <= RULE_END) & np.isin(months, WINTER_MONTHS)
    threshold = float(np.nanpercentile(daily[rule], 10))
    cold = (daily <= threshold) & np.isfinite(daily) & np.isin(months, WINTER_MONTHS)
    episodes, start = [], None
    for index in range(len(days) + 1):
        is_cold = index < len(days) and cold[index]
        if is_cold and start is None:
            start = index
        elif not is_cold and start is not None:
            if index - start >= MIN_SNAP_DAYS and days[index - 1] > RULE_END:
                episodes.append([str(days[start]), str(days[index - 1])])
            start = None
    return {
        "threshold_c": threshold,
        "n_rule_days": int(rule.sum()),
        "cold_days_after_rule_window": [str(day) for day in days[cold & (days > RULE_END)]],
        "episodes_after_rule_window": episodes,
    }


def main() -> None:
    """Write `work/decisions.json`, refusing to overwrite it."""
    path = common.WORK_DIR / "decisions.json"
    if path.exists():
        msg = f"{path} exists; the rule is fixed once"
        raise FileExistsError(msg)
    content = {site: site_decisions(site=site) for site in common.SCORED_SITES}
    text = json.dumps(content, indent=2, sort_keys=True)
    content["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    content["written"] = datetime.now(tz=UTC).date().isoformat()
    common.write_json_atomic(path=path, content=content)
    for site in common.SCORED_SITES:
        item = content[site]
        print(
            site, round(item["threshold_c"], 2), len(item["episodes_after_rule_window"]), "episodes"
        )
    print(content["sha256"])


if __name__ == "__main__":
    main()
