# ruff: noqa: E501
"""Print every number the write-up of experiment 354 quotes, and save it to `work/report.txt`.

uv run --no-sync python ml-experiments/00354-*/e00354_report.py
"""

import json
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00354-d8-own-residual-bands-and-longer-fit-history" / "work"
)


def fmt(values: list[float]) -> str:
    """Return `point (low to high)` to two decimals."""
    return f"{values[0]:+.2f} ({values[1]:+.2f} to {values[2]:+.2f})"


def main() -> None:
    """Print the report."""
    bands = json.loads((WORK / "bands_scores.json").read_text())
    fits = json.loads((WORK / "fits_scores.json").read_text())
    lines: list[str] = []
    d8 = bands["D8"]
    lines.append(f"D8 rows {d8['rows']} capacity {d8['capacity_mw']:.2f} MW")
    for flag, values in d8["absolute"].items():
        lines.append(f"abs {flag}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))
    for column, by_flag in d8["contrast"].items():
        lines.append(
            f"contrast {column}: "
            + "; ".join(
                f"{f} 99% {fmt(v['ci99'])} 95% {fmt(v['ci95'])}" for f, v in by_flag.items()
            )
        )
    for month, v in d8["closure_by_month"].items():
        share = v["closed_share_arm2"]
        lines.append(
            f"closure {month}: gap {v['gap_arm1_minus_bg2']:.2f} arm2-bg2 {v['arm2_minus_bg2']:.2f} "
            f"closed {'n/a' if share is None else f'{share:.2f}'}"
        )
    for arm, quarters in d8["coverage_by_quarter"].items():
        lines.append(
            f"coverage {arm}: "
            + ", ".join(f"{q} {v['inside_q05_q95']:.3f}" for q, v in quarters.items())
        )
    for arm, tails in d8["tails_all"].items():
        lines.append(f"tails {arm}: " + ", ".join(f"{k} {v:.3f}" for k, v in tails.items()))
    lines.append(f"checks D8: {json.dumps(d8['checks'])}")
    for site in ("D2", "D3", "D4"):
        s = bands[site]
        lines.append(
            f"{site} rows {s['rows']} abs all: "
            + ", ".join(f"{k} {v:.3f}" for k, v in s["absolute"]["all"].items())
        )
        lines.extend(
            f"{site} {column} all 99% {fmt(s['contrast'][column]['all']['ci99'])} 95% {fmt(s['contrast'][column]['all']['ci95'])}"
            for column in ("c_arm2_arm1", "c_arm3_arm1")
        )
        lines.append(
            f"{site} below q01/q05: "
            + ", ".join(
                f"{a} {t['below_q01']:.3f}/{t['below_q05']:.3f}" for a, t in s["tails_all"].items()
            )
        )
        lines.append(
            f"{site} coverage inside q05-q95: "
            + ", ".join(f"{a} {t['inside_q05_q95']:.3f}" for a, t in s["tails_all"].items())
        )
    lines.append(f"fits rows {fits['rows']}")
    for flag, values in fits["absolute"].items():
        lines.append(f"fits abs {flag}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))
    for column, by_flag in fits["contrast"].items():
        lines.append(
            f"fits contrast {column}: "
            + "; ".join(f"{f} 99% {fmt(v['ci99'])}" for f, v in by_flag.items())
        )
    for name in fits["tails_all"]:
        t = fits["tails_all"][name]
        lines.append(
            f"fits tails {name}: "
            + ", ".join(f"{k} {v:.3f}" for k, v in t.items())
            + f" | cov by quarter {fits['coverage_by_quarter'][name]}"
        )
    lines.append(f"fits checks: {json.dumps(fits['checks'])}")
    for month, v in fits["monthly"].items():
        lines.append(
            f"fits monthly {month}: "
            + ", ".join(
                f"{k} {x:.2f}"
                for k, x in v.items()
                if k.endswith(("d_arm3", "d_donated")) or k == "bg2"
            )
        )
    lines.append(
        f"step levels MW per hour band by cut: {json.dumps(fits['step_levels_mw_by_cut'])}"
    )
    text = "\n".join(lines)
    (WORK / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
