"""Known-answer cards (injected transfers, placebo windows, jittered logs) and card rendering."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from e00041_common import MVA_SERIES, WORK_DIR
from e00041_features import CardSpec, Features
from e00041_panel import (
    ADJACENCY,
    INDEX,
    N_STAMPS,
    WEEK,
    Panel,
    first_affected,
    last_affected,
    reference,
    time_of,
)

INTERVALS_00037: Final = WORK_DIR.parent / "00037-switching-first-look" / "intervals.parquet"
LOG_END_STAMP: Final[int] = 4 * 365 * 48 + 85 * 48  # about 2026-03-25 on the grid
QUIET_STAMPS: Final[int] = 7 * 48
P_UNMETERED: Final[float] = 0.15


@dataclass
class Truth:
    """The answer behind a known-answer card; never shown to a labeller."""

    card_id: str
    kind: str
    onset_stamp: int | None = None
    end_stamp: int | None = None
    donor: int | None = None
    receivers: list[dict] | None = None
    unmetered_receiver: bool = False
    moved_mw_p50: float | None = None
    moved_mw_p10: float | None = None
    moved_mw_p90: float | None = None
    moved_share_of_receiver: float | None = None
    log_jitter_hours: float = 0.0


def real_operations() -> pl.DataFrame:
    """Real closed operations of 2 hours to 30 days since 2022, one row each, from experiment 37."""
    frame = pl.read_parquet(INTERVALS_00037).filter(pl.col("hours").is_between(2, 30 * 24))
    return (
        frame.with_columns(
            a=pl.col("side_a").cast(pl.Int64, strict=False),
            b=pl.col("side_b").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("a").is_not_null())
        .unique(subset=["operation"], keep="first")
    )


QUIET_RULE: dict[str, bool] = {"relaxed": False}
"""`relaxed` keeps only the event's own surroundings free of logged intervals (SW2-0b sentinels)."""


def is_quiet(*, panel: Panel, rows: list[int], first: int, last: int) -> bool:
    """True if the rows are unexcluded and mostly observed around the window."""
    relaxed = QUIET_RULE["relaxed"]
    pad = 2 * 48 if relaxed else QUIET_STAMPS
    lo, hi = first - pad - (0 if relaxed else 4 * WEEK), last + pad
    if lo < (4 * WEEK if relaxed else 0) or hi >= min(N_STAMPS, LOG_END_STAMP):
        return False
    for row in rows:
        if panel.excluded[row, (lo if relaxed else lo + 4 * WEEK) : hi].any():
            return False
        if np.isnan(panel.obs[row, first : last + 1]).mean() > (0.8 if relaxed else 0.2):
            return False
    return True


def draw_quiet_start(
    *, panel: Panel, rows: list[int], length: int, rng: np.random.Generator
) -> int | None:
    """Draw a start stamp whose window is quiet on every row, or None."""
    for _ in range(400):
        first = int(rng.integers(5 * WEEK, LOG_END_STAMP - length - QUIET_STAMPS))
        if is_quiet(panel=panel, rows=rows, first=first, last=first + length - 1):
            return first
    return None


def _walk(*, length: int, rng: np.random.Generator) -> np.ndarray:
    """A slow random walk in [0.5, 2] times one, reflected at both ends."""
    log = np.zeros(length)
    for i in range(1, length):
        log[i] = log[i - 1] + rng.normal(0, 0.03)
        low, high = np.log(0.5), np.log(2.0)
        if log[i] > high:
            log[i] = 2 * high - log[i]
        if log[i] < low:
            log[i] = 2 * low - log[i]
    return np.exp(np.clip(log, np.log(0.5), np.log(2.0)))


def _quiet_shift(
    *, panel: Panel, op: dict, rows: list[int], rng: np.random.Generator
) -> tuple[int, int, int] | None:
    """Shift a real operation by whole weeks into a quiet period; return `(weeks, first, last)`."""
    for _ in range(300):
        shift_weeks = int(rng.integers(-60, 60))
        close_utc = op["start_utc"] + timedelta(days=7 * shift_weeks)
        open_utc = op["end_utc"] + timedelta(days=7 * shift_weeks)
        first, last = first_affected(close_utc=close_utc), last_affected(open_utc=open_utc)
        if (
            shift_weeks != 0
            and last > first + 2
            and is_quiet(panel=panel, rows=rows, first=first, last=last)
        ):
            return shift_weeks, first, last
    return None


def _receivers_for(
    *, donor: int, other: list[int], n_receivers: int, rng: np.random.Generator
) -> tuple[list[int], bool]:
    """Pick the metered receivers; a sheet with an unmetered side sends all load to it."""
    if not other:
        return [], True
    receivers = list(other)
    candidates = [n for n in ADJACENCY[donor] if n not in receivers]
    rng.shuffle(candidates)
    while len(receivers) < n_receivers and candidates:
        receivers.append(candidates.pop())
    return receivers, False


def make_injection(
    *, panel: Panel, ops: pl.DataFrame, rng: np.random.Generator, jitter: bool
) -> tuple[CardSpec, Truth, dict] | None:
    """Draw one injected transfer on a real operation's log times, shifted by whole weeks.

    The log times, the sheet's sides (including an unmetered side B), the clock time and the
    duration are the real operation's, exactly as in `make_placebo`, so a card carries no tell
    that separates it from a placebo card.
    """
    op = ops.row(int(rng.integers(0, ops.height)), named=True)
    pair = [int(op["a"])] + ([int(op["b"])] if op["b"] is not None else [])
    if any(p not in INDEX for p in pair):
        return None
    donor = int(rng.choice(pair))
    other = [p for p in pair if p != donor]
    receivers, unmetered = _receivers_for(
        donor=donor, other=other, n_receivers=int(rng.integers(1, 4)), rng=rng
    )
    rows = sorted({INDEX[donor]} | {INDEX[r] for r in receivers} | {INDEX[p] for p in pair})
    shifted = _quiet_shift(panel=panel, op=op, rows=rows, rng=rng)
    if shifted is None:
        return None
    shift_weeks, first, last = shifted
    length = last - first + 1
    stamps = np.arange(first, last + 1)
    donor_obs = panel.obs[INDEX[donor], first : last + 1].copy()
    recv_ref = np.array(
        [np.nanmean(reference(panel=panel, row=INDEX[r], stamps=stamps)) for r in receivers]
    )
    donor_ref = np.nanmean(reference(panel=panel, row=INDEX[donor], stamps=stamps))
    fraction = float(rng.uniform(0.05, 0.40))
    target_mw = fraction * (np.nansum(recv_ref) if receivers else donor_ref * 0.5)
    s0 = min(0.8, target_mw / donor_ref) if donor_ref > 0 else 0.0
    moved = np.minimum(s0 * _walk(length=length, rng=rng), 0.8) * np.nan_to_num(
        donor_obs, nan=donor_ref
    )
    weights = rng.dirichlet(np.ones(len(receivers) + (1 if unmetered or rng.random() < 0.1 else 0)))
    recv_weights = weights[: len(receivers)]
    close_utc = op["start_utc"] + timedelta(days=7 * shift_weeks)
    open_utc = op["end_utc"] + timedelta(days=7 * shift_weeks)
    jitter_hours = 0.0
    if jitter:
        jitter_hours = float(rng.choice([-1, 1]) * rng.uniform(1, 3))
        close_utc += timedelta(hours=jitter_hours)
        open_utc += timedelta(hours=float(rng.choice([-1, 1]) * rng.uniform(1, 3)))
    card_id = secrets.token_hex(5)
    spec = CardSpec(
        card_id=card_id,
        side_a=pair[0],
        side_b=pair[1] if len(pair) > 1 else None,
        close_utc=close_utc,
        open_utc=open_utc,
    )
    inj = {
        "first": first,
        "last": last,
        "donor": donor,
        "receivers": receivers,
        "weights": recv_weights,
        "moved": moved,
    }
    share = (
        float(np.nanmedian(moved) / max(np.nansum(recv_ref), 1e-9))
        if receivers
        else float(np.nanmedian(moved) / max(0.5 * donor_ref, 1e-9))
    )
    truth = Truth(
        card_id=card_id,
        kind="jittered" if jitter else "injected",
        onset_stamp=first,
        end_stamp=last,
        donor=donor,
        receivers=[
            {"series": r, "share": float(w)} for r, w in zip(receivers, recv_weights, strict=True)
        ],
        unmetered_receiver=bool(unmetered or recv_weights.sum() < 0.999),
        moved_mw_p50=float(np.percentile(moved, 50)),
        moved_mw_p10=float(np.percentile(moved, 10)),
        moved_mw_p90=float(np.percentile(moved, 90)),
        moved_share_of_receiver=share,
        log_jitter_hours=jitter_hours,
    )
    return spec, truth, inj


def apply_injection(*, panel: Panel, inj: dict) -> None:
    """Move the injected load from the donor to the receivers on `panel`."""
    first, last = inj["first"], inj["last"]
    panel.obs[INDEX[inj["donor"]], first : last + 1] -= inj["moved"]
    for r, w in zip(inj["receivers"], inj["weights"], strict=True):
        panel.obs[INDEX[r], first : last + 1] += w * inj["moved"]


def revert_injection(*, panel: Panel, inj: dict) -> None:
    """Undo `apply_injection` on `panel`."""
    first, last = inj["first"], inj["last"]
    panel.obs[INDEX[inj["donor"]], first : last + 1] += inj["moved"]
    for r, w in zip(inj["receivers"], inj["weights"], strict=True):
        panel.obs[INDEX[r], first : last + 1] -= w * inj["moved"]


def make_placebo(
    *, panel: Panel, ops: pl.DataFrame, rng: np.random.Generator
) -> tuple[CardSpec, Truth] | None:
    """A real operation's sheet pair and clock time, shifted by whole weeks into a quiet period."""
    op = ops.row(int(rng.integers(0, ops.height)), named=True)
    sides = [int(op["a"])] + ([int(op["b"])] if op["b"] is not None else [])
    if any(s not in INDEX for s in sides):
        return None
    shifted = _quiet_shift(panel=panel, op=op, rows=[INDEX[s] for s in sides], rng=rng)
    if shifted is None:
        return None
    shift_weeks = shifted[0]
    card_id = secrets.token_hex(5)
    spec = CardSpec(
        card_id=card_id,
        side_a=sides[0],
        side_b=sides[1] if len(sides) > 1 else None,
        close_utc=op["start_utc"] + timedelta(days=7 * shift_weeks),
        open_utc=op["end_utc"] + timedelta(days=7 * shift_weeks),
    )
    return spec, Truth(card_id=card_id, kind="placebo")


def render_card(*, spec: CardSpec, features: Features, panel: Panel, note: str = "") -> str:
    """Render one card as markdown for a labeller: log times, features, and an hourly table."""
    v = features.values
    lines = [f"# Card {spec.card_id}", ""]

    def local(t: datetime) -> str:
        return t.astimezone(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M")

    lines += [
        (
            f"- Logged close (UK local): {local(spec.close_utc)}; logged open (UK local): "
            f"{local(spec.open_utc)}"
        ),
        (
            f"- First affected stamp: {spec.first}; last affected stamp: {spec.last} (half-hour "
            f"stamps end at their time; stamp {spec.first} ends "
            f"{time_of(stamp=spec.first):%Y-%m-%d %H:%M} UTC)"
        ),
        (
            f"- Side A: series {spec.side_a} "
            f"({'MVA, absolute value' if spec.side_a in MVA_SERIES else 'MW'}); "
        )
        + (
            (
                f"side B: series {spec.side_b} "
                f"({'MVA, absolute value' if spec.side_b in MVA_SERIES else 'MW'})"
            )
            if spec.side_b in INDEX
            else "side B: not metered"
        ),
        (
            f"- Step window: {int(v['window_half_hours'])} half-hours either side of the close "
            "and the open"
        ),
        "",
        "## Step features (MW; percentile against 400 quiet-week draws at the same clock time)",
        "",
        "| Feature | Side A | Side B |",
        "|---|---|---|",
    ]

    def f(x: float | str | None, d: int = 2) -> str:
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

    for name, label in (
        ("close_step", "F1 step at close"),
        ("open_step", "F2 step at open (expect reverse sign)"),
    ):
        lines.append(
            f"| {label} (MW) | {f(v.get(name + '_a_mw', float('nan')))} | "
            f"{f(v.get(name + '_b_mw', float('nan')))} |"
        )
        lines.append(
            f"| {label} percentile | {f(v.get(name + '_a_pct', float('nan')))} | "
            f"{f(v.get(name + '_b_pct', float('nan')))} |"
        )
    lines.append(
        f"| Quiet-week SD of step (MW) | {f(v.get('null_sd_a_mw', float('nan')))} | "
        f"{f(v.get('null_sd_b_mw', float('nan')))} |"
    )
    lines.append(
        f"| Reference mean over event (MW) | {f(v.get('ref_mean_a_mw', float('nan')))} | "
        f"{f(v.get('ref_mean_b_mw', float('nan')))} |"
    )
    lines.append(
        "| F7 hindsight z, after minus before close | "
        f"{f(v.get('hindsight_z_a', float('nan')))} | "
        f"{f(v.get('hindsight_z_b', float('nan')))} |"
    )
    lines.append(
        "| F7 hindsight mean z inside event | "
        f"{f(v.get('hindsight_mean_z_in_event_a', float('nan')))} | "
        f"{f(v.get('hindsight_mean_z_in_event_b', float('nan')))} |"
    )
    lines += [
        "",
        (
            "## F3, F4: donor to other series (excess = observed minus reference; transfer = (y "
            "- x)/2, common = (x + y)/2)"
        ),
        "",
        (
            "| Donor | Other | Other unit | Donor step | Other step | Other step pct | Mean "
            "transfer | Mean common | Slope y on -x | R2 | Shape share | Shape R2 | Share ratio "
            "2nd/1st half |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in features.neighbours:
        lines.append(
            f"| {n['donor']} | {n['other']} | {n['other_unit']} | "
            f"{f(n['donor_close_step_mw'])} | {f(n['other_close_step_mw'])} | "
            f"{f(n['other_close_step_pct'])} | "
            f"{f(n['mean_transfer_mw'])} | {f(n['mean_common_mw'])} | "
            f"{f(n['slope_y_on_minus_x'])} | {f(n['r2_y_on_minus_x'])} | "
            f"{f(n['shape_share'], 3)} | {f(n['shape_r2'])} | "
            f"{f(n['share_ratio_second_over_first'])} |"
        )
    lines += [
        "",
        (
            "- F5 step in the sum of side A and its metered neighbours: "
            f"{f(v['nbhd_sum_step_mw'])} MW; largest single step {f(v['nbhd_largest_step_mw'])} "
            f"MW; sum over largest {f(v['nbhd_sum_over_largest'])}"
        ),
        (
            "- F6 share of the other series with a same-sign step beyond 2 quiet SDs: "
            f"{f(v['fleet_same_sign_share'])}"
        ),
    ]
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"
