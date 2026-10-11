"""Where the effective-capacity study keeps its files, and the loaders every script shares.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.

**A smoke run writes to its own directory and can never be mistaken for a real run.** `claim_root`
writes a `RUN_KIND.txt` into the output root on first use, naming it `smoke` or `real`, and raises
if a later run of the other kind points at the same root. A smoke root is `<study
directory>/_smoke`, and a real run refuses any root whose path holds `_smoke`.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np
from studies.capacity_estimators import CapacityRecord, RawEstimate
from studies.capacity_recovery import PlantedEvent, calendar_month_spans, plant_losses
from studies.sources import study_dir_for

import studies as studies_package

STUDY_DIR: Final[Path] = study_dir_for(study="effective_capacity")
"""The study's folder under `data/studies/per_study/`."""

SMOKE_DIR_NAME: Final[str] = "_smoke"
"""The directory under the study's folder that holds smoke runs."""

LAG_DIR: Final[Path] = study_dir_for(study="lag_features") / "ens_mean"
"""The lag-features study's outputs, which this study reads and never writes."""

LAG_FRAME_PATH: Final[Path] = LAG_DIR / "lag_frame_ens_mean_day1.parquet"
"""The 29,445 lead-day 1 rows the forecast contrast scores."""

LAG_FRAME_ROWS: Final[int] = 29445
"""The row count the lag study published."""

LAG_LOSSES_PATH: Final[Path] = LAG_DIR / "losses_ens_mean.parquet"
"""The lag study's per-row losses, from which B0 is reused."""

LAG_FOLLOWUPS2_LOSSES_PATH: Final[Path] = (
    LAG_DIR / "followups2" / "losses_followups2_ens_mean.parquet"
)
"""The lag study's round-2 losses, which hold B0 without day of year."""

OUTPUT_RATIOS_PATH: Final[Path] = LAG_DIR / "followups3" / "output_ratios_ens_mean.parquet"
"""Each plant-month's output over its neighbours'."""

DERATED_PLAN_PATH: Final[Path] = LAG_DIR / "followups3" / "derated_plan.json"
"""Which plant and months the lag study found producing less than its neighbours."""

DRAW_SEEDS: Final[tuple[int, ...]] = tuple(11430 + index for index in range(11))
"""One seed per draw; draw 0 is the tuning draw and draws 1 to 10 are the selection draws."""

CONTROL_SEED: Final[int] = 114300
"""The seed of the positive control's months."""

FALSE_MOVEMENT_SEED: Final[int] = 114301
"""The seed that chooses the quiet plant-months."""

FALSE_MOVEMENTS_PER_PLANT: Final[int] = 4
"""The most quiet plant-months chosen at each plant."""

CONTROL_SIZE: Final[float] = 0.30
"""The positive control's step."""

CONTROL_MONTHS_PER_PLANT: Final[int] = 3
"""How many scored months the positive control plants at each plant."""

DERANGEMENT_SEED: Final[int] = 114302
"""The seed of the negative control's derangement."""

ORIGINAL: Final[str] = "original"
CONTROL: Final[str] = "control"
"""Record identifiers besides `draw00` to `draw10`."""


def draw_id(*, index: int) -> str:
    """Return the record identifier of one draw.

    Args:
        index: 0 for the tuning draw, 1 to 10 for the selection draws.

    Returns:
        `draw00`, `draw01`, and so on.
    """
    return f"draw{index:02d}"


def claim_root(*, smoke: bool) -> Path:
    """Return the output root for a run and mark it as real or smoke.

    Args:
        smoke: Whether this is a smoke run.

    Returns:
        The root directory, created if need be.

    Raises:
        RuntimeError: If the root was claimed by the other kind of run, or a real run is pointed at
            a smoke path.
    """
    root = STUDY_DIR / SMOKE_DIR_NAME if smoke else STUDY_DIR
    if not smoke and SMOKE_DIR_NAME in root.parts:
        msg = f"a real run may not write under {SMOKE_DIR_NAME}"
        raise RuntimeError(msg)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "RUN_KIND.txt"
    kind = "smoke" if smoke else "real"
    if marker.exists():
        if marker.read_text().strip() != kind:
            msg = f"{root} was claimed by a {marker.read_text().strip()} run, not a {kind} run"
            raise RuntimeError(msg)
    else:
        marker.write_text(kind + "\n")
    return root


def sha256_of(*, path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: The file.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


ESTIMATOR_MODULES: Final[tuple[str, ...]] = (
    "capacity_estimators",
    "capacity_convex",
    "capacity_physics_torch",
    "capacity_recovery",
)
"""The package modules whose code decides what an estimate or an event draw is."""

ALL_MODULES: Final[tuple[str, ...]] = (*ESTIMATOR_MODULES, "capacity_scaling")
"""The modules whose code decides what a fitted arm is."""


def fingerprint(*, root: Path, modules: tuple[str, ...]) -> str:
    """Hash the built inputs and the code that turns them into results.

    Args:
        root: The output root, whose `built/record.npz` and `built/events.json` are hashed.
        modules: Package module names under `studies` whose source is hashed, plus this folder's
            `capacity_paths.py` and `capacity_scores.py`.

    Returns:
        A hexadecimal digest that changes if any input file or any source file changes.
    """
    package = Path(studies_package.__file__).parent
    files = [
        *(package / f"{name}.py" for name in modules),
        Path(__file__),
        Path(__file__).with_name("capacity_scores.py"),
        inputs_dir(root=root) / "record.npz",
        inputs_dir(root=root) / "events.json",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + sha256_of(path=path).encode())
    return digest.hexdigest()


def check_stamp(*, directory: Path, stamp: str) -> None:
    """Create a directory's stamp file, or check that an existing one matches.

    Args:
        directory: A directory of results.
        stamp: `fingerprint`'s result.

    Raises:
        RuntimeError: If the directory holds results made from other inputs or other code, so that
            no stale estimate, setting, or checkpoint is reused. The caller moves the directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "STAMP.txt"
    if path.exists():
        if path.read_text().strip() != stamp:
            msg = f"{directory} was made from other inputs or code; move it to superseded/"
            raise RuntimeError(msg)
    elif any(directory.iterdir()):
        msg = f"{directory} holds results and no stamp; move it to superseded/"
        raise RuntimeError(msg)
    else:
        path.write_text(stamp + "\n")


def inputs_dir(*, root: Path) -> Path:
    """Return the directory of the built inputs."""
    return root / "built"


def estimates_dir(*, root: Path) -> Path:
    """Return the directory of the estimators' daily estimates."""
    return root / "estimates"


def save_record(*, path: Path, record: CapacityRecord, power_anm_blind: np.ndarray) -> None:
    """Write a record and the power with the export-capped hours left in.

    Args:
        path: The `.npz` file.
        record: The record.
        power_anm_blind: The power before the export-capped hours were removed.
    """
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        sites=np.array(record.sites),
        dates=record.dates.astype("datetime64[D]").astype(np.int64),
        power_mw=record.power_mw,
        power_anm_blind=power_anm_blind,
        clear_sky_w_m2=record.clear_sky_w_m2,
        cams_ghi_w_m2=record.cams_ghi_w_m2,
        cams_shape=record.cams_shape,
        clear_shape=record.clear_shape,
        static_p99_mw=record.static_p99_mw,
    )
    temporary.rename(path)


def load_record(*, root: Path, anm_blind: bool = False) -> CapacityRecord:
    """Read the record the build script wrote.

    Args:
        root: The output root.
        anm_blind: Whether to read the power with the export-capped hours left in.

    Returns:
        The record.
    """
    with np.load(inputs_dir(root=root) / "record.npz") as data:
        return CapacityRecord(
            sites=tuple(str(site) for site in data["sites"]),
            dates=data["dates"].astype("datetime64[D]"),
            power_mw=data["power_anm_blind" if anm_blind else "power_mw"],
            clear_sky_w_m2=data["clear_sky_w_m2"],
            cams_ghi_w_m2=data["cams_ghi_w_m2"],
            cams_shape=data["cams_shape"],
            clear_shape=data["clear_shape"],
            static_p99_mw=data["static_p99_mw"],
        )


def event_to_json(*, event: PlantedEvent) -> dict[str, Any]:
    """Return an event as a JSON-ready mapping."""
    return {**asdict(event), "hit_days": list(event.hit_days)}


def event_from_json(*, payload: dict[str, Any]) -> PlantedEvent:
    """Rebuild an event from `event_to_json`'s result."""
    return PlantedEvent(
        plant=int(payload["plant"]),
        shape=payload["shape"],
        size=float(payload["size"]),
        first_day=int(payload["first_day"]),
        month_days=int(payload["month_days"]),
        hit_days=tuple(int(day) for day in payload["hit_days"]),
    )


def write_json_atomic(*, path: Path, payload: object) -> None:
    """Write JSON through a temporary file.

    Args:
        path: The destination.
        payload: A JSON-serialisable value.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.rename(path)


class Events:
    """Everything `build_capacity_inputs.py` fixes about the events and the windows."""

    def __init__(self, *, payload: dict[str, Any]) -> None:
        """Parse the events file.

        Args:
            payload: The JSON the build script wrote.
        """
        self.draws: list[list[PlantedEvent]] = [
            [event_from_json(payload=e) for e in draw] for draw in payload["draws"]
        ]
        self.control: list[PlantedEvent] = [event_from_json(payload=e) for e in payload["control"]]
        self.reserved: list[PlantedEvent] = [
            event_from_json(payload=e) for e in payload["reserved"]
        ]
        self.false_movement_windows: list[tuple[int, int, int]] = [
            (int(a), int(b), int(c)) for a, b, c in payload["false_movement_windows"]
        ]
        self.scored_months: list[str] = list(payload["scored_months"])
        self.low_plant: int = int(payload["low_plant"])
        self.low_months: list[str] = list(payload["low_months"])
        self.anm_plant: int = int(payload["anm_plant"])


def load_events(*, root: Path) -> Events:
    """Read the events file.

    Args:
        root: The output root.

    Returns:
        The events and windows.
    """
    path = inputs_dir(root=root) / "events.json"
    return Events(payload=json.loads(path.read_text()))


# The DP1 grid gained total_variation_weight 0.5, 1, and 2 and the data_weight axis after the first
# tuning pass, which left DP1 worse than doing nothing; that too used the tuning draw only.
# The KF, HM, CX1, and CX2 grids were widened after the first tuning pass, because that pass's
# winners sat at an edge of the grid (the largest KF process_sd, the smallest HM stay_probability,
# and the smallest CX1 and CX2 fused_weight). The widening used the tuning draw only, so it leaks
# nothing from the selection draws. The other tuned values (KF and HM observation_sd, CX quantile,
# CX2 hinge_weight) are each a single fixed value, so none can sit at an edge.
TUNING_GRIDS: Final[dict[str, list[dict[str, float]]]] = {
    "S0": [{}],
    "EM": [{}],
    "R30": [{}],
    "CSQ": [{}],
    "NR": [{}],
    "PR": [{}],
    "KF": [{"process_sd": sd, "observation_sd": 0.1} for sd in (0.005, 0.01, 0.02, 0.04, 0.08)],
    "HM": [
        {"stay_probability": stay, "observation_sd": 0.15} for stay in (0.8, 0.9, 0.95, 0.98, 0.995)
    ],
    "CX1": [{"quantile": 0.9, "fused_weight": weight} for weight in (0.3, 1.0, 3.0, 10.0, 30.0)],
    "CX2": [
        {"quantile": 0.7, "fused_weight": weight, "hinge_weight": 1.0}
        for weight in (0.3, 1.0, 3.0, 10.0, 30.0)
    ],
    "DP1": [
        {
            "quantile": 0.7,
            "total_variation_weight": weight,
            "data_weight": data_weight,
            "learning_rate": 0.05,
            "samples": 4.0,
            "first_steps": 300.0,
            "later_steps": 80.0,
        }
        for weight, data_weight in (
            (0.5, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (5.0, 1.0),
            (20.0, 1.0),
            (5.0, 20.0),
            (20.0, 20.0),
            (100.0, 20.0),
        )
    ],
}
"""Each estimator's settings. The tuning draw alone chooses among them, never a forecast error."""


def setting_name(*, params: dict[str, float]) -> str:
    """Return a file-name-safe name for a setting.

    Args:
        params: The estimator's settings.

    Returns:
        `default` for no settings, else the keys and values joined, such as
        `fused_weight=10_quantile=0.9`.
    """
    if not params:
        return "default"
    return "_".join(f"{key}={value:g}" for key, value in sorted(params.items()))


def estimate_path(*, root: Path, estimator: str, setting: str, record_id: str) -> Path:
    """Return the file holding one estimator's raw estimates on one record.

    Args:
        root: The output root.
        estimator: The estimator's name.
        setting: `setting_name`'s result.
        record_id: A record identifier, such as `draw03` or `original__reversed`.

    Returns:
        The `.npz` path.
    """
    return estimates_dir(root=root) / estimator / setting / f"{record_id}.npz"


def tuned_path(*, root: Path, estimator: str) -> Path:
    """Return the file recording which setting the tuning draw chose for an estimator."""
    return root / "tuned" / f"tuned_{estimator}.json"


def load_tuned(*, root: Path, estimator: str) -> dict[str, float]:
    """Read the setting the tuning draw chose.

    Args:
        root: The output root.
        estimator: The estimator's name.

    Returns:
        The settings.

    Raises:
        FileNotFoundError: If the tuning has not run, naming the script that runs it.
    """
    path = tuned_path(root=root, estimator=estimator)
    if not path.exists():
        msg = f"{path} is missing; run score_capacity_recovery.py --phase tuning first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())["params"]


def save_estimate(*, path: Path, level: np.ndarray, sd: np.ndarray | None) -> None:
    """Write raw estimates atomically.

    Args:
        path: The `.npz` file.
        level: The level, shape (plants, days).
        sd: The standard deviation, or `None`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, level=level, sd=np.empty((0,)) if sd is None else sd)
    temporary.rename(path)


def load_estimate(*, path: Path) -> RawEstimate:
    """Read raw estimates.

    Args:
        path: The `.npz` file.

    Returns:
        The raw estimate.
    """
    with np.load(path) as data:
        sd = data["sd"]
        return RawEstimate(level=data["level"], sd=sd if sd.size else None)


def record_for(*, root: Path, record_id: str) -> CapacityRecord:
    """Build a record by identifier.

    Args:
        root: The output root.
        record_id: `original`, `anm_blind`, `control`, `draw00` to `draw10`, each optionally with a
            `__reversed` suffix.

    Returns:
        The record, with its losses planted and its days reversed as the identifier says.
    """
    reversed_record = record_id.endswith("__reversed")
    base_id = record_id.removesuffix("__reversed")
    events = load_events(root=root)
    if base_id == "anm_blind":
        record = load_record(root=root, anm_blind=True)
    else:
        record = load_record(root=root)
        if base_id == CONTROL:
            record = plant_losses(record=record, events=events.control)
        elif base_id.startswith("draw"):
            record = plant_losses(record=record, events=events.draws[int(base_id[4:])])
        elif base_id != ORIGINAL:
            msg = f"unknown record identifier {record_id}"
            raise ValueError(msg)
    return record.reversed() if reversed_record else record


def scored_days(*, record: CapacityRecord, events: Events) -> np.ndarray:
    """Return the day indices of every scored month of the lag frame.

    Args:
        record: The record.
        events: The events file.

    Returns:
        The sorted day indices.
    """
    spans = calendar_month_spans(record=record)
    return np.concatenate(
        [np.arange(spans[m][0], spans[m][0] + spans[m][1]) for m in events.scored_months]
    )
