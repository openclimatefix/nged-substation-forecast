"""Historical polar probes must reject component code before reading private caches."""

import subprocess
import sys
from pathlib import Path


def test_historical_audit_rejects_component_checkout_before_reading_data(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "studies/production_feature_faults/audit_production_features.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frames",
            str(tmp_path / "missing-frames"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Historical polar-wind audit requires checkout 5ac2e16a" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert not (tmp_path / "output").exists()
