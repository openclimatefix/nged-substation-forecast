"""Tests of `scripts/deploy/verify_bundle.py`, the image build's check of the step-5 bundle."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from _step5_bundle_test_data import write_bundle

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy" / "verify_bundle.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_bundle", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_bundle = _load_script()

# Manifest entries the image leaves out, so the check must not report them as missing.
_NOT_SHIPPED = ("fixtures/origin_2026-04-17/outputs_native.parquet", "components/step5/seed0.pt")


def _seal(root: Path) -> None:
    """Record every file's SHA-256 in the manifest, plus two entries the image does not ship."""
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    } | dict.fromkeys(_NOT_SHIPPED, "0" * 64)
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    root = write_bundle(tmp_path / "bundle")
    _seal(root)
    return root


def test_a_sealed_bundle_has_no_faults_and_loads(bundle_dir: Path) -> None:
    assert verify_bundle.bundle_faults(bundle_dir) == []
    assert verify_bundle.main(bundle_dir) == 0


def test_each_kind_of_fault_is_reported(bundle_dir: Path) -> None:
    (bundle_dir / "primaries.json").unlink()
    (bundle_dir / "extra.json").write_text("{}")
    (bundle_dir / "config.json").write_text((bundle_dir / "config.json").read_text() + " ")
    assert verify_bundle.bundle_faults(bundle_dir) == [
        "missing: primaries.json",
        "not in the manifest: extra.json",
        "SHA-256 differs: config.json",
    ]
    assert verify_bundle.main(bundle_dir) == 1


def test_a_bundle_the_service_cannot_serve_fails_the_check(bundle_dir: Path) -> None:
    config = json.loads((bundle_dir / "config.json").read_text())
    config["components"]["step5"]["b1_constants"]["cams_columns_source"] = "arm replay"
    (bundle_dir / "config.json").write_text(json.dumps(config))
    _seal(bundle_dir)
    assert verify_bundle.bundle_faults(bundle_dir) == []
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode != 0
    assert "cams_columns_source" in result.stderr
