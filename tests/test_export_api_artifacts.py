"""Synthetic-only tests; no real PTB-XL artifacts are opened."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.export_api_artifacts import FILES, export_artifacts, verify_artifacts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root):
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    selection_path = root / FILES[0]
    checkpoint_path = root / FILES[1]
    norm_path = root / FILES[2]
    report_path = root / FILES[3]
    selection = {
        "selected_experiment": "cnn1d_baseline_v1",
        "status": "frozen_before_test_evaluation",
        "thresholds": [0.5] * 5,
        "class_order": ["NORM", "MI", "STTC", "CD", "HYP"],
        "selected_checkpoint": "/content/drive/artifacts/cnn1d_baseline_v1/best_model.pt",
        "checkpoint_sha256": digest(checkpoint_path),
    }
    selection_path.write_text(json.dumps(selection))
    report_path.write_text(json.dumps({
        "checkpoint_sha256": digest(checkpoint_path),
        "normalization_sha256": digest(norm_path),
        "selection_record_sha256": digest(selection_path),
        "test_fold": 10,
    }))


def test_verified_export_contains_only_four_files(tmp_path):
    root = tmp_path / "artifacts"
    make_fixture(root)
    exported = tmp_path / "export.zip"
    result = export_artifacts(root, exported)
    assert len(result) == 4
    with ZipFile(exported) as archive:
        assert sorted(archive.namelist()) == sorted(FILES)
        assert archive.read(FILES[1]) == (root / FILES[1]).read_bytes()


def test_refuse_reexport_without_overwrite(tmp_path):
    root = tmp_path / "artifacts"
    make_fixture(root)
    target = tmp_path / "export.zip"
    export_artifacts(root, target)
    with pytest.raises(FileExistsError):
        export_artifacts(root, target)


def test_reject_checkpoint_tampering(tmp_path):
    root = tmp_path / "artifacts"
    make_fixture(root)
    (root / FILES[1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifacts(root)


def test_reject_output_inside_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    make_fixture(root)
    with pytest.raises(ValueError, match="inside"):
        export_artifacts(root, root / "unsafe.zip")
