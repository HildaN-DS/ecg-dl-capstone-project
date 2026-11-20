"""Export ONLY frozen model inference artifacts to a portable ZIP.

Example on Colab with Drive mounted:
python scripts/export_api_artifacts.py \
  --artifacts-dir '/content/drive/MyDrive/ECG DL Capstone Project/artifacts' \
  --output '/content/ecg_frozen_api_artifacts.zip'

No waveform cache, raw patient records, predictions or test labels are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

FILES = (
    "final_model_selection_v1.json",
    "cnn1d_baseline_v1/best_model.pt",
    "ptbxl_normalization_v1.json",
    "final_test_cnn1d_v1/test_metrics.json",  # hashes/provenance only; contains aggregate test metrics
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(root: Path) -> dict[str, str]:
    for name in FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing frozen artifact: {root / name}")

    selection_path = root / FILES[0]
    checkpoint_path = root / FILES[1]
    normalization_path = root / FILES[2]
    provenance_path = root / FILES[3]

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    if selection.get("selected_experiment") != "cnn1d_baseline_v1":
        raise ValueError("Selected experiment does not match the frozen CNN")
    if selection.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Model selection was not frozen")
    if selection.get("thresholds") != [0.5] * 5:
        raise ValueError("Frozen thresholds changed")
    if selection.get("class_order") != ["NORM", "MI", "STTC", "CD", "HYP"]:
        raise ValueError("Class order changed")
    if Path(selection.get("selected_checkpoint", "")).parts[-2:] != (
        "cnn1d_baseline_v1", "best_model.pt"
    ):
        raise ValueError("Unexpected selected checkpoint path")

    hashes = {name: sha256(root / name) for name in FILES}
    if hashes[FILES[1]] != selection.get("checkpoint_sha256"):
        raise ValueError("Frozen CNN checkpoint checksum mismatch")
    if hashes[FILES[1]] != provenance.get("checkpoint_sha256"):
        raise ValueError("Final report checkpoint checksum mismatch")
    if hashes[FILES[2]] != provenance.get("normalization_sha256"):
        raise ValueError("Normalization checksum mismatch")
    if hashes[FILES[0]] != provenance.get("selection_record_sha256"):
        raise ValueError("Model-selection checksum mismatch")
    if provenance.get("test_fold") != 10:
        raise ValueError("Unexpected final evaluation fold")
    return hashes


def export_artifacts(root: Path, output: Path) -> dict[str, str]:
    hashes = verify_artifacts(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output ZIP must not be written inside frozen artifact directory")

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(root / name, arcname=name)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hashes = export_artifacts(args.artifacts_dir, args.output)
    print("FROZEN API ARTIFACT EXPORT")
    for name in FILES:
        print(f"  {name}: VERIFIED")
    print("  Checkpoint SHA256:", hashes[FILES[1]])
    print("  Archive:", args.output)
    print("  Patient-level ECG data included: NO")
    print("FROZEN API ARTIFACT EXPORT: PASSED")


if __name__ == "__main__":
    main()
