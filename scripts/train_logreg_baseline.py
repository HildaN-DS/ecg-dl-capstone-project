
"""Reproduce the PTB-XL logistic regression baseline.

Reads verified feature artifacts and the training-fitted scaler.
Retrains the model, evaluates validation data, and compares
predictions with the preserved baseline.

This script is read-only: it does not overwrite artifacts.
"""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ecg.evaluate.metrics import evaluate_multilabel


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main(root):
    feature_dir = root / "data/processed/ptbxl_time_features_v1"

    scaler_path = (
        root / "artifacts/ptbxl_time_feature_scaler_v1.joblib"
    )

    reference_dir = root / "artifacts/logreg_time108_v1"

    with (reference_dir / "run_metadata.json").open(
        encoding="utf-8"
    ) as stream:
        reference = json.load(stream)

    print("LOGISTIC REGRESSION REPRODUCIBILITY TEST")
    print("=" * 55)

    # 1. Verify the source artifacts have not changed.
    source_files = {
        "training_features": feature_dir / "train.npz",
        "validation_features": feature_dir / "val.npz",
        "feature_scaler": scaler_path,
    }

    for name, path in source_files.items():
        actual_hash = sha256_file(path)
        expected_hash = reference["source_sha256"][name]

        assert actual_hash == expected_hash, (
            f"Source checksum mismatch: {name}"
        )

    print("Source artifact checksums: PASS")

    # 2. Load the verified feature matrices.
    with np.load(
        feature_dir / "train.npz", allow_pickle=False
    ) as data:
        X_train = data["X"].copy()
        y_train = data["y"].copy()
        train_folds = data["folds"].copy()
        train_patients = data["patient_ids"].copy()

    with np.load(
        feature_dir / "val.npz", allow_pickle=False
    ) as data:
        X_val = data["X"].copy()
        y_val = data["y"].copy()
        val_folds = data["folds"].copy()
        val_ids = data["ecg_ids"].copy()
        val_patients = data["patient_ids"].copy()

    assert X_train.shape == (17084, 108)
    assert X_val.shape == (2146, 108)

    assert y_train.shape == (17084, 5)
    assert y_val.shape == (2146, 5)

    assert set(np.unique(train_folds)) == set(range(1, 9))
    assert set(np.unique(val_folds)) == {9}

    assert np.intersect1d(
        train_patients, val_patients
    ).size == 0

    assert np.isin(y_train, [0, 1]).all()
    assert np.isin(y_val, [0, 1]).all()

    assert reference["class_order"] == CLASSES

    print("Partitions and targets: PASS")

    # 3. Apply the existing training-fitted scaler.
    # Only load trusted joblib artifacts.
    scaler = joblib.load(scaler_path)

    assert scaler.n_features_in_ == 108
    assert int(scaler.n_samples_seen_) == 17084

    Z_train = scaler.transform(X_train)
    Z_val = scaler.transform(X_val)

    assert np.isfinite(Z_train).all()
    assert np.isfinite(Z_val).all()

    print("Training-only feature scaler: PASS")

    # 4. Recreate the exact original model configuration.
    config = reference["configuration"]

    assert config["estimator"] == (
        "OneVsRestClassifier(LogisticRegression)"
    )
    assert config["class_weight"] is None
    assert config["threshold"] == 0.5

    classifier = LogisticRegression(
        penalty=config["penalty"],
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
        class_weight=config["class_weight"],
    )

    model = OneVsRestClassifier(
        classifier, n_jobs=1
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(
            "always", ConvergenceWarning
        )

        model.fit(
            Z_train,
            y_train.astype(np.int64),
        )

    convergence_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, ConvergenceWarning)
    ]

    assert not convergence_warnings, (
        "Logistic regression convergence warning detected."
    )

    print("Model retraining: PASS")
    print("Convergence warnings: 0")

    # 5. Predict and evaluate the complete validation fold.
    probabilities = model.predict_proba(Z_val)

    results = evaluate_multilabel(
        y_val,
        probabilities,
        threshold=0.5,
    )

    summary = results["summary"]

    print("\nRETRAINED VALIDATION METRICS")

    for metric in (
        "macro_ap",
        "macro_roc_auc",
        "macro_f1",
        "micro_f1",
        "mean_binary_log_loss",
    ):
        actual = summary[metric]
        expected = reference["validation_metrics"][metric]

        print(f"{metric}: {actual:.4f}")

        assert np.isclose(
            actual,
            expected,
            atol=1e-6,
            rtol=0,
        ), f"Metric reproduction failed: {metric}"

    print("\nValidation metric reproduction: PASS")

    # 6. Compare against the previously saved predictions.
    prediction_path = (
        reference_dir / "validation_predictions.npz"
    )

    assert sha256_file(prediction_path) == (
        reference["predictions_sha256"]
    )

    with np.load(
        prediction_path, allow_pickle=False
    ) as saved:
        np.testing.assert_array_equal(
            val_ids, saved["ecg_ids"]
        )

        np.testing.assert_array_equal(
            y_val, saved["y_true"]
        )

        np.testing.assert_array_equal(
            val_folds, saved["folds"]
        )

        np.testing.assert_allclose(
            probabilities,
            saved["y_prob"],
            atol=1e-6,
            rtol=1e-6,
        )

    print("Saved prediction reproduction: PASS")

    print("\nTest partition accessed: NO")
    print("Artifacts overwritten: NO")
    print("\nBASELINE REPRODUCIBILITY: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Google Drive ECG Capstone Project directory",
    )

    args = parser.parse_args()

    main(args.root)
