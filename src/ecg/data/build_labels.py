
"""Build a five-superclass label manifest from PTB-XL metadata."""

import argparse
import ast
from pathlib import Path

import pandas as pd


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def build_labels(data_dir: Path, output: Path) -> None:
    """Create a label manifest without modifying raw data."""

    metadata = pd.read_csv(data_dir / "ptbxl_database.csv")
    scp = pd.read_csv(
        data_dir / "scp_statements.csv",
        index_col=0,
    )

    required = {
        "ecg_id", "patient_id", "strat_fold",
        "filename_lr", "scp_codes",
    }

    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    # Use only diagnostic SCP statements mapped to target classes.
    diagnostic_map = scp.loc[
        scp["diagnostic"].eq(1)
        & scp["diagnostic_class"].isin(CLASSES),
        "diagnostic_class",
    ].to_dict()

    def extract_classes(raw_codes: str) -> set[str]:
        codes = ast.literal_eval(raw_codes)

        if not isinstance(codes, dict):
            raise ValueError("Expected an SCP code dictionary.")

        # Code presence determines the label. No likelihood cutoff.
        return {
            diagnostic_map[code]
            for code in codes
            if code in diagnostic_map
        }

    metadata["mapped_classes"] = metadata["scp_codes"].map(
        extract_classes
    )

    # Keep only recordings with at least one mapped class.
    eligible = metadata.loc[
        metadata["mapped_classes"].map(bool)
    ].copy()

    if eligible["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs detected.")

    if (
        eligible["patient_id"].isna().any()
        or not eligible["patient_id"].mod(1).eq(0).all()
    ):
        raise ValueError("Invalid patient IDs detected.")

    if not eligible["strat_fold"].between(1, 10).all():
        raise ValueError("Unexpected fold assignments detected.")

    eligible["patient_id"] = eligible["patient_id"].astype(int)

    if (
        eligible.groupby("patient_id")["strat_fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Patients found in multiple folds.")

    # Build five binary target columns.
    for label in CLASSES:
        eligible[label] = eligible["mapped_classes"].map(
            lambda classes: int(label in classes)
        )

    columns = [
        "ecg_id", "patient_id", "strat_fold",
        "filename_lr", *CLASSES,
    ]

    manifest = eligible[columns].copy()

    # Protect the original dataset directory.
    if output.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("Output must be outside the raw data directory.")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    print("PTB-XL LABEL PREPARATION")
    print("=" * 40)
    print("Original recordings:", len(metadata))
    print("Eligible recordings:", len(manifest))
    print("Excluded without mapped labels:",
          len(metadata) - len(manifest))

    print("\nTarget frequencies:")
    print(manifest[list(CLASSES)].sum().to_string())

    print("\nEligible recordings by partition:")
    fold = manifest["strat_fold"]

    print("Training:", int(fold.between(1, 8).sum()))
    print("Validation:", int(fold.eq(9).sum()))
    print("Test:", int(fold.eq(10).sum()))

    print("\nMulti-label recordings:",
          int(manifest[list(CLASSES)].sum(axis=1).gt(1).sum()))

    print("NORM overlaps:",
          int((
              manifest["NORM"].eq(1)
              & manifest[list(CLASSES[1:])].sum(axis=1).gt(0)
          ).sum()))

    print("\nManifest saved to:", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PTB-XL five-class label manifest."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    build_labels(args.data_dir, args.output)


if __name__ == "__main__":
    main()
