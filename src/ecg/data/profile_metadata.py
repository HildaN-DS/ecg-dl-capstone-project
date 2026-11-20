"""Inspect the PTB-XL metadata before preprocessing."""
import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "ecg_id",
    "patient_id",
    "scp_codes",
    "strat_fold",
    "filename_lr",
]

def profile_metadata(metadata_path: Path) -> None:
    """Display basic metadata structure and completeness."""

    df = pd.read_csv(metadata_path)

    missing_columns = [col for col in REQUIRED_COLUMNS
                        if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("PTB-XL Metadata Profile:")

    print("\nDataset dimensions:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nRecording identifiers:")
    print("Unique ECG IDs:", df["ecg_id"].nunique())
    print("Duplicate ECG IDs:", df["ecg_id"].duplicated().sum())
    print("Unique patients:", df["patient_id"].nunique())

    print("\nFold distribution:")
    print(df["strat_fold"].value_counts().sort_index())

    print("\nSample records:")
    print(
        df[REQUIRED_COLUMNS]
        .head(5)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile PTB-XL metadata."
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to ptbxl_database.csv",
    )

    args = parser.parse_args()

    profile_metadata(args.metadata)

if __name__ == "__main__":
    main()