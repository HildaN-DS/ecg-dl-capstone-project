
"""Smoke-test remote PTB-XL waveform access."""

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd
import wfdb


def validate_records(metadata: Path, limit: int) -> None:
    """Check waveform dimensions and sampling rates."""
    records = pd.read_csv(metadata)
    paths = records["filename_lr"].dropna().head(limit)

    passed = 0

    for path in paths:
        relative = PurePosixPath(path)
        remote_dir = f"ptb-xl/1.0.3/{relative.parent}"

        try:
            record = wfdb.rdrecord(
                relative.name,
                pn_dir=remote_dir,
            )

            valid = (
                record.p_signal.shape == (1000, 12)
                and record.fs == 100
            )

            print(f"{relative.name}: {'PASS' if valid else 'FAIL'}")
            passed += int(valid)

        except Exception as error:
            print(f"{relative.name}: ERROR - {error}")

    print(f"\nValidated: {passed}/{len(paths)}")

    if passed != len(paths):
        raise SystemExit("Some waveform checks failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    validate_records(args.metadata, args.limit)
