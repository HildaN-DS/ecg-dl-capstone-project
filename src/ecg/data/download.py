"""Download and validate PTB-XL v1.0.3 data from PhysioNet."""

import argparse
import shutil
import time
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import numpy as np
import pandas as pd
import wfdb


BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"

METADATA_FILES = [
    "ptbxl_database.csv",
    "scp_statements.csv",
]

EXPECTED_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def download_file(url: str, target: Path) -> None:
    """Download a file safely, without retaining partial downloads."""
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and target.stat().st_size > 0:
        return

    temporary = target.with_name(target.name + ".part")

    try:
        with urlopen(url, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)

        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded empty file: {target.name}")

        temporary.replace(target)

    finally:
        temporary.unlink(missing_ok=True)


def download_metadata(destination: Path) -> None:
    """Download and check the two PTB-XL metadata files."""
    destination.mkdir(parents=True, exist_ok=True)

    for filename in METADATA_FILES:
        target = destination / filename
        url = f"{BASE_URL}/{filename}"

        download_file(url, target)

        data = pd.read_csv(target)

        if data.empty:
            raise ValueError(f"Empty metadata: {filename}")

        print(f"Metadata ready: {filename} ({len(data)} rows)")


def validate_waveform(record_path: Path) -> None:
    """Check the decoded shape, frequency, leads and numerical values."""
    record = wfdb.rdrecord(str(record_path))

    if record.p_signal.shape != (1000, 12):
        raise ValueError("Unexpected ECG dimensions")

    if record.fs != 100:
        raise ValueError("Unexpected sampling frequency")

    if [name.upper() for name in record.sig_name] != EXPECTED_LEADS:
        raise ValueError("Unexpected ECG lead names or order")

    if not np.isfinite(record.p_signal).all():
        raise ValueError("ECG contains non-finite values")


def download_waveforms(
    destination: Path,
    limit: int | None = None,
    retries: int = 3,
) -> None:
    """Download WFDB record pairs, reusing valid cached recordings."""
    metadata = pd.read_csv(destination / "ptbxl_database.csv")

    paths = metadata["filename_lr"].dropna().unique()

    if limit is not None:
        paths = paths[:limit]

    completed = 0
    failed = []

    for index, path in enumerate(paths, start=1):
        relative = PurePosixPath(path)

        # Restrict acquisition to expected 100 Hz dataset paths.
        if (
            relative.parts[0] != "records100"
            or ".." in relative.parts
            or relative.suffix
        ):
            raise ValueError(f"Invalid recording path: {path}")

        local_record = destination.joinpath(*relative.parts)

        header = local_record.with_suffix(".hea")
        signal_file = local_record.with_suffix(".dat")

        remote_base = f"{BASE_URL}/{relative.as_posix()}"

        # Existing files must decode correctly before being skipped.
        if header.exists() and signal_file.exists():
            try:
                validate_waveform(local_record)
                completed += 1
                print(f"[{index}/{len(paths)}] CACHED: {relative.name}")
                continue
            except (OSError, ValueError, IndexError):
                header.unlink(missing_ok=True)
                signal_file.unlink(missing_ok=True)

        success = False

        for attempt in range(1, retries + 1):
            try:
                download_file(remote_base + ".hea", header)
                download_file(remote_base + ".dat", signal_file)

                validate_waveform(local_record)
                success = True
                break

            except Exception as error:
                print(
                    f"Attempt {attempt}/{retries} failed "
                    f"for {relative.name}: {error}"
                )

                # Remove potentially invalid files before retrying.
                header.unlink(missing_ok=True)
                signal_file.unlink(missing_ok=True)

                if attempt < retries:
                    time.sleep(2)

        if success:
            completed += 1
            print(f"[{index}/{len(paths)}] PASS: {relative.name}")
        else:
            failed.append(path)
            print(f"[{index}/{len(paths)}] FAILED: {relative.name}")

    print(f"\nValidated recordings: {completed}/{len(paths)}")
    print(f"Failed recordings: {len(failed)}")

    if failed:
        failure_file = destination / "download_failures.csv"
        pd.DataFrame({"record_path": failed}).to_csv(
            failure_file, index=False
        )
        print(f"Failure log: {failure_file}")
        raise SystemExit(1)

    # Remove a stale failure log after a successful complete run.
    (destination / "download_failures.csv").unlink(missing_ok=True)


def main() -> None:
    """Run the requested acquisition tasks."""
    parser = argparse.ArgumentParser(
        description="Download PTB-XL metadata and 100 Hz waveforms."
    )

    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory.",
    )

    parser.add_argument(
        "--waveforms",
        action="store_true",
        help="Download and validate 100 Hz ECG recordings.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of recordings to download.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    download_metadata(args.dest)

    if args.waveforms:
        download_waveforms(
            destination=args.dest,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()