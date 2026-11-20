"""Build a resumable, sharded PTB-XL waveform cache without altering raw data."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import wfdb

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ecg_id", "patient_id", "strat_fold", "filename_lr", *CLASSES}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Manifest is empty or has missing required data")
    if df["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs")
    if not df["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("Unexpected folds")
    if not df[list(CLASSES)].isin([0, 1]).all().all():
        raise ValueError("Nonbinary label")
    if df[list(CLASSES)].sum(axis=1).lt(1).any():
        raise ValueError("All-zero targets")
    if df.groupby("patient_id")["strat_fold"].nunique().gt(1).any():
        raise ValueError("Patient appears in multiple folds")
    for name in df["filename_lr"]:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix:
            raise ValueError(f"Unsafe or unexpected WFDB relative stem: {name}")
    return df.reset_index(drop=True)


def build_shard(rows: pd.DataFrame, raw_dir: Path, local_dir: Path) -> Path:
    waveforms = np.empty((len(rows), 12, 1000), dtype=np.float32)
    targets = rows[list(CLASSES)].to_numpy(dtype=np.float32)
    ids = rows["ecg_id"].to_numpy(dtype=np.int64)
    patient_ids = rows["patient_id"].to_numpy(dtype=np.int64)
    folds = rows["strat_fold"].to_numpy(dtype=np.int8)
    wfdb_dir = local_dir / "wfdb"

    for position, row in enumerate(rows.itertuples(index=False)):
        if position % 8 == 0:
            print(
                f"  Processing ECG {position + 1}/{len(rows)} "
                f"(ID {row.ecg_id})",
                flush=True,
            )
        rel = PurePosixPath(row.filename_lr)
        local_stem = wfdb_dir.joinpath(*rel.parts)
        remote_stem = raw_dir.joinpath(*rel.parts)
        local_stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in (".hea", ".dat"):
            shutil.copyfile(
                remote_stem.with_suffix(extension),
                local_stem.with_suffix(extension),
            )
        signal, fields = wfdb.rdsamp(str(local_stem))
        if signal.shape != (1000, 12) or fields["fs"] != 100:
            raise ValueError(f"ECG {row.ecg_id}: unexpected signal dimensions or frequency")
        if [lead.casefold() for lead in fields["sig_name"]] != [lead.casefold() for lead in LEADS]:
            raise ValueError(f"ECG {row.ecg_id}: unexpected lead order")
        if not np.isfinite(signal).all():
            raise ValueError(f"ECG {row.ecg_id}: non-finite values")
        waveforms[position] = signal.T.astype(np.float32)
        if (position + 1) % 8 == 0 or position + 1 == len(rows):
            print(
                f"  Decoded {position + 1}/{len(rows)} ECGs",
                flush=True,
            )

    path = local_dir / "shard.npz"
    np.savez(
        path, signals=waveforms, targets=targets,
        ecg_ids=ids, patient_ids=patient_ids, folds=folds,
    )
    return path


def run(args: argparse.Namespace) -> None:
    manifest_file = args.manifest.resolve()
    raw_dir = args.data_dir.resolve()
    output = args.output.resolve()
    if output.is_relative_to(raw_dir):
        raise ValueError("Cache output cannot be inside raw data directory")
    rows = load_manifest(manifest_file)
    manifest_hash = sha256(manifest_file)
    output.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": VERSION,
        "manifest_sha256": manifest_hash,
        "recordings": len(rows),
        "shard_size": args.shard_size,
        "classes": list(CLASSES),
        "signal_shape": [12, 1000],
        "dtype": "float32",
    }
    spec_path = output / "cache_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Cache specification changed. Use a different output directory.")
    else:
        if any(output.iterdir()):
            raise ValueError("Output contains files but no cache specification")
        write_json_atomic(spec_path, spec)

    count = (len(rows) + args.shard_size - 1) // args.shard_size
    max_shards = count if args.max_shards is None else min(count, args.max_shards)
    print(f"Manifest: {len(rows)} ECGs; shards: {count}; processing up to: {max_shards}", flush=True)

    for shard_num in range(max_shards):
        start = shard_num * args.shard_size
        stop = min(len(rows), start + args.shard_size)
        shard_file = output / f"shard_{shard_num:04d}.npz"
        marker_file = output / f"shard_{shard_num:04d}.json"
        expected = {"version": VERSION, "manifest_sha256": manifest_hash,
                    "index": shard_num, "start": start, "stop": stop}

        if marker_file.exists():
            marker = json.loads(marker_file.read_text(encoding="utf-8"))
            if (any(marker.get(k) != v for k, v in expected.items())
                    or not shard_file.is_file()
                    or sha256(shard_file) != marker.get("file_sha256")):
                raise ValueError(f"Completed shard {shard_num} failed integrity verification")
            print(f"Shard {shard_num + 1}/{count}: verified, skipped", flush=True)
            continue

        # No completion marker means an interrupted attempt may be rebuilt.
        with tempfile.TemporaryDirectory(prefix="ptbxl_shard_") as work:
            local_shard = build_shard(rows.iloc[start:stop], raw_dir, Path(work))
            tmp_remote = shard_file.with_suffix(".npz.part")
            shutil.copyfile(local_shard, tmp_remote)
            file_hash = sha256(local_shard)
            if sha256(tmp_remote) != file_hash:
                raise IOError(f"Shard {shard_num}: copy verification failed")
            os.replace(tmp_remote, shard_file)
            write_json_atomic(marker_file, {**expected, "file_sha256": file_hash})
        print(f"Shard {shard_num + 1}/{count}: saved {stop-start} ECGs", flush=True)

    if max_shards == count:
        write_json_atomic(output / "READY.json", {**spec, "shards": count})
        print("CACHE COMPLETE: all shards verified and marked ready", flush=True)
    else:
        print("PILOT/INCOMPLETE: cache is not ready for model training", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable PTB-XL waveform shards")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()
    if args.shard_size < 1 or (args.max_shards is not None and args.max_shards < 1):
        parser.error("--shard-size and --max-shards must be positive")
    run(args)


if __name__ == "__main__":
    main()
