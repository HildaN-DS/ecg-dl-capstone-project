"""Reproduce Phase 1 PTB-XL EDA findings without changing source data.

Usage:
    python scripts/phase1_eda.py --data-dir /path/to/ptb-xl
"""

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
QUALITY_FIELDS = (
    "baseline_drift", "static_noise", "burst_noise", "electrodes_problems"
)
EXPECTED_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
REQUIRED_METADATA = (
    "ecg_id", "patient_id", "scp_codes", "strat_fold", "filename_lr",
    *QUALITY_FIELDS,
)


def parse_codes(value: str) -> dict:
    """Convert an SCP dictionary stored as CSV text into a dictionary."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected SCP dictionary, received {type(parsed).__name__}")
    return parsed


def summarize_metadata(records: pd.DataFrame) -> None:
    print("\n1. METADATA AND PATIENT PARTITIONS")
    print("=" * 45)
    print(f"Recordings: {len(records)}")
    print(f"Metadata columns: {len(records.columns)}")
    print(f"Unique ECG IDs: {records.ecg_id.nunique()}")
    print(f"Duplicate ECG IDs: {records.ecg_id.duplicated().sum()}")
    print(f"Unique patients: {records.patient_id.nunique()}")
    print("\nFold distribution:")
    print(records.strat_fold.value_counts().sort_index().to_string())

    partitions = {
        "Train (1-8)": records.strat_fold.between(1, 8),
        "Validation (9)": records.strat_fold.eq(9),
        "Test (10)": records.strat_fold.eq(10),
    }
    patients = {}
    print("\nProposed partitions (ECGs / patients):")
    for name, mask in partitions.items():
        patients[name] = set(records.loc[mask, "patient_id"])
        print(f"  {name}: {int(mask.sum())} / {len(patients[name])}")
    print("\nPatient overlaps:")
    print("  Train vs validation:", len(patients["Train (1-8)"] & patients["Validation (9)"]))
    print("  Train vs test:", len(patients["Train (1-8)"] & patients["Test (10)"]))
    print("  Validation vs test:", len(patients["Validation (9)"] & patients["Test (10)"]))
    print("  Patients in >1 fold:", int((records.groupby("patient_id").strat_fold.nunique() > 1).sum()))


def summarize_diagnostics(records: pd.DataFrame, statements: pd.DataFrame) -> None:
    print("\n2. DIAGNOSTIC ANNOTATIONS")
    print("=" * 45)
    mapping = statements.loc[
        statements.diagnostic.eq(1) & statements.diagnostic_class.isin(LABELS),
        "diagnostic_class",
    ].to_dict()
    print(f"SCP statement definitions: {len(statements)}")
    print(f"Mapped diagnostic definitions: {len(mapping)}")

    codes = records.scp_codes.map(parse_codes)
    classes = codes.map(lambda d: {mapping[k] for k in d if k in mapping})
    counts = Counter(label for labels in classes for label in labels)
    print("\nRecording-level superclass counts:")
    for label in LABELS:
        print(f"  {label}: {counts[label]} ({counts[label] / len(records):.1%})")

    multiplicity = classes.map(len).value_counts().sort_index()
    print("\nNumber of diagnostic superclasses per ECG:")
    for count in range(len(LABELS) + 1):
        print(f"  {count} labels: {int(multiplicity.get(count, 0))}")

    unmapped = classes.map(len).eq(0)
    print("\nUnmapped ECGs by fold:")
    print(records.loc[unmapped, "strat_fold"].value_counts().sort_index().to_string())
    top_codes = Counter(code for d in codes[unmapped] for code in d)
    print("\nMost common SCP codes among unmapped ECGs:")
    print("  " + ", ".join(f"{code}={count}" for code, count in top_codes.most_common(10)))
    print("  Unrecognized SCP codes:", sorted(set(top_codes) - set(statements.index)))
    print(f"  Eligible ECGs (>=1 mapped class): {int((~unmapped).sum())}")

    overlapping = classes.map(lambda x: "NORM" in x and len(x) > 1)
    overlap_counts = Counter(label for x in classes[overlapping] for label in x if label != "NORM")
    print(f"\nNORM + abnormal superclass ECGs: {int(overlapping.sum())}")
    for label in LABELS[1:]:
        print(f"  NORM with {label}: {overlap_counts[label]}")
    unknown = sum(
        value == 0
        for d in codes
        for key, value in d.items()
        if key in mapping
    )
    print(f"Diagnostic SCP entries with likelihood 0: {unknown}")


def summarize_quality_annotations(records: pd.DataFrame) -> None:
    print("\n3. SIGNAL-QUALITY ANNOTATIONS")
    print("=" * 45)
    flags = pd.DataFrame(index=records.index)
    for field in QUALITY_FIELDS:
        text = records[field].astype("string").fillna("")
        meaningful = text.str.replace(r"[\s,;]+", "", regex=True).ne("")
        flags[field] = meaningful
        placeholders = int((records[field].notna() & ~meaningful).sum())
        print(f"  {field}: {int(meaningful.sum())} meaningful, {placeholders} empty placeholders")
    flag_count = flags.sum(axis=1)
    print("\nQuality flags per recording:")
    print(flag_count.value_counts().sort_index().to_string())
    print(f"  With >=1 quality flag: {int(flag_count.gt(0).sum())}")
    print(f"  With >1 quality flag: {int(flag_count.gt(1).sum())}")


def summarize_waveforms(records: pd.DataFrame, data_dir: Path, n: int, seed: int) -> None:
    import wfdb

    print("\n4. TRAINING-FOLD WAVEFORM SAMPLE")
    print("=" * 45)
    train = records.loc[records.strat_fold.between(1, 8)]
    sample = (
        train.groupby("strat_fold", group_keys=False)
        .sample(n=n, random_state=seed)
        .sort_values(["strat_fold", "ecg_id"])
    )
    qc_rows = []
    for row in sample.itertuples(index=False):
        signal, fields = wfdb.rdsamp(str(data_dir / row.filename_lr))
        finite = bool(np.isfinite(signal).all())
        qc_rows.append({
            "ecg_id": row.ecg_id,
            "fold": row.strat_fold,
            "shape_ok": signal.shape == (1000, 12),
            "fs_ok": fields["fs"] == 100,
            "leads_ok": [x.casefold() for x in fields["sig_name"]]
            == [x.casefold() for x in EXPECTED_LEADS],
            "finite": finite,
            "constant_leads": int(np.count_nonzero(np.ptp(signal, axis=0) == 0)) if finite else None,
            "min_lead_std_mV": float(np.std(signal, axis=0).min()) if finite else np.nan,
            "max_abs_mV": float(np.abs(signal).max()) if finite else np.nan,
        })
    qc = pd.DataFrame(qc_rows)
    print("Recordings screened:", len(qc))
    for field in ("shape_ok", "fs_ok", "leads_ok", "finite"):
        print(f"  {field}: {int(qc[field].sum())}/{len(qc)}")
    print("  Recordings with constant leads:", int(qc.constant_leads.fillna(0).gt(0).sum()))
    print("\nAmplitude statistics (mV):")
    print(qc[["min_lead_std_mV", "max_abs_mV"]].describe().loc[["min", "50%", "max"]].to_string())
    failures = qc.loc[
        ~qc[["shape_ok", "fs_ok", "leads_ok", "finite"]].all(axis=1)
        | qc.constant_leads.fillna(0).gt(0)
    ]
    print("\nRecords requiring structural review:")
    print("None" if failures.empty else failures.to_string(index=False))
    print("\nFive largest absolute amplitudes:")
    print(qc.nlargest(5, "max_abs_mV")[["ecg_id", "fold", "max_abs_mV"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PTB-XL Phase 1 EDA findings")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing PTB-XL metadata and records100/")
    parser.add_argument("--sample-per-fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-waveforms", action="store_true", help="Run metadata-only checks")
    args = parser.parse_args()
    if args.sample_per_fold < 1:
        parser.error("--sample-per-fold must be >= 1")

    records = pd.read_csv(args.data_dir / "ptbxl_database.csv")
    statements = pd.read_csv(args.data_dir / "scp_statements.csv", index_col=0)
    missing = sorted(set(REQUIRED_METADATA) - set(records.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    missing_scp = {"diagnostic", "diagnostic_class"} - set(statements.columns)
    if missing_scp:
        raise ValueError(f"Missing SCP columns: {sorted(missing_scp)}")

    print("PTB-XL PHASE 1 — REPRODUCIBLE EDA")
    print("Dataset directory:", args.data_dir)
    summarize_metadata(records)
    summarize_diagnostics(records, statements)
    summarize_quality_annotations(records)
    if not args.skip_waveforms:
        summarize_waveforms(records, args.data_dir, args.sample_per_fold, args.seed)


if __name__ == "__main__":
    main()
