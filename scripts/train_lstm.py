"""Train a compact bidirectional LSTM for PTB-XL multi-label ECG classification.

Controlled architectural comparison against CNN Baseline v1:
- Identical train/validation folds, training-fitted normalization and seed.
- Unweighted BCEWithLogitsLoss; validation macro AP selects the checkpoint.
- Held-out test fold 10 is never used for optimization or evaluation.

The underlying CachedECGStore may decode all cached waveform shards into CPU
memory, including fold 10; only folds 1-8/9 are passed to the loaders.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ecg.data.cached_dataset import CachedECGStore
from src.ecg.features.normalize import ECGStandardizer
from src.ecg.models.lstm import ECGLSTM
from src.ecg.evaluate.metrics import evaluate_multilabel

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
EXPERIMENT_NAME = "lstm_bidirectional_v1"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, normalizer, criterion, optimizer, device):
    model.train()
    loss_total = 0.0
    records_total = 0
    for x, y in loader:
        x = normalizer.transform(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = x.shape[0]
        loss_total += loss.item() * count
        records_total += count
    return loss_total / records_total


@torch.no_grad()
def validate(model, loader, normalizer, criterion, device):
    model.eval()
    loss_total = 0.0
    records_total = 0
    probabilities, targets = [], []
    for x, y in loader:
        x = normalizer.transform(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())
        count = x.shape[0]
        loss_total += loss.item() * count
        records_total += count
    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probabilities)
    metrics = evaluate_multilabel(y_true, y_prob, threshold=0.5)
    return loss_total / records_total, metrics, y_true, y_prob


def run_smoke_test(model, train_loader, val_loader, normalizer,
                   criterion, optimizer, device) -> None:
    """Execute one complete optimizer step and one inference batch; write nothing."""
    print("\nLSTM GPU/CPU SMOKE TEST", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    x, y = next(iter(train_loader))
    x = normalizer.transform(x.to(device, non_blocking=True))
    y = y.to(device, non_blocking=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    assert logits.shape == (len(x), len(CLASSES))
    assert torch.isfinite(loss), "Smoke test: non-finite training loss"
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    ), "Smoke test: missing or non-finite gradients"
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    vx, vy = next(iter(val_loader))
    vx = normalizer.transform(vx.to(device, non_blocking=True))
    model.eval()
    with torch.inference_mode():
        v_logits = model(vx)
        v_loss = criterion(v_logits, vy.to(device, non_blocking=True))
        v_prob = torch.sigmoid(v_logits)
    assert v_prob.shape == (len(vx), len(CLASSES))
    assert torch.isfinite(v_loss) and torch.isfinite(v_prob).all()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_seconds = time.perf_counter() - start

    print("Training batch:", tuple(x.shape))
    print("Unweighted training loss:", f"{loss.item():.6f}")
    print("Training step elapsed (seconds):", f"{train_seconds:.3f}")
    print("Validation batch:", tuple(vx.shape))
    print("Unweighted validation loss:", f"{v_loss.item():.6f}")
    print("Validation step elapsed (seconds):", f"{val_seconds:.3f}")
    if device.type == "cuda":
        print(
            "Peak CUDA memory allocated (MiB):",
            f"{torch.cuda.max_memory_allocated(device) / (1024**2):.1f}",
        )
    print("Backward pass: PASS")
    print("Optimizer step: PASS")
    print("Validation inference: PASS")
    print("Test fold used for training/evaluation: NO")
    print("Artifacts written: NO")
    print("\nLSTM TRAINING SMOKE TEST: PASSED", flush=True)


def main(args) -> None:
    set_seed(args.seed)
    root = args.root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PTB-XL BIDIRECTIONAL LSTM TRAINING PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    manifest_path = root / "data/processed/ptbxl_labels.csv"
    cache_dir = root / "data/processed/ptbxl_waveform_cache_v1"
    print("\nLoading verified waveform cache...", flush=True)
    store = CachedECGStore(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        local_dir=Path("/content/ptbxl_cached_shards"),
    )
    train_dataset = store.split("train")
    val_dataset = store.split("val")
    assert len(train_dataset) == 17084
    assert len(val_dataset) == 2146

    # Independent fold and patient-overlap checks.
    train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
    val_indices = np.asarray(val_dataset.indices, dtype=np.int64)
    assert np.isin(store.folds[train_indices], np.arange(1, 9)).all()
    assert np.all(store.folds[val_indices] == 9)
    manifest = pd.read_csv(manifest_path)
    train_patients = set(manifest.loc[
        manifest["strat_fold"].between(1, 8), "patient_id"
    ])
    val_patients = set(manifest.loc[
        manifest["strat_fold"] == 9, "patient_id"
    ])
    assert train_patients.isdisjoint(val_patients)
    val_ids = manifest.iloc[val_indices]["ecg_id"].to_numpy(dtype=np.int64)
    assert len(np.unique(val_ids)) == len(val_ids)

    normalizer = ECGStandardizer.load(
        root / "artifacts/ptbxl_normalization_v1.json"
    )
    assert normalizer.training_ecgs == 17084
    assert normalizer.samples_per_lead == 17084000

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ECGLSTM(
        num_classes=len(CLASSES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 139909, (
        f"Unexpected LSTM parameter count: {parameter_count}"
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nDATASET")
    print("Training ECGs:", len(train_dataset))
    print("Validation ECGs:", len(val_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Patient separation: PASS")
    print("Training-only normalization: PASS")
    print("\nMODEL")
    print("Architecture: 2-layer bidirectional LSTM")
    print("Hidden units per direction:", HIDDEN_SIZE)
    print("Parameters:", parameter_count)
    print("Loss: unweighted BCEWithLogitsLoss")
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)

    if args.smoke_test:
        run_smoke_test(model, train_loader, val_loader, normalizer,
                       criterion, optimizer, device)
        return

    output_dir = root / "artifacts" / args.experiment_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a fresh run directory before retraining."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "experiment": args.experiment_name,
        "architecture": "ECGLSTM",
        "model_parameters": parameter_count,
        "num_classes": len(CLASSES),
        "class_order": CLASSES,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "bidirectional": True,
        "dropout": DROPOUT,
        "training_folds": list(range(1, 9)),
        "validation_fold": 9,
        "training_ecgs": len(train_dataset),
        "validation_ecgs": len(val_dataset),
        "loss_weighting": "none",
        "validation_loss_weighted": False,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "model_selection": "validation_macro_ap",
        "normalization": "ptbxl_normalization_v1.json",
        "test_partition_used_for_training_or_evaluation": False,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, allow_nan=False)

    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, normalizer, criterion, optimizer, device
        )
        val_loss, results, y_true, y_prob = validate(
            model, val_loader, normalizer, criterion, device
        )
        summary = results["summary"]
        macro_ap = float(summary["macro_ap"])
        if not np.isfinite(macro_ap):
            raise RuntimeError("Non-finite validation macro AP")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_ap": macro_ap,
            "val_macro_roc_auc": float(summary["macro_roc_auc"]),
            "val_macro_f1": float(summary["macro_f1"]),
            "duration_seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_ap={macro_ap:.4f} | "
            f"val_macro_f1={summary['macro_f1']:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if macro_ap > best_ap:
            best_ap = macro_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_ap": best_ap,
                    "model_name": "ECGLSTM",
                    "num_classes": len(CLASSES),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "bidirectional": True,
                    "dropout": DROPOUT,
                    "class_order": CLASSES,
                    "loss_weighting": "none",
                },
                output_dir / "best_model.pt",
            )
            np.savez_compressed(
                output_dir / "best_val_predictions.npz",
                ecg_ids=val_ids,
                y_true=y_true,
                y_prob=y_prob,
                class_order=np.asarray(CLASSES),
            )
            with (output_dir / "best_metrics.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "summary": summary,
                        "per_class": results["per_class"],
                    },
                    f,
                    indent=2,
                    allow_nan=False,
                )
            print("  New best validation macro AP — saved.", flush=True)
        else:
            epochs_without_improvement += 1

        with (output_dir / "history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2, allow_nan=False)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}.", flush=True)
            break

    print("\nFINAL LSTM TRAINING SUMMARY")
    print("=" * 65)
    print("Epochs completed:", len(history))
    print("Best epoch:", best_epoch)
    print("Best validation macro AP:", f"{best_ap:.4f}")
    print("CNN Baseline v1 macro AP: 0.7997")
    print("Difference:", f"{best_ap - 0.7997:+.4f}")
    print("Artifacts:", output_dir)
    print("Test fold used for training/evaluation: NO")
    print("\nLSTM TRAINING: COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and validate a bidirectional LSTM on PTB-XL."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-name",
        choices=[EXPERIMENT_NAME, "lstm_bidirectional_40ep_v1"],
        default=EXPERIMENT_NAME,
        help="Artifact directory for the selected LSTM experiment.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("Batch size and epochs must be positive")
    if args.patience < 1 or args.learning_rate <= 0:
        parser.error("Patience and learning rate must be positive")
    main(args)
