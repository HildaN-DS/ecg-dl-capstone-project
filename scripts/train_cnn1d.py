
"""Train baseline or positive-class-weighted 1D CNN on PTB-XL.

Uses verified waveform shards, training-only normalization, and patient-disjoint
training (folds 1-8) and validation (fold 9) subsets. Selects checkpoints using
validation macro average precision. The cache may load all waveform shards into
memory, but the test fold (10) is never used for training or evaluation.
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
from src.ecg.models.cnn1d import ECG1DCNN
from src.ecg.evaluate.metrics import evaluate_multilabel


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model, loader, normalizer, criterion, optimizer, device
):
    model.train()

    total_loss = 0.0
    total_records = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = normalizer.transform(x)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        optimizer.step()

        count = x.shape[0]
        total_loss += loss.item() * count
        total_records += count

    return total_loss / total_records


@torch.no_grad()
def validate(model, loader, normalizer, criterion, device):
    model.eval()

    total_loss = 0.0
    total_records = 0
    probabilities = []
    targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = normalizer.transform(x)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")

        prob = torch.sigmoid(logits)

        probabilities.append(prob.cpu().numpy())
        targets.append(y.cpu().numpy())

        count = x.shape[0]
        total_loss += loss.item() * count
        total_records += count

    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probabilities)

    results = evaluate_multilabel(
        y_true,
        y_prob,
        threshold=0.5,
    )

    return (
        total_loss / total_records,
        results,
        y_true,
        y_prob,
    )


def main(args):
    set_seed(args.seed)

    root = args.root
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PTB-XL CNN TRAINING PIPELINE", flush=True)
    print("=" * 60, flush=True)

    print("Device:", device, flush=True)

    if device.type == "cuda":
        print(
            "GPU:", torch.cuda.get_device_name(0),
            flush=True,
        )

    # Prepare the verified waveform store.
    manifest_path = (
        root / "data/processed/ptbxl_labels.csv"
    )

    cache_dir = (
        root / "data/processed/ptbxl_waveform_cache_v1"
    )

    print("\nLoading waveform cache...", flush=True)

    store = CachedECGStore(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        local_dir=Path("/content/ptbxl_cached_shards"),
    )

    train_dataset = store.split("train")
    val_dataset = store.split("val")

    assert len(train_dataset) == 17084
    assert len(val_dataset) == 2146

    # Independently check patient separation.
    manifest = pd.read_csv(manifest_path)

    train_patients = set(
        manifest.loc[
            manifest["strat_fold"].between(1, 8),
            "patient_id",
        ]
    )

    val_patients = set(
        manifest.loc[
            manifest["strat_fold"] == 9,
            "patient_id",
        ]
    )

    assert train_patients.isdisjoint(val_patients)

    # Keep validation identifiers in dataset iteration order.
    val_ids = manifest.iloc[
        val_dataset.indices
    ]["ecg_id"].to_numpy(dtype=np.int64)

    normalizer = ECGStandardizer.load(
        root / "artifacts/ptbxl_normalization_v1.json"
    )

    assert normalizer.training_ecgs == 17084
    assert normalizer.samples_per_lead == 17084000

    # Calculate positive-class weights from training labels only. Do not use
    # validation/test labels when estimating prevalence.
    train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
    assert len(train_indices) == 17084
    assert np.isin(store.folds[train_indices], np.arange(1, 9)).all()

    train_targets = store.dataset.tensors[1][
        torch.as_tensor(train_indices, dtype=torch.long)
    ]
    assert train_targets.shape == (17084, len(CLASSES))
    assert torch.isin(
        train_targets, torch.tensor([0.0, 1.0])
    ).all()

    positive_counts = train_targets.sum(dim=0, dtype=torch.float64)
    negative_counts = len(train_dataset) - positive_counts
    assert torch.all(positive_counts > 0)
    assert torch.all(negative_counts > 0)

    positive_weights = (
        negative_counts / positive_counts
    ).to(torch.float32)
    assert torch.isfinite(positive_weights).all()

    if args.loss_weighting == "inverse_prevalence":
        training_pos_weight = positive_weights.to(device)
    else:
        training_pos_weight = None

    print("\nLOSS CONFIGURATION", flush=True)
    print("Weighting:", args.loss_weighting, flush=True)
    print(
        "Training positive counts:",
        positive_counts.to(torch.int64).tolist(),
        flush=True,
    )
    print(
        "Training-derived positive weights:",
        positive_weights.tolist(),
        flush=True,
    )
    print("Validation loss: unweighted BCE", flush=True)

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

    print("\nDATASET")
    print("Training ECGs:", len(train_dataset))
    print("Validation ECGs:", len(val_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Patient separation: PASS")
    print("Training-only normalization: PASS")

    model = ECG1DCNN(
        num_classes=5,
        dropout=0.3,
    ).to(device)

    train_criterion = nn.BCEWithLogitsLoss(
        pos_weight=training_pos_weight
    )
    val_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("\nMODEL")
    print("Parameters:", parameter_count)
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)

    # Smoke test: exactly one training and validation batch.
    if args.smoke_test:
        print("\nRUNNING SMOKE TEST")

        x, y = next(iter(train_loader))

        x = normalizer.transform(
            x.to(device)
        )
        y = y.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = train_criterion(logits, y)

        assert torch.isfinite(loss)
        assert logits.shape == (len(x), 5)

        loss.backward()

        assert all(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )

        optimizer.step()

        model.eval()

        vx, vy = next(iter(val_loader))

        with torch.no_grad():
            vx = normalizer.transform(
                vx.to(device)
            )

            v_logits = model(vx)

            v_loss = val_criterion(
                v_logits,
                vy.to(device),
            )

            v_prob = torch.sigmoid(v_logits)

        assert torch.isfinite(v_loss)
        assert v_prob.shape == (len(vx), 5)
        assert torch.isfinite(v_prob).all()

        print("Training batch:", tuple(x.shape))
        print("Training loss (configured):", f"{loss.item():.6f}")
        print("Validation batch:", tuple(vx.shape))
        print(
            "Validation loss (unweighted):",
            f"{v_loss.item():.6f}",
        )
        print("Backward pass: PASS")
        print("Optimizer step: PASS")
        print("Validation inference: PASS")
        print("Test fold used for training/evaluation: NO")
        print("Artifacts written: NO")
        print("\nCNN TRAINING SMOKE TEST: PASSED")
        return

    # Full training: keep the weighted experiment separate from the baseline.
    experiment_name = (
        "cnn1d_posweighted_v1"
        if args.loss_weighting == "inverse_prevalence"
        else "cnn1d_baseline_v1"
    )
    output_dir = root / "artifacts" / experiment_name

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new run directory before retraining."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Store provenance and hyperparameters for independent reproduction.
    run_config = {
        "experiment": experiment_name,
        "architecture": "ECG1DCNN",
        "model_parameters": parameter_count,
        "class_order": CLASSES,
        "training_folds": list(range(1, 9)),
        "validation_fold": 9,
        "training_ecgs": len(train_dataset),
        "validation_ecgs": len(val_dataset),
        "loss_weighting": args.loss_weighting,
        "positive_counts": positive_counts.to(torch.int64).tolist(),
        "positive_weights": (
            positive_weights.tolist()
            if training_pos_weight is not None
            else None
        ),
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
    with (output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_config, f, indent=2, allow_nan=False)

    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            normalizer,
            train_criterion,
            optimizer,
            device,
        )

        val_loss, results, y_true, y_prob = validate(
            model,
            val_loader,
            normalizer,
            val_criterion,
            device,
        )

        summary = results["summary"]
        macro_ap = summary["macro_ap"]

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_ap": float(macro_ap),
            "val_macro_roc_auc": float(
                summary["macro_roc_auc"]
            ),
            "val_macro_f1": float(
                summary["macro_f1"]
            ),
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

        # Save the best model based on validation macro AP.
        if macro_ap > best_ap:
            best_ap = macro_ap
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_ap": best_ap,
                    "model_name": "ECG1DCNN",
                    "num_classes": 5,
                    "dropout": 0.3,
                    "class_order": CLASSES,
                    "loss_weighting": args.loss_weighting,
                    "positive_weights": (
                        positive_weights.tolist()
                        if training_pos_weight is not None
                        else None
                    ),
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

            with (
                output_dir / "best_metrics.json"
            ).open("w", encoding="utf-8") as f:
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

            print(
                "  New best validation macro AP — saved.",
                flush=True,
            )

        else:
            epochs_without_improvement += 1

        with (
            output_dir / "history.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after epoch {epoch}.",
                flush=True,
            )
            break

    print("\nFINAL TRAINING SUMMARY")
    print("=" * 60)
    print("Epochs completed:", len(history))
    print("Best epoch:", best_epoch)
    print("Best validation macro AP:", f"{best_ap:.4f}")
    print("Logistic regression benchmark: 0.6358")
    print("Difference:", f"{best_ap - 0.6358:+.4f}")
    print("Artifacts:", output_dir)
    print("Loss weighting:", args.loss_weighting)
    print("Test fold used for training/evaluation: NO")
    print("\nCNN TRAINING: COMPLETED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--loss-weighting",
        choices=["none", "inverse_prevalence"],
        default="none",
        help="Weight positive-class terms using training-only prevalence.",
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("Batch size and epochs must be positive.")

    if args.patience < 1 or args.learning_rate <= 0:
        parser.error("Patience and learning rate must be positive.")


    main(args)
