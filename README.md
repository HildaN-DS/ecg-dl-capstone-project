# Deep Learning-Based Classification of Cardiac Abnormalities from 12-Lead ECG Signals

**DSA 8401 – Applied Machine Learning | MSc Data Science & Analytics | Strathmore University**  
**Student:** Hilda Ndambuki | **Registration No.:** 053837

> **Research use only.** This prototype is not validated for clinical diagnosis, triage, treatment, or patient-care decisions. Use only synthetic or appropriately authorized, de-identified ECG data.

## Project overview

This capstone implements **multi-label classification of 12-lead ECG signals** using the PTB-XL v1.0.3 dataset. Each input is a 10-second ECG sampled at 100 Hz (12 leads × 1,000 samples). The selected, frozen **1D convolutional neural network (CNN)** produces independent scores for five diagnostic superclasses: **NORM, MI, STTC, CD, and HYP**. The fixed decision threshold is 0.50 per class. More than one label may be selected; an absence of positive labels must **not** be interpreted as NORM.

The repository provides data-processing and training code, the serialized CNN model and normalization artifacts, a **FastAPI** inference service, a **Docker Compose** setup, automated tests, a synthetic ECG sample, the final report, and an AI-assistance disclosure. Original PTB-XL recordings are **not** redistributed.

## Approach and results

The eligible waveform dataset comprised **21,388 recordings**, partitioned by patient into training (folds 1–8: **17,084**), validation (fold 9: **2,146**), and held-out testing (fold 10: **2,158**). Normalization was fitted on training data only. Model selection used validation macro average precision (AP); the test set was reserved for final evaluation.

| Model / metric | Result |
|---|---:|
| Constant-prevalence reference — validation macro AP | 0.2596 |
| Logistic regression (108 features) — validation macro AP | 0.6358 |
| **Unweighted 1D CNN (selected, epoch 17)** — validation macro AP | **0.7997** |
| Class-weighted CNN — validation macro AP | 0.7988 |
| Extended BiLSTM — validation macro AP | 0.7932 |
| **Selected CNN — held-out macro AP** | **0.8042** |
| Held-out macro AP — 95% patient-bootstrap CI | 0.7859–0.8199 |
| Held-out macro ROC-AUC | 0.9147 |
| Held-out macro F1 | 0.7104 |

These are **research evaluation results**, not guarantees of performance in clinical settings.


## Run the frozen model locally

**Requirements:** Docker Desktop with Linux containers and Python 3.12. The repository includes the four model files under `release_artifacts/`:

```text
release_artifacts/
├── final_model_selection_v1.json
├── ptbxl_normalization_v1.json
├── cnn1d_baseline_v1/best_model.pt
└── final_test_cnn1d_v1/test_metrics.json
```

From the repository root in **Windows PowerShell**:

```powershell
$env:ECG_ARTIFACTS_HOST_DIR = (Resolve-Path .\release_artifacts).Path
docker compose -f compose.api.yaml up -d --build
docker compose -f compose.api.yaml ps
Invoke-RestMethod http://127.0.0.1:8000/health
python scripts/predict_sample.py --validate-only
python scripts/predict_sample.py
```

The included `samples/sample_ecg_synthetic.csv` is a **nonpatient** test input. `scripts/predict_sample.py` validates it and submits it directly to `POST /v1/predict`; no graphical frontend is required. The API also exposes `GET /health`. For the automated container smoke test, run `python scripts/smoke_docker_api.py`. Stop the service using `docker compose -f compose.api.yaml down`.

The selected checkpoint SHA-256 is:

```text
7bd6298ad5840de15977b25b3f8b9ed494ba633e2f19c682716bbe2c128a6176
```

## Data Pipeline and Model Training

The project uses [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), which can be downloaded independently from PhysioNet. The dataset is not included in this repository.

The reproducible implementation is organized as follows:

| Component | Location |
|---|---|
| Data acquisition and preprocessing | `src/ecg/data/` |
| Feature extraction and normalization | `src/ecg/features/` |
| Model architectures | `src/ecg/models/` |
| Model training and evaluation | `scripts/` and `src/ecg/evaluate/` |
| Experiment configuration | `configs/` |

Training was conducted using Google Colab with patient-disjoint PTB-XL folds: 1–8 for training, 9 for validation and 10 for testing.

Full retraining requires the external PTB-XL dataset, preprocessing outputs and training-fitted artifacts. These are not all included in the submission. The provided frozen CNN checkpoint and normalization artifacts support reproducible inference without retraining.

For immediate verification, use the Docker API and synthetic ECG sample described in the Quick Start section.

## Repository contents

| Path | Purpose |
|---|---|
| `src/ecg/data/`, `src/ecg/features/` | Dataset acquisition, labels, validation, waveform preparation and features |
| `src/ecg/models/`, `scripts/train_*.py` | CNN, LSTM and baseline training |
| `src/ecg/evaluate/`, `tests/` | Evaluation utilities and automated checks |
| `src/ecg/inference/`, `release_artifacts/` | Frozen inference and serialized model files |
| `src/ecg/api/main.py` | FastAPI service |
| `Dockerfile.api`, `compose.api.yaml` | Container build and local deployment |
| `samples/`, `scripts/predict_sample.py` | Synthetic ECG and direct API test |
| `reports/` | Final LaTeX report source and compiled PDF |
| `AI_USE.md` | AI assistance disclosure |

## References

- Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7, 154. [DOI: 10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
