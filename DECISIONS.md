# Technical Decision Log

## D001 - Python Version

**Decision:** Use Python 3.12.10 for local development.

**Alternatives:** Python 3.11 and Python 3.13.

**Reason:** Python 3.12 provides a suitable environment for the selected scientific Python libraries. Compatibility with Colab and PyTorch environments is verified separately.

**Verification:** The local environment was created and the required dependencies were successfully imported.

## D002 - Development and Training Environments

**Decision:** Use VS Code for local development and Google Colab for GPU training.

**Alternatives:** Local CPU training or entirely notebook-based development.

**Reason:** Separating development from GPU training supports a scripted workflow while making use of Colab's computational resources.

**Verification:** Local development and Colab execution were tested. Access to a Tesla T4 GPU was previously confirmed. GPU-based model training has not yet begun.

## D003 - Dataset Storage

**Decision:** Store PTB-XL data and persistent training outputs in Google Drive.

**Alternatives:** Local laptop storage or repeated direct downloads from PhysioNet.

**Reason:** Persistent storage reduces repeated downloads and preserves data across temporary Colab sessions.

**Verification:** The full dataset was acquired and retained in Google Drive across multiple Colab sessions.

## D004 - Version Control

**Decision:** Use Git and GitHub to version-control source code, configuration and documentation.

**Alternatives:** Manually copying project files.

**Reason:** Git provides a traceable development history and allows Colab to execute the same source code maintained locally.

**Verification:** GitHub synchronization was tested, and the basic continuous integration workflow passed.

## D005 - Dataset Version and Sampling Rate

**Decision:** Use PTB-XL v1.0.3, beginning with its 100 Hz ECG recordings.

**Alternatives:** 500 Hz recordings.

**Reason:** The 100 Hz recordings retain the 12-lead, ten-second structure required by the project while reducing storage and processing demands.

**Verification:** All 21,799 recordings were acquired and passed the implemented waveform validation checks.

An independent inventory confirmed 21,799 matching `.hea` and `.dat` file pairs, with no missing or unexpected waveform files.

**Limitation:** The 100 Hz representation may omit higher-frequency information available in the 500 Hz recordings.

## D006 - Raw Data Preservation

**Decision:** Preserve the original PTB-XL WFDB recordings and store processed data separately.

**Alternatives:** Convert or overwrite the original recordings during acquisition.

**Reason:** Retaining the original files supports reproducibility, auditing and comparison of subsequent preprocessing methods.

**Verification:** Acquisition preserved the original `.hea` and `.dat` format. The complete stored-file inventory was verified.

## D007 - Resumable Dataset Acquisition

**Decision:** Implement cached-file reuse, temporary download files, retry handling and WFDB validation.

**Alternatives:** Download all recordings in one operation without recovery support.

**Reason:** Colab sessions may be interrupted, and re-downloading the complete dataset would waste time and bandwidth.

**Verification:** Five-record and 100-record acquisition tests passed. Missing-file recovery and truncated-file recovery were verified using isolated temporary directories.

The full acquisition was interrupted and subsequently completed using previously cached recordings.

**Limitation:** The current downloader revalidates cached ECGs during subsequent runs, creating overhead when processing thousands of files.

## D008 - Separate Colab Dependency Specification

**Decision:** Maintain `requirements-colab.txt` separately from the local `requirements.txt`.

**Alternatives:** Install the local requirements file directly into Colab.

**Reason:** Installing the original local requirements in Colab introduced package conflicts. A minimal Colab-specific dependency specification avoids unnecessary replacement of runtime-provided libraries.

**Verification:** The selected Colab environment passed `pip check` and a library interoperability test.

**Limitation:** The Colab requirements file is a supplementary dependency specification rather than a complete environment lockfile. Runtime-provided packages may change between Colab sessions.

## D009 - Reusable Colab Startup Procedure

**Decision:** Maintain a consolidated Colab startup notebook containing only environment initialization and lightweight readiness checks.

**Alternatives:** Repeated manual commands or a notebook that executes every acquisition and validation experiment during startup.

**Reason:** A small startup workflow reduces duplicated setup commands and prevents unintended repetition of time-consuming operations.

**Verification:** The consolidated notebook was tested successfully using `Runtime → Run all`.

Completed acquisition and recovery experiments were retained as Markdown documentation instead of executable startup cells.

## D010 - Dataset Integrity Verification

**Decision:** Use both acquisition-stage WFDB validation and an independent file inventory audit.

**Alternatives:** Rely exclusively on successful HTTP downloads or file counts.

**Reason:** File presence alone does not establish that a waveform can be decoded. Combining structural signal validation with file inventory checks provides stronger acquisition evidence.

**Verification:** All 21,799 recordings passed the implemented acquisition validation checks. The independent inventory identified no missing or unexpected WFDB file pairs.

**Limitation:** These checks do not establish diagnostic-label correctness or detect every possible form of signal corruption.