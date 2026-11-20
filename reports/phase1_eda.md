# Phase 1 — Exploratory Data Analysis Report

**Project:** Deep Learning-Based Classification of Cardiac Abnormalities from 12-Lead ECG Signals
**Course:** DSA 8401 — Applied Machine Learning
**Dataset:** PTB-XL v1.0.3, 100 Hz

## 1. Objective

I conducted exploratory data analysis to understand the structure, diagnostic annotations, class distributions and selected signal-quality characteristics of PTB-XL before developing the preprocessing and classification pipelines.

## 2. Dataset Structure

The dataset contains 21,799 ECG recordings from 18,869 unique patients, with 28 metadata columns.

All recording identifiers are unique. Essential fields, including ECG identifiers, patient identifiers, SCP annotations, fold assignments and waveform paths, contain no missing values.

Some auxiliary metadata fields have substantial missingness, including height (68.0%) and weight (56.8%). These fields are not required as inputs for the planned waveform-based classification model.

## 3. Patient-Level Data Partitioning

I verified the predefined PTB-XL folds using the proposed partitioning:

| Partition  | Folds | Recordings | Unique patients |
| ---------- | ----- | ---------: | --------------: |
| Training   | 1–8   |     17,418 |          15,023 |
| Validation | 9     |      2,183 |           1,942 |
| Testing    | 10    |      2,198 |           1,904 |

No patients appeared in more than one fold, and no patient overlap was identified between the training, validation and test partitions.

These findings support the use of the predefined folds for patient-disjoint model evaluation.

## 4. Diagnostic Annotation Analysis

The SCP reference table contains 71 statements, of which 44 map to five diagnostic superclasses: NORM, MI, STTC, CD and HYP.

Using the presence of mapped diagnostic SCP statements, I obtained the following recording-level distributions:

| Class | Recordings | Percentage |
| ----- | ---------: | ---------: |
| NORM  |      9,514 |      43.6% |
| MI    |      5,469 |      25.1% |
| STTC  |      5,235 |      24.0% |
| CD    |      4,898 |      22.5% |
| HYP   |      2,649 |      12.2% |

The class frequencies indicate an imbalanced classification problem. Since categories overlap, their percentages do not sum to 100%.

### 4.1 Label Multiplicity

The analysis identified:

- 411 recordings with no mapped diagnostic superclass.
- 16,244 recordings with one superclass.
- 4,068 recordings with two superclasses.
- 919 recordings with three superclasses.
- 157 recordings with four superclasses.

Overall, 5,144 recordings (23.6%) contain more than one diagnostic superclass, supporting the use of multi-label classification.

### 4.2 Unmapped Diagnostic Recordings

The 411 recordings without a mapped diagnostic superclass predominantly contain rhythm or waveform-form statements, including PACE, AFIB and AFLT.

These recordings should not automatically be assigned NORM or interpreted as negative examples for all five diagnostic classes.

For the primary supervised five-superclass task, I provisionally propose excluding these recordings while preserving them in the original dataset.

This would leave 21,388 eligible recordings, subject to verification during label preparation.

### 4.3 NORM and Abnormal-Class Overlap

I identified 445 recordings where NORM coexists with at least one abnormal diagnostic superclass.

The overlap occurs predominantly with CD (415 recordings). Several examples contain high likelihood values for both NORM and a conduction-disturbance statement.

I propose retaining the mapped labels in the primary multi-label analysis and documenting these combinations for possible sensitivity analysis.

## 5. ECG Signal Exploration

I loaded and visualized Lead II from three selected recordings, then conducted a reproducible structural screening of 40 recordings sampled from training folds 1–8.

All 40 sampled recordings passed the following checks:

- Expected dimensions of 1,000 samples × 12 leads.
- Sampling frequency of 100 Hz.
- Expected lead order after case-insensitive normalization.
- Finite numerical values.
- No constant leads.

The maximum absolute amplitude across the sampled recordings ranged from 1.066 to 4.426 mV.

These checks establish structural consistency within the sample but do not establish that all recordings are artifact-free.

## 6. Signal-Quality Annotations

The metadata includes fields for baseline drift, static noise, burst noise and electrode problems.

After checking for empty placeholders, the analysis identified:

| Annotation         | Meaningful entries |
| ------------------ | -----------------: |
| Baseline drift     |              1,598 |
| Static noise       |              3,260 |
| Burst noise        |                612 |
| Electrode problems |                 30 |

A total of 5,010 recordings (23.0%) contain at least one meaningful quality flag, while 16,789 have no populated meaningful quality flag.

Among all recordings, 478 contain more than one quality-flag type.

The absence of a quality flag does not independently establish that a waveform is free from artifacts.

## 7. Implications for Model Development

The EDA findings support the following proposed approach:

1. Use the verified patient-disjoint folds for model development and evaluation.
2. Treat the task as multi-label classification with five diagnostic outputs.
3. Provisionally exclude recordings without mapped diagnostic superclasses from supervised classification.
4. Preserve NORM overlaps and other multi-label combinations in the primary analysis.
5. Use training data only when estimating preprocessing parameters or selecting class weights.
6. Evaluate class-specific performance alongside aggregate metrics because of the observed class imbalance.
7. Retain signal-quality annotations to support subgroup analysis.

These are proposed methodological decisions. Their implementation and resulting dataset counts will be verified in subsequent phases.

## 8. Limitations

This EDA relies on the supplied PTB-XL metadata and a limited structural waveform sample. It does not provide comprehensive automated artifact detection, independent clinical validation of diagnoses or confirmation of the eventual model's predictive performance.

## 9. Conclusion

The EDA established the dataset structure, verified patient-disjoint partitions, characterized multi-label diagnostic annotations and confirmed structural consistency in a sample of ECG signals.

The findings provide a basis for implementing a reproducible label-preparation and preprocessing pipeline while explicitly addressing annotation eligibility, class imbalance and signal-quality limitations.
