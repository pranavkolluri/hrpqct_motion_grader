# motion_v6 — Design Document

## Overview

motion_v6 is a motion grade classification system for HR-pQCT scans. It uses a two-stage cascade neural network to classify scans into three groups (Grade 1, Grade 2, Grade 3-5) and provides an operator-facing GUI for reviewing and correcting predictions. Operator corrections are logged and can be used to fine-tune the model over time.

The system is split into two audiences:

- **Operators** use a graphical interface to load a scan list, review machine predictions, approve or correct grades, and export results.
- **Administrators / researchers** use command-line tools to set up the environment, build training datasets, train models, and incorporate operator corrections into future training runs.

---

## Grade System

The model classifies scans on a 3-bin scale internally, but operators always work in the conventional 1-5 scale.

| Model bin | Grade range | Meaning |
|-----------|-------------|---------|
| 0 | Grade 1 | Minimal motion artifact |
| 1 | Grade 2 | Mild motion artifact |
| 2 | Grade 3-5 | Significant motion artifact |

The bin labels shown to operators are configurable in `config.py` under `GRADE_BIN_LABELS`.

---

## Cascade Architecture

The classifier uses two binary stages rather than a single three-class model. This is motivated by the severe class imbalance in the training data (roughly 84% Grade 1 scans).

**Stage 1** — trained on all scans, predicts Grade 1 vs. Rest.

**Stage 2** — trained only on Grade 2-5 scans, predicts Grade 2 vs. Grade 3-5. Only runs if Stage 1 predicts "Rest".

Confidence is reported as the average softmax probability of the winning class across all slices. Scans below the threshold set in `config.py` are flagged for operator attention.

Both stages use EfficientNet-B0 with a grayscale-adapted stem (3-channel weights averaged to 1-channel). A Sobel edge filter is optionally applied before each forward pass to highlight structural blur patterns consistent with how technicians manually assess motion.

---

## Folder Structure

```
motion_v6/
├── config.py                   Single configuration file for all paths and settings
├── run_grader.bat              Operator launcher (double-click)
├── setup_env.bat               First-time environment setup (run once)
├── requirements.txt            Python dependencies (torch installed separately)
├── DESIGN.md                   This document
│
├── app/
│   ├── grading_app.py          Operator GUI (CustomTkinter)
│   ├── cascade_infer.py        Inference engine (live ISQ or pre-processed .npy)
│   └── accumulation_log.py     Append-only corrections log manager
│
├── training/
│   ├── train_cascade.py        Main training entry point
│   ├── train_models.py         General non-cascade training
│   ├── trainer.py              Training loop, k-fold CV, fine-tuning, cascade evaluation
│   ├── dataset.py              PyTorch dataset, Sobel filter, k-fold splitting
│   ├── model.py                Model architectures and fine-tune mode helpers
│   ├── prepare_dataset.py      Build or update .npy dataset from ISQs and corrections
│   ├── analyze_data_distribution.py  Grade distribution analysis utility
│   └── rebuild_scores.py       Regenerate scores.xlsx from a resolved CSV
│
├── data/
│   ├── isq_reader.py           Read ISQ files via itk-ioscanco
│   ├── bone_cropper.py         Per-slice bone detection and centroid cropping
│   ├── csv_path_resolver.py    Map sample/measurement numbers to ISQ paths on Y:\
│   └── isq_data_builder.py     Convert ISQ files to .npy training slices
│
├── models/
│   ├── tibia/stage1/           Trained Stage 1 Tibia models (kfold dirs)
│   ├── tibia/stage2/           Trained Stage 2 Tibia models
│   ├── radius/stage1/          Trained Stage 1 Radius models
│   └── radius/stage2/          Trained Stage 2 Radius models
│
└── logs/
    └── grading_log.csv         Cumulative operator grading history (auto-created)
```

---

## Configuration

All system-wide settings live in `config.py`. This is the only file that needs to be edited for deployment or reconfiguration.

```python
ISQ_ROOT = r"Y:\data"               # Mounted XCT drive root
ARCH = "efficientnet"               # Model architecture used during training
EDGE_FILTER = True                  # Whether Sobel filter was used during training
CROP_SIZE = 1024                    # ISQ crop size in pixels

MODELS = {
    "Tibia": {
        "stage1": r"models/tibia/stage1",
        "stage2": r"models/tibia/stage2",
    },
    "Radius": {
        "stage1": r"models/radius/stage1",
        "stage2": r"models/radius/stage2",
    },
}

GRADE_BIN_LABELS = {
    0: "Grade 1  (minimal motion)",
    1: "Grade 2  (mild motion)",
    2: "Grade 3-5  (significant motion)",
}

LOW_CONFIDENCE_THRESHOLD = 0.65    # Below this, row is flagged yellow in review table
LOG_FILE = r"logs/grading_log.csv"
DEBUG_DATASET_DIR = None           # Set to a .npy dataset path to enable debug mode
```

The `ARCH` and `EDGE_FILTER` values must match what was used during training. Mismatches will not cause an error at load time but will produce incorrect predictions.

---

## ISQ File Location Convention

Scans live at:

```
Y:\data\{sample_number:08d}\{measurement_number:08d}\
```

For example, sample 362, measurement 1201 resolves to `Y:\data\00000362\00001201\`. The system looks for any file with a `.ISQ` or `.ISQ;*` extension in that directory.

---

## Operator Guide

### First-time setup (done once by an administrator)

1. Ensure Python 3.10 or later is installed and on the system PATH.
2. Double-click `setup_env.bat`. This creates a `.venv` directory and installs all dependencies including PyTorch with CUDA support. This takes a few minutes and requires an internet connection.
3. Ensure the trained model files are in place under `models/tibia/` and `models/radius/`. If they are not present, the app will show an error when processing begins.

### Launching the application

Double-click `run_grader.bat`. No terminal or command prompt is needed.

### Loading a scan list

The application expects a CSV file with at minimum these columns:

- `Participant ID` — used to identify bone type (must contain "Tibia" or "Radius")
- `Sample Number` — the 1-8 digit sample number
- `Measurement Number` — the 1-8 digit measurement number

A grade column is not required. If one is present it is ignored; the machine prediction is always used as the starting point for review.

Click **Browse** to select the file, then **Start Processing**. The app will locate each scan on `Y:\data`, read the ISQ file, run inference, and populate the review table.

### Reviewing predictions

Each row in the review table shows:

| Column | Description |
|--------|-------------|
| Participant ID | From the input CSV |
| Bone | Tibia or Radius (extracted from Participant ID) |
| Machine Grade | The model's prediction (bin label) |
| Conf. | Model confidence as a percentage |
| Operator Grade | The grade that will be exported (starts as machine suggestion) |
| Status | Pending / Approved / Modified / Error |

Rows with confidence below the threshold in `config.py` are highlighted yellow as a prompt to review them carefully.

Clicking a row opens the edit panel on the right. Select a grade (1-5) using the radio buttons and optionally add a note. Then:

- **Approve** — confirms the grade as-is and moves to the next scan
- **Override** — saves a different grade (marked as "Modified" if it differs from the machine suggestion)

Use the **Prev / Next** buttons or click any row in the table to navigate.

### Exporting results

Click **Export Results** at the bottom of the review screen. A file dialog will prompt for a save location. The output CSV contains:

| Column | Description |
|--------|-------------|
| `participant_id` | From input CSV |
| `sample_number` | Sample number |
| `measurement_number` | Measurement number |
| `bone_type` | Tibia or Radius |
| `machine_grade_bin` | Model output bin (0, 1, or 2) |
| `machine_grade_label` | Human-readable bin label |
| `operator_grade` | Final grade in 1-5 scale |
| `operator_approved` | True if accepted as-is, False if changed |
| `operator_notes` | Any notes entered during review |
| `confidence` | Model confidence percentage |
| `graded_at` | Timestamp |

At the same time, all rows are appended to `logs/grading_log.csv`. This file is cumulative — it grows with every export session and is used as the source of corrections for future model fine-tuning.

### Error rows

Scans that could not be processed (ISQ file missing, read error) appear with an "Error" status and an error description in the Machine Grade column. They are excluded from the export CSV but shown in the table for awareness.

---

## Administrator and Research Guide

### Python environment

All training scripts must be run with the virtual environment active:

```powershell
.venv\Scripts\activate.bat
```

Or prefix calls with the venv Python directly:

```powershell
.venv\Scripts\python.exe training\train_cascade.py ...
```

### Training from a pre-processed dataset

If the `.npy` dataset already exists (i.e., `scores.xlsx` is present in the dataset directory):

```powershell
python training\train_cascade.py `
  --dataset_dir F:\data\training_dataset `
  --bone_type Tibia `
  --arch efficientnet `
  --edge_filter `
  --n_folds 5 `
  --num_models 1 `
  --num_epochs_s1 15 `
  --num_epochs_s2 20
```

This trains Stage 1 (Grade 1 vs Rest) and Stage 2 (Grade 2 vs Grade 3-5) using stratified k-fold cross-validation, then runs a full cascade evaluation with no data leakage (each scan is evaluated by the fold model that never saw it during training). Results are saved under `models/ensemble/`.

### Building a dataset from ISQ files

If no `.npy` dataset exists yet, add `--build_dataset` and point to the input CSV:

```powershell
python training\train_cascade.py `
  --dataset_dir F:\data\training_dataset `
  --build_dataset `
  --input_csv "F:\Motion grades spreadsheet for Pranav - CKD.csv" `
  --bone_type Tibia `
  --arch efficientnet `
  --edge_filter
```

The script will resolve ISQ paths, convert each scan's slices to `.npy` files, write `scores.xlsx`, then proceed directly into training.

To build a dataset without immediately training:

```powershell
python training\prepare_dataset.py `
  --dataset_dir F:\data\training_dataset `
  --input_csv "path\to\labeled.csv"
```

### scores.xlsx format

The file `scores.xlsx` in the dataset directory maps each scan to its grade. The `Slice` column must contain scan folder names in the format `s{sample:08d}_m{measurement:08d}` (e.g. `s00000362_m00001201`), not individual slice filenames. One row per scan. The `isq_data_builder.py` in this project generates this format correctly.

If you have an older `scores.xlsx` that used slice filenames, regenerate it using:

```powershell
python training\rebuild_scores.py `
  --csv F:\data\resolved_all_ok.csv `
  --dataset_dir F:\data\training_dataset
```

### Incorporating operator corrections into training

After grading sessions have accumulated in `logs/grading_log.csv`, add the corrected scans to the training dataset:

```powershell
python training\prepare_dataset.py `
  --dataset_dir F:\data\training_dataset `
  --corrections F:\motion_v6\logs\grading_log.csv
```

This reads ISQ files for any corrected scan not already in the dataset, processes them into `.npy` slices, and appends to `scores.xlsx`. Scans already present are skipped (no duplicates).

### Fine-tuning existing models

Fine-tuning is preferred over full retraining when incorporating a modest number of new corrections (tens of scans rather than hundreds). It runs in two phases: head-only training at a higher learning rate, followed by full-model training at a much lower learning rate.

```powershell
python training\train_cascade.py `
  --dataset_dir F:\data\training_dataset `
  --finetune `
  --stage1_pretrained models\tibia\stage1 `
  --stage2_pretrained models\tibia\stage2 `
  --bone_type Tibia `
  --arch efficientnet `
  --edge_filter `
  --num_head_epochs_s1 5 `
  --num_head_epochs_s2 5 `
  --num_epochs_s1 10 `
  --num_epochs_s2 15
```

The `--stage1_pretrained` and `--stage2_pretrained` paths should point to the kfold output directories produced by a previous training run (containing `fold_0/`, `fold_1/`, etc.). Each fold's model is fine-tuned from its corresponding pretrained fold.

Use full retraining (omit `--finetune`) when the correction volume is large enough to substantially change the class distribution, or when model performance has degraded significantly.

### Deploying trained models

After training, copy the two kfold output directories to the model locations configured in `config.py`:

```
models/tibia/stage1/   <- stage 1 kfold dir (contains fold_0/, fold_1/, config.json, ...)
models/tibia/stage2/   <- stage 2 kfold dir
```

Ensure the `ARCH` and `EDGE_FILTER` values in `config.py` match what was used during that training run (recorded in `config.json` inside each kfold dir).

### Debug mode

Debug mode runs the full operator GUI workflow using pre-processed `.npy` files instead of live ISQ reads. It is intended for testing the app without access to the XCT machine or the `Y:\` drive.

1. Set `DEBUG_DATASET_DIR` in `config.py` to the path of a `.npy` dataset directory.
2. Launch with:

```powershell
run_grader.bat --debug
```

The app will behave identically to normal mode except ISQ files are not read. Scans in the CSV must have corresponding folders in `DEBUG_DATASET_DIR`.

### Analyzing data distribution

Before training, it is useful to check the grade distribution across anatomical sites:

```powershell
python training\analyze_data_distribution.py --csv "path\to\labeled.csv"
```

This prints grade counts by bone type and site (UD Tibia, D Radius, etc.) and helps inform decisions about bone-type stratification.

---

## Key Design Decisions

**Why a cascade classifier rather than a single 3-class model?**
The training data has approximately 84% Grade 1 scans, 11% Grade 2, and 5% Grade 3-5. A single multi-class model trained on this distribution consistently collapses to predicting the majority class. The cascade approach decomposes the problem into two more tractable binary problems with better class balance (roughly 5:1 for Stage 1, 2:1 for Stage 2).

**Why EfficientNet rather than the custom CNN?**
EfficientNet-B0 pretrained on ImageNet extracts richer texture features from limited training data than a network trained from scratch on ~300 scans. The stem convolution is adapted from 3-channel to 1-channel by averaging pretrained weights, preserving the learned edge and texture detectors rather than random-initialising them.

**Why a Sobel edge filter?**
Technicians assess motion grade primarily by examining structural edge sharpness. Motion artifacts manifest as blur and streaking at bone boundaries. The Sobel filter makes this signal explicit to the model and is consistent with the manual assessment process.

**Why k-fold cross-validation rather than a single train/test split?**
With ~300 scans total, a single split wastes too much data on evaluation and produces unstable accuracy estimates. K-fold ensures every scan is evaluated exactly once by a model that never saw it, giving a reliable estimate of generalisation performance across the full dataset.

**Why on-the-fly ISQ reading rather than pre-processing to disk?**
Storage constraints on the operator workstation make caching processed slices impractical. ISQ files on the XCT drive are read, cropped, and normalised in memory during inference. This is slower (a few seconds per scan) but requires no disk space beyond the model weights.
