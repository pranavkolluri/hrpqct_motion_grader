"""
motion_v6 configuration — edit this file to configure paths, model locations, and labels.
"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── ISQ data source ───────────────────────────────────────────────────────────
# Mounted XCT drive root. Scans live at ISQ_ROOT/<sample:08d>/<measurement:08d>/
ISQ_ROOT = r"Y:\data"

# ── Trained model locations ───────────────────────────────────────────────────
# Each stage dir must contain fold_<n>/model_<i>.pt (from kfold_ensemble_trainer)
# or model_<i>.pt directly (from ensemble_trainer).
MODELS = {
    "Tibia": {
        "stage1": os.path.join(_ROOT, "models", "tibia", "stage1"),
        "stage2": os.path.join(_ROOT, "models", "tibia", "stage2"),
    },
    "Radius": {
        "stage1": os.path.join(_ROOT, "models", "radius", "stage1"),
        "stage2": os.path.join(_ROOT, "models", "radius", "stage2"),
    },
}

# ── Model architecture ────────────────────────────────────────────────────────
# Must match the arch used during training. 'efficientnet' | 'convnext' | 'custom'
ARCH = "efficientnet"

# ── Preprocessing ─────────────────────────────────────────────────────────────
EDGE_FILTER = True       # Apply Sobel edge filter (must match training setting)
EDGE_BLEND = 0.5         # Edge-vs-raw blend weight (must match the deployed model's config.json)
EDGE_LEGACY = False      # True only for models trained before the blend/fixed-scale change
CROP_SIZE = 1024         # ISQ crop size in pixels

# 'mean' (per-slice predictions averaged) or 'attention' (2.5D MIL, learned
# slice weighting) — must match the deployed model's config.json
AGGREGATION = "mean"
Z_CONTEXT = 3            # neighboring-slice stack depth, attention mode only
MIL_INPUT_SIZE = 224     # instance resize resolution, attention mode only

# ── Grade bin labels (shown to operators) ─────────────────────────────────────
# Keys are model output bins (0, 1, 2). Change the text here if needed.
GRADE_BIN_LABELS = {
    0: "Grade 1  (minimal motion)",
    1: "Grade 2  (mild motion)",
    2: "Grade 3–5  (significant motion)",
}

# ── Inference device ──────────────────────────────────────────────────────────
# 'cpu' is recommended for the grading app — inference on individual scans is
# fast enough and avoids CUDA version mismatch issues on operator workstations.
# Set to 'cuda' only if you have a matching PyTorch/CUDA build and need speed.
INFERENCE_DEVICE = "cuda"

# ── Confidence threshold ──────────────────────────────────────────────────────
# Scans below this confidence are flagged yellow in the review table.
LOW_CONFIDENCE_THRESHOLD = 0.65

# ── Operator corrections log ──────────────────────────────────────────────────
LOG_FILE = os.path.join(_ROOT, "logs", "grading_log.csv")

# ── Debug / development mode ──────────────────────────────────────────────────
# Set to a path containing pre-processed .npy slices to bypass live ISQ reading.
# Folder layout must match: DEBUG_DATASET_DIR/s<sample:08d>_m<measurement:08d>/*.npy
# Set to None (or leave empty string) to use live ISQ reading.
DEBUG_DATASET_DIR = r"F:\data\training_dataset"
