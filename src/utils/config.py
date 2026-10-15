"""Central paths and constants. Every module imports from here so train, eval and
the Streamlit app can never disagree on image size, class order or file locations."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "chest_xray"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

MODELS_DIR = ROOT / "models_saved"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"   # one JSON per model -> model_comparison.md

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

# Order matters: index i in every softmax output == CLASS_NAMES[i]
CLASS_NAMES = ["NORMAL", "PNEUMONIA_BACTERIA", "PNEUMONIA_VIRUS"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Deep learning
IMG_SIZE = 224
BATCH_SIZE = 32
USE_CLAHE = True          # contrast-limited histogram equalisation, applied in train AND inference

# EfficientNetV2 resolution study (A)
EFFNET_VARIANT = "B0"
EFFNET_RESOLUTIONS = (224, 384, 512)
EFFNET_UNFREEZE_FROM = "block5a"   # stage 2 fine-tunes block5a..top
EFFNET_BATCH_SIZE = 16             # same at every resolution (512px at 32 risks running out of memory)

# Conformal prediction (C): miscoverage alpha -> sets cover the true class with prob >= 1 - alpha
CONFORMAL_ALPHA = 0.10
CONFORMAL_ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
CONFORMAL_METHOD = "lac"           # score used by the app: "lac", "aps" or "lac_classwise"

# Classical ML
ML_IMG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

# Split
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Grad-CAM: final activation of DenseNet121 (after the last BN). Verify with model.summary().
GRADCAM_LAYER = "relu"

# Selective prediction: below this confidence the app says "refer to radiologist"
DEFAULT_ABSTAIN_THRESHOLD = 0.70

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
