"""One place that knows every deep model: its file, input size and Grad-CAM layer.

Training (train_dl.py), reporting (compare.py, gradcam_report.py, conformal.py)
and the app (inference.py) all look models up here, so adding a model or a
resolution never means editing several hard-coded lists.
"""
from dataclasses import dataclass

from src.models import cnn_scratch, efficientnet_transfer
from src.utils import config as C


@dataclass(frozen=True)
class ModelSpec:
    key: str             # prefix of reports/results/<key>_{val,test}_probs.npy
    name: str            # human-readable
    file: str | None     # models_saved/<file>; None for sklearn models
    img_size: int
    gradcam_layer: str | None
    note: str
    deployable: bool     # final (fine-tuned) model that may be served by the app


def effnet_key(res: int, stage: str, variant: str = C.EFFNET_VARIANT) -> str:
    return f"effnetv2_{variant.lower()}_{res}_{stage}"


def _build() -> list[ModelSpec]:
    specs = [
        ModelSpec("logreg", "Logistic Regression (HOG)", None, C.ML_IMG_SIZE, None, "classical baseline", False),
        ModelSpec("random_forest", "Random Forest (HOG)", None, C.ML_IMG_SIZE, None, "classical baseline", False),
        ModelSpec("cnn_scratch", "CNN (scratch)", "cnn_scratch.keras", C.IMG_SIZE,
                  cnn_scratch.GRADCAM_LAYER, "learned features", True),
        ModelSpec("densenet_frozen", "DenseNet121 (frozen)", "densenet121_frozen.keras", C.IMG_SIZE,
                  C.GRADCAM_LAYER, "transfer, stage 1", False),
        ModelSpec("densenet_finetuned", "DenseNet121 (fine-tuned)", "densenet121_finetuned.keras", C.IMG_SIZE,
                  C.GRADCAM_LAYER, "transfer, stage 2", True),
    ]
    v = C.EFFNET_VARIANT
    for res in C.EFFNET_RESOLUTIONS:
        for stage, label, deployable in (("frozen", "frozen", False), ("finetuned", "fine-tuned", True)):
            k = effnet_key(res, stage)
            specs.append(ModelSpec(k, f"EfficientNetV2-{v} {res}px ({label})", f"{k}.keras", res,
                                   efficientnet_transfer.GRADCAM_LAYER,
                                   f"resolution study, stage {1 if stage == 'frozen' else 2}", deployable))
    return specs


SPECS: list[ModelSpec] = _build()
BY_KEY: dict[str, ModelSpec] = {s.key: s for s in SPECS}
