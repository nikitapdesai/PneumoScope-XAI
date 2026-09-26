"""Classical ML baselines on HOG features: Logistic Regression and Random Forest.

Usage: python -m src.models.baseline_ml
"""
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import load_hog
from src.evaluation.metrics import evaluate, save_probs
from src.utils import config as C


def build_models(n_train: int) -> dict:
    return {
        "logreg": ("Logistic Regression (HOG)", make_pipeline(
            StandardScaler(), PCA(n_components=min(256, n_train - 1), random_state=C.SEED),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))),
        "random_forest": ("Random Forest (HOG)", RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
            n_jobs=-1, random_state=C.SEED)),
    }


def main():
    import pandas as pd
    df = pd.read_csv(C.MANIFEST_PATH)
    df["abspath"] = df["filepath"].map(lambda p: str(C.ROOT / p))

    X_tr, y_tr = load_hog(df, "train")
    X_va, _ = load_hog(df, "val")
    X_te, y_te = load_hog(df, "test")
    print(f"HOG dims: {X_tr.shape[1]}")

    for key, (pretty, model) in build_models(len(y_tr)).items():
        model.fit(X_tr, y_tr)
        joblib.dump(model, C.MODELS_DIR / f"{key}.pkl")
        save_probs(key, "val", model.predict_proba(X_va))
        p_te = model.predict_proba(X_te)
        save_probs(key, "test", p_te)
        evaluate(key, y_te, p_te, pretty)


if __name__ == "__main__":
    main()
