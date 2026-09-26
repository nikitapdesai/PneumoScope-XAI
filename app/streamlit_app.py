"""Streamlit diagnostic-assistant demo.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app.inference import Predictor
from src.utils import config as C

PRETTY = {"NORMAL": "Normal", "PNEUMONIA_BACTERIA": "Bacterial pneumonia", "PNEUMONIA_VIRUS": "Viral pneumonia"}
SAMPLES_DIR = ROOT / "app" / "assets" / "samples"

st.set_page_config(page_title="Chest X-ray Pneumonia Classifier", page_icon="🫁", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    return Predictor()


st.title("🫁 Chest X-ray: Normal / Bacterial / Viral pneumonia")
st.error("**Educational demo only.** This tool is NOT a medical device and must not be used for diagnosis. "
         "Always consult a qualified clinician.", icon="⚠️")

tab_predict, tab_results, tab_about = st.tabs(["Diagnose", "Model comparison", "About"])

with tab_predict:
    left, right = st.columns([1, 2])
    with left:
        upload = st.file_uploader("Upload a chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"])
        samples = sorted(SAMPLES_DIR.glob("*.*")) if SAMPLES_DIR.exists() else []
        choice = None
        if samples:
            choice = st.selectbox("…or try a sample", ["—"] + [s.name for s in samples])

    source = upload or (SAMPLES_DIR / choice if choice and choice != "—" else None)
    if source is None:
        st.info("Upload an X-ray to get a prediction and a Grad-CAM heatmap.")
    else:
        try:
            predictor = get_predictor()
        except FileNotFoundError as e:
            st.warning(str(e))
            st.stop()

        image = np.array(Image.open(source).convert("L"))
        with st.spinner("Analysing…"):
            pred = predictor.predict(image)

        with left:
            if pred.abstain:
                st.warning(f"**Uncertain ({pred.confidence:.0%}).** Model leans towards *{PRETTY[pred.label]}*, "
                           f"but its confidence is below the {predictor.threshold:.0%} safety threshold. "
                           "Refer to a radiologist.")
            else:
                st.success(f"**{PRETTY[pred.label]}** ({pred.confidence:.0%} calibrated confidence)")
            st.bar_chart(pd.DataFrame({"probability": pred.probs},
                                      index=[PRETTY[c] for c in C.CLASS_NAMES]), horizontal=True)
            if pred.border_attention > 0.5:
                st.caption(f"⚠️ Focus check: {pred.border_attention:.0%} of the model's attention is near the image "
                           "border (labels, markers), not the lungs. Treat this prediction with extra caution.")
            else:
                st.caption(f"Focus check: {1 - pred.border_attention:.0%} of attention is on the central chest.")

        with right:
            c1, c2 = st.columns(2)
            c1.image(pred.model_input, caption="Pre-processed input (CLAHE, 224×224)", width="stretch")
            c2.image(pred.heatmap_overlay, caption="Grad-CAM: where the model looked", width="stretch")

SHORT_NAMES = {"logreg": "1 · LogReg", "random_forest": "2 · Random Forest", "cnn_scratch": "3 · CNN",
               "densenet_frozen": "4 · DenseNet frozen", "densenet_finetuned": "5 · DenseNet tuned"}


def pct(x):
    return f"{x * 100:.1f}%"


def figure(name, caption=None):
    path = C.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


with tab_results:
    summary_path = C.RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        st.info("Run `python -m src.evaluation.compare` to generate the comparison results.")
    else:
        S = json.loads(summary_path.read_text())
        models = S["models"]
        best = next(m for m in models if m["is_best"])
        runner_up = max((m for m in models if not m["is_best"]), key=lambda m: m["accuracy"])
        sel, cal = S["selective"], S["calibration"]

        st.subheader(f"Best model: {best['name']}")
        st.caption(f"Test set: {S['test_n']} X-rays from patients never seen in training.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", pct(best["accuracy"]),
                  f"{(best['accuracy'] - runner_up['accuracy']) * 100:+.1f} pts vs {runner_up['name']}", border=True)
        k2.metric("Macro-F1", f"{best['macro_f1']:.3f}",
                  f"95% CI {best['macro_f1_ci'][0]:.3f}–{best['macro_f1_ci'][1]:.3f}", delta_color="off",
                  delta_arrow="off", border=True)
        k3.metric("ROC-AUC (one-vs-rest)", f"{best['roc_auc_ovr']:.3f}",
                  f"Normal vs Pneumonia: {best['binary_auc']:.3f}", delta_color="off", delta_arrow="off", border=True)
        k4.metric("Accuracy when confident", pct(sel["accuracy_answered"]),
                  f"answers {pct(sel['coverage'])} of cases, refers the rest", delta_color="off",
                  delta_arrow="off", border=True)

        # ---------- all models ----------
        st.divider()
        st.subheader("All models")
        table = pd.DataFrame([{
            "Model": ("⭐ " if m["is_best"] else "") + m["name"],
            "Accuracy": m["accuracy"] * 100,
            "Accuracy 95% CI": f"{m['accuracy_ci'][0] * 100:.1f}–{m['accuracy_ci'][1] * 100:.1f}",
            "Macro-F1": m["macro_f1"],
            "ROC-AUC": m["roc_auc_ovr"],
            "ECE ↓": m["ece"],
            "p vs best": "—" if m["p_vs_best"] is None else f"{m['p_vs_best']:.1e}",
            "Type": m["note"],
        } for m in models])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=50, max_value=100),
            "Macro-F1": st.column_config.ProgressColumn(format="%.3f", min_value=0.5, max_value=1),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "ECE ↓": st.column_config.NumberColumn(format="%.3f", help="Expected calibration error: lower = more honest confidence"),
            "p vs best": st.column_config.TextColumn(help="McNemar test: p < 0.05 means the best model is significantly better"),
        })
        st.caption("⭐ = deployed model. Every other model is significantly worse (McNemar p < 0.01).")

        # ---------- where the difficulty is ----------
        st.divider()
        st.subheader("Where the difficulty actually is")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Two sub-tasks, very different difficulty**")
            split = pd.DataFrame({
                "Normal vs Pneumonia": [m["binary_accuracy"] * 100 for m in models],
                "Bacterial vs Viral": [m["subtype_accuracy"] * 100 for m in models],
            }, index=[SHORT_NAMES.get(m["key"], m["name"]) for m in models])
            st.bar_chart(split, horizontal=True, stack=False, x_label="", y_label="Accuracy (%)", height=320)
            st.caption(f"Spotting pneumonia is nearly solved ({pct(best['binary_accuracy'])} for the best model). "
                       f"Telling bacterial from viral is where every model struggles ({pct(best['subtype_accuracy'])}).")
        with c2:
            st.markdown("**F1 per class**")
            per_class = pd.DataFrame([{
                "Model": m["name"],
                "Normal": m["per_class_f1"]["NORMAL"],
                "Bacterial": m["per_class_f1"]["PNEUMONIA_BACTERIA"],
                "Viral": m["per_class_f1"]["PNEUMONIA_VIRUS"],
            } for m in models])
            bar = st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)
            st.dataframe(per_class, hide_index=True, width="stretch",
                         column_config={"Normal": bar, "Bacterial": bar, "Viral": bar})
            st.caption("Viral pneumonia is the weakest class for every model; radiologists also find it hard to separate from bacterial on a single X-ray.")

        # ---------- trust ----------
        st.divider()
        st.subheader("Can you trust its confidence?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Refer to radiologist when unsure**")
            st.markdown(f"Below **{sel['threshold']:.0%}** confidence the app refuses to answer. "
                        f"It then answers **{pct(sel['coverage'])}** of X-rays at **{pct(sel['accuracy_answered'])}** "
                        f"accuracy (vs {pct(sel['accuracy_all'])} when forced to answer everything).")
            figure("risk_coverage.png")
        with c2:
            st.markdown("**Calibration**")
            better = cal["ece_after"] < cal["ece_before"]
            st.markdown(f"Temperature T = **{cal['temperature']:.2f}**. ECE {cal['ece_before']:.3f} → {cal['ece_after']:.3f}. "
                        + ("Scaling made the confidence values more honest." if better else
                           "T ≈ 1, so the model was already well calibrated and scaling didn't help on test."))
            figure("reliability_best.png")

        # ---------- deeper figures ----------
        st.divider()
        best_key = best["key"]
        with st.expander("Confusion matrix and ROC curves (best model)"):
            c1, c2 = st.columns(2)
            with c1:
                figure(f"cm_{best_key}.png")
            with c2:
                figure(f"roc_{best_key}.png")
        with st.expander("Grad-CAM examples: correct and wrong predictions"):
            st.caption("Red = regions that drove the prediction. In one wrong case the model looked at the 'R' marker, not the lungs.")
            g1, g2, g3 = st.columns([1, 2, 1])
            with g2:
                figure(f"gradcam_grid_{best_key}.png")
        with st.expander("Training curves"):
            figure("history_densenet.png", "DenseNet121: stage 1 (frozen) then stage 2 (fine-tune)")
            figure("history_cnn_scratch.png", "CNN from scratch")

with tab_about:
    st.markdown(f"""
**Task.** 3-class classification of paediatric chest X-rays (Kermany et al., Kaggle *chest-xray-pneumonia*),
relabelled from filenames into Normal / Bacterial / Viral.

**Pipeline.** Grayscale → CLAHE contrast enhancement → 224×224 → DenseNet121 (ImageNet), fine-tuned in two stages.

**Why it's trustworthy(ish):**
- *Patient-level split*: no patient appears in both training and test data.
- *Calibrated confidence*: temperature scaling so "90%" really means ~90%.
- *Abstains when unsure*: below the confidence threshold it refers to a human.
- *Grad-CAM + focus check*: shows where the model looked and warns if it's looking at the border.

**Limitations.** Single-hospital paediatric dataset; bacterial vs viral is hard even for radiologists;
no external validation. Model: `{C.MODELS_DIR.name}/densenet121_finetuned.keras`.
""")
