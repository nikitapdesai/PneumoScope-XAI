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

        levels = predictor.conformal_levels()
        coverage = None
        if levels:
            with left:
                coverage = st.select_slider(
                    "Guaranteed coverage of the prediction set", options=levels,
                    value=round(1 - C.CONFORMAL_ALPHA, 2), format_func=lambda v: f"{v:.0%}",
                    help="Conformal prediction: across X-rays like the calibration set, the true diagnosis is inside "
                         "the returned set at least this often. Higher = safer but larger sets.")
        image = np.array(Image.open(source).convert("L"))
        with st.spinner("Analysing…"):
            pred = predictor.predict(image, coverage)

        with left:
            if pred.abstain:
                st.warning(f"**Uncertain ({pred.confidence:.0%}).** Model leans towards *{PRETTY[pred.label]}*, "
                           f"but its confidence is below the {predictor.threshold:.0%} safety threshold. "
                           "Refer to a radiologist.")
            else:
                st.success(f"**{PRETTY[pred.label]}** ({pred.confidence:.0%} calibrated confidence)")
            if pred.prediction_set:
                labels = " · ".join(PRETTY[c] for c in pred.prediction_set)
                if len(pred.prediction_set) == 1:
                    st.info(f"**Conformal set ({pred.coverage:.0%}):** {labels}  \nOnly one diagnosis is plausible.")
                else:
                    st.info(f"**Conformal set ({pred.coverage:.0%}):** {labels}  \n"
                            f"The model cannot rule these out at this guarantee level.")
            elif predictor.conformal:
                st.info(f"**Conformal set ({pred.coverage:.0%}):** empty: the image looks unlike any class the model "
                        "was calibrated on. Treat as out-of-distribution.")
            st.bar_chart(pd.DataFrame({"probability": pred.probs},
                                      index=[PRETTY[c] for c in C.CLASS_NAMES]), horizontal=True)
            if pred.border_attention > 0.5:
                st.caption(f"⚠️ Focus check: {pred.border_attention:.0%} of the model's attention is near the image "
                           "border (labels, markers), not the lungs. Treat this prediction with extra caution.")
            else:
                st.caption(f"Focus check: {1 - pred.border_attention:.0%} of attention is on the central chest.")

        with right:
            c1, c2 = st.columns(2)
            size = predictor.spec.img_size
            c1.image(pred.model_input, caption=f"Pre-processed input (CLAHE, {size}×{size})", width="stretch")
            c2.image(pred.heatmap_overlay, caption="Grad-CAM: where the model looked", width="stretch")
            st.caption(f"Model: {predictor.name}")

SHORT_NAMES = {"logreg": "1 · LogReg", "random_forest": "2 · Random Forest", "cnn_scratch": "3 · CNN",
               "densenet_frozen": "4 · DenseNet frozen", "densenet_finetuned": "5 · DenseNet tuned",
               **{f"effnetv2_{C.EFFNET_VARIANT.lower()}_{r}_finetuned": f"{6 + i} · EffNetV2 {r}px"
                  for i, r in enumerate(C.EFFNET_RESOLUTIONS)}}


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
        others = [m["p_vs_best"] for m in models if m["p_vs_best"] is not None]
        n_sig = sum(p < 0.05 for p in others)
        st.caption(f"⭐ = deployed model, picked by validation macro-F1 (the test set played no part). "
                   f"{n_sig} of {len(others)} other models are significantly worse (McNemar p < 0.05).")

        # ---------- where the difficulty is ----------
        st.divider()
        st.subheader("Where the difficulty actually is")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Two sub-tasks, very different difficulty**")
            shown = [m for m in models if m["key"] in SHORT_NAMES]   # one bar pair per model family
            split = pd.DataFrame({
                "Normal vs Pneumonia": [m["binary_accuracy"] * 100 for m in shown],
                "Bacterial vs Viral": [m["subtype_accuracy"] * 100 for m in shown],
            }, index=[SHORT_NAMES[m["key"]] for m in shown])
            st.bar_chart(split, horizontal=True, stack=False, x_label="", y_label="Accuracy (%)", height=380)
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
                           "Scaling fitted on validation did not improve test calibration."))
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
            st.caption("Red = regions that drove the prediction. Wrong predictions are shown too, to expose shortcuts.")
            g1, g2, g3 = st.columns([1, 2, 1])
            with g2:
                figure(f"gradcam_grid_{best_key}.png")
        with st.expander("Training curves"):
            for r in C.EFFNET_RESOLUTIONS:
                figure(f"history_effnetv2_{C.EFFNET_VARIANT.lower()}_{r}_finetuned.png",
                       f"EfficientNetV2-{C.EFFNET_VARIANT} {r}px: stage 1 (frozen) then stage 2 (fine-tune)")
            figure("history_densenet.png", "DenseNet121: stage 1 (frozen) then stage 2 (fine-tune)")
            figure("history_cnn_scratch.png", "CNN from scratch")

        # ---------- resolution study (A) ----------
        res_path = C.RESULTS_DIR / "resolution_study.json"
        if res_path.exists():
            st.divider()
            st.subheader(f"Resolution study: EfficientNetV2-{C.EFFNET_VARIANT}")
            R = json.loads(res_path.read_text())
            st.caption("Same recipe, seed, split and batch size at every resolution; only the input size changes.")
            st.dataframe(pd.DataFrame([{
                "Resolution": f"{r['res']}px",
                "Accuracy": r["accuracy"] * 100,
                "Macro-F1": r["macro_f1"],
                "Normal F1": r["f1"]["NORMAL"],
                "Bacterial F1": r["f1"]["PNEUMONIA_BACTERIA"],
                "Viral F1": r["f1"]["PNEUMONIA_VIRUS"],
                "GFLOPs / image": r.get("gflops_per_image"),
                "ms / image (b32)": r.get("latency_ms_per_image_batch32"),
                "Train time (min)": r.get("train_minutes_total"),
            } for r in R]), hide_index=True, width="stretch", column_config={
                "Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=50, max_value=100),
                "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                "Normal F1": st.column_config.NumberColumn(format="%.3f"),
                "Bacterial F1": st.column_config.NumberColumn(format="%.3f"),
                "Viral F1": st.column_config.NumberColumn(format="%.3f"),
                "GFLOPs / image": st.column_config.NumberColumn(format="%.2f"),
                "ms / image (b32)": st.column_config.NumberColumn(format="%.2f"),
                "Train time (min)": st.column_config.NumberColumn(format="%.0f"),
            })
            figure("resolution_accuracy.png")
            c1, c2 = st.columns([3, 1])
            with c1:
                figure("resolution_cost.png")
            with c2:
                figure("resolution_tradeoff.png")

        # ---------- conformal prediction (C) ----------
        conf_path = C.RESULTS_DIR / "conformal_summary.json"
        if conf_path.exists():
            st.divider()
            CF = json.loads(conf_path.read_text())
            m = CF["methods"][C.CONFORMAL_METHOD]
            st.subheader("Conformal prediction: answers with a guarantee")
            st.markdown(f"Instead of one label, the model returns every diagnosis it can't rule out. Calibrated on "
                        f"{CF['n_cal']} validation X-rays so that the true diagnosis is in the set at least "
                        f"**{1 - CF['alpha']:.0%}** of the time.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Test coverage", pct(m["coverage"]), f"target {1 - CF['alpha']:.0%}", delta_color="off",
                      delta_arrow="off", border=True)
            k2.metric("Average set size", f"{m['avg_set_size']:.2f}", "out of 3 classes", delta_color="off",
                      delta_arrow="off", border=True)
            k3.metric("Single-label answers", pct(m["singleton_rate"]),
                      f"{pct(m['singleton_accuracy'])} correct", delta_color="off", delta_arrow="off", border=True)
            k4.metric("Viral coverage", pct(m["per_class_coverage"]["PNEUMONIA_VIRUS"]), "hardest class",
                      delta_color="off", delta_arrow="off", border=True)
            c1, c2 = st.columns(2)
            with c1:
                figure("conformal_set_composition.png")
            with c2:
                figure("conformal_alpha_sweep.png")

with tab_about:
    st.markdown(f"""
**Task.** 3-class classification of paediatric chest X-rays (Kermany et al., Kaggle *chest-xray-pneumonia*),
relabelled from filenames into Normal / Bacterial / Viral.

**Pipeline.** Grayscale → CLAHE contrast enhancement → resize → CNN backbone (ImageNet-pretrained DenseNet121 or
EfficientNetV2, fine-tuned in two stages). The served model is the one with the best validation macro-F1.

**Why it's trustworthy(ish):**
- *Patient-level split*: no patient appears in both training and test data.
- *Calibrated confidence*: temperature scaling so "90%" really means ~90%.
- *Abstains when unsure*: below the confidence threshold it refers to a human.
- *Conformal prediction sets*: a set of diagnoses that contains the true one with a guaranteed probability.
- *Grad-CAM + focus check*: shows where the model looked and warns if it's looking at the border.

**Limitations.** Single-hospital paediatric dataset; bacterial vs viral is hard even for radiologists;
no external validation. The conformal guarantee assumes new X-rays resemble the calibration set.
""")
