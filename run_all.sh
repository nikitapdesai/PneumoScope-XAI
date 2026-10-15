#!/usr/bin/env bash
# Full pipeline, end to end. Pass --quick for a fast smoke test of the DL models.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
QUICK=${1:-}

[ -f data/processed/manifest.csv ] || $PY -m src.data.make_manifest --download
$PY -m src.data.eda
$PY -m src.models.baseline_ml
$PY -m src.models.train_dl --model cnn $QUICK
$PY -m src.models.train_dl --model densenet $QUICK
for RES in 224 384 512; do                      # resolution study (A)
  $PY -m src.models.train_dl --model effnetv2 --img-size $RES $QUICK
done
$PY -m src.evaluation.compare                   # picks + records the deployed model
$PY -m src.evaluation.resolution_study
$PY -m src.evaluation.conformal                 # conformal prediction (C) for the deployed model
$PY -m src.evaluation.gradcam_report
echo "Done. See reports/*.md, then: streamlit run app/streamlit_app.py"
