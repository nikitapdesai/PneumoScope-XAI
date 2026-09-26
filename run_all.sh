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
$PY -m src.evaluation.compare
$PY -m src.evaluation.gradcam_report
echo "Done. See reports/model_comparison.md, then: streamlit run app/streamlit_app.py"
