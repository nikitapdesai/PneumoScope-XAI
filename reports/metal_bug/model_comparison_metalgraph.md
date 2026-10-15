# Model comparison

Test set: 837 X-rays, patient-disjoint from train/val. 95% bootstrap confidence intervals in brackets. p = McNemar test vs the best model.

**Best model = EfficientNetV2-B0 384px (fine-tuned)**, chosen by validation macro-F1 (0.809) among fine-tuned models, so the test set played no part in the choice.

| Model | Val macro-F1 | Accuracy | Macro-F1 | ROC-AUC (OvR) | ECE ↓ | p vs best | Notes |
|---|---|---|---|---|---|---|---|
| Logistic Regression (HOG) | 0.748 | 0.771 [0.744–0.798] | 0.762 [0.734–0.790] | 0.900 [0.884–0.915] | 0.062 | 0.00024 | classical baseline |
| Random Forest (HOG) | 0.754 | 0.780 [0.751–0.808] | 0.747 [0.717–0.779] | 0.905 [0.890–0.921] | 0.113 | 0.0024 | classical baseline |
| CNN (scratch) | 0.778 | 0.800 [0.775–0.826] | 0.787 [0.761–0.813] | 0.904 [0.889–0.920] | 0.041 | 0.079 | learned features |
| DenseNet121 (frozen) | 0.775 | 0.800 [0.774–0.827] | 0.779 [0.752–0.807] | 0.908 [0.894–0.924] | 0.030 | 0.098 | transfer, stage 1 |
| DenseNet121 (fine-tuned) | 0.779 | 0.829 [0.805–0.853] | 0.811 [0.785–0.838] | 0.924 [0.912–0.938] | 0.040 | 0.58 | transfer, stage 2 |
| EfficientNetV2-B0 224px (frozen) | 0.752 | 0.772 [0.743–0.798] | 0.759 [0.729–0.787] | 0.902 [0.886–0.917] | 0.032 | 0.0001 | resolution study, stage 1 |
| EfficientNetV2-B0 224px (fine-tuned) | 0.801 | 0.808 [0.781–0.835] | 0.800 [0.774–0.827] | 0.925 [0.911–0.938] | 0.045 | 0.13 | resolution study, stage 2 |
| EfficientNetV2-B0 384px (frozen) | 0.785 | 0.790 [0.763–0.816] | 0.777 [0.749–0.802] | 0.912 [0.897–0.927] | 0.023 | 0.0093 | resolution study, stage 1 |
| **EfficientNetV2-B0 384px (fine-tuned)** | 0.809 | 0.822 [0.797–0.847] | 0.815 [0.788–0.840] | 0.922 [0.907–0.936] | 0.051 | — | resolution study, stage 2 |
| EfficientNetV2-B0 512px (frozen) | 0.787 | 0.802 [0.777–0.829] | 0.796 [0.769–0.823] | 0.919 [0.905–0.933] | 0.050 | 0.082 | resolution study, stage 1 |
| EfficientNetV2-B0 512px (fine-tuned) | 0.809 | 0.812 [0.787–0.840] | 0.803 [0.776–0.831] | 0.928 [0.915–0.941] | 0.022 | 0.46 | resolution study, stage 2 |

## Per-class F1

| Model | Normal | Bacterial | Viral |
|---|---|---|---|
| Logistic Regression (HOG) | 0.892 | 0.797 | 0.597 |
| Random Forest (HOG) | 0.895 | 0.814 | 0.532 |
| CNN (scratch) | 0.911 | 0.822 | 0.630 |
| DenseNet121 (frozen) | 0.895 | 0.836 | 0.608 |
| DenseNet121 (fine-tuned) | 0.928 | 0.854 | 0.653 |
| EfficientNetV2-B0 224px (frozen) | 0.856 | 0.808 | 0.614 |
| EfficientNetV2-B0 224px (fine-tuned) | 0.927 | 0.825 | 0.650 |
| EfficientNetV2-B0 384px (frozen) | 0.869 | 0.834 | 0.628 |
| EfficientNetV2-B0 384px (fine-tuned) | 0.933 | 0.841 | 0.671 |
| EfficientNetV2-B0 512px (frozen) | 0.911 | 0.823 | 0.655 |
| EfficientNetV2-B0 512px (fine-tuned) | 0.924 | 0.838 | 0.648 |

## Where the difficulty is: binary vs subtype

| Model | Normal vs Pneumonia acc | Normal vs Pneumonia AUC | Bacterial vs Viral acc | Bacterial vs Viral macro-F1 |
|---|---|---|---|---|
| Logistic Regression (HOG) | 0.938 | 0.983 | 0.748 | 0.728 |
| Random Forest (HOG) | 0.934 | 0.983 | 0.763 | 0.698 |
| CNN (scratch) | 0.951 | 0.985 | 0.782 | 0.752 |
| DenseNet121 (frozen) | 0.937 | 0.981 | 0.792 | 0.753 |
| DenseNet121 (fine-tuned) | 0.959 | 0.993 | 0.809 | 0.781 |
| EfficientNetV2-B0 224px (frozen) | 0.919 | 0.978 | 0.776 | 0.746 |
| EfficientNetV2-B0 224px (fine-tuned) | 0.967 | 0.995 | 0.777 | 0.760 |
| EfficientNetV2-B0 384px (frozen) | 0.928 | 0.981 | 0.797 | 0.772 |
| EfficientNetV2-B0 384px (fine-tuned) | 0.965 | 0.994 | 0.795 | 0.778 |
| EfficientNetV2-B0 512px (frozen) | 0.951 | 0.987 | 0.781 | 0.763 |
| EfficientNetV2-B0 512px (fine-tuned) | 0.958 | 0.994 | 0.794 | 0.770 |

## Calibration (EfficientNetV2-B0 384px (fine-tuned))

Temperature fitted on validation: **T = 1.04**. Accuracy is unchanged; only confidence changes.

| | ECE ↓ |
|---|---|
| Before | 0.051 |
| After temperature scaling | 0.050 |

![reliability](figures/reliability_best.png)

## Selective prediction ("refer to radiologist")

Confidence threshold chosen on validation to reach 90% accuracy on answered cases: **0.73**.

On test the model answers **69.7%** of X-rays with **88.5%** accuracy (vs 82.2% if forced to answer everything); the other 30.3% are flagged for human review.

![risk-coverage](figures/risk_coverage.png)
