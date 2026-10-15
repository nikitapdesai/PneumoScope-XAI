# Resolution study: EfficientNetV2-B0

Same split, seed, augmentation, two-stage recipe and batch size (16) at every resolution; only the input size changes. Test set n = 837.

## Accuracy

| Resolution | Val macro-F1 | Test accuracy | Test macro-F1 | ROC-AUC | ECE | McNemar p vs 224px |
|---|---|---|---|---|---|---|
| 224px | 0.801 | 0.808 [0.781–0.835] | 0.800 [0.774–0.827] | 0.925 | 0.045 | — |
| 384px | 0.809 | 0.822 [0.797–0.847] | 0.815 [0.788–0.840] | 0.922 | 0.051 | 0.13 |
| 512px | 0.809 | 0.812 [0.787–0.840] | 0.803 [0.776–0.831] | 0.928 | 0.022 | 0.76 |

## Per class (test)

| Resolution | Normal P / R / F1 | Bacterial P / R / F1 | Viral P / R / F1 | Normal vs Pneumonia acc | Bacterial vs Viral acc |
|---|---|---|---|---|---|
| 224px | 0.90 / 0.95 / **0.927** | 0.85 / 0.80 / **0.825** | 0.64 / 0.66 / **0.650** | 0.967 | 0.777 |
| 384px | 0.91 / 0.95 / **0.933** | 0.86 / 0.82 / **0.841** | 0.66 / 0.69 / **0.671** | 0.965 | 0.795 |
| 512px | 0.96 / 0.89 / **0.924** | 0.82 / 0.86 / **0.838** | 0.65 / 0.64 / **0.648** | 0.958 | 0.794 |

![accuracy](figures/resolution_accuracy.png)

## Computational cost (measured on the training machine)

| Resolution | Params | GFLOPs / image | Inference ms / image (batch 1) | Inference ms / image (batch 32) | Stage-1 s / epoch | Stage-2 s / epoch | Epochs (S1 + S2) | Total train time | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 224px | 6,248,019 | 1.46 | 93.5 | 5.89 | 14 | 60 | 8 + 14 | 16 min | 3.9 GB |
| 384px | 6,248,019 | 4.27 | 97.3 | 10.52 | 65 | 132 | 8 + 14 | 40 min | 5.3 GB |
| 512px | 6,248,019 | 7.59 | 134.9 | 30.15 | 96 | 252 | 15 + 11 | 72 min | 6.7 GB |

![cost](figures/resolution_cost.png)

![tradeoff](figures/resolution_tradeoff.png)
