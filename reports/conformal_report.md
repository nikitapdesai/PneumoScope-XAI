# Conformal prediction: EfficientNetV2-B0 224px (fine-tuned)

Calibrated on the validation split (n = 837), evaluated on the test split (n = 837). Target coverage 1 − α = **90%**.

## Methods at the target level

| Method | q̂ | Coverage | Avg set size | Singletons | Accuracy on singletons | Empty sets | Viral coverage | Coverage over 500 random re-splits |
|---|---|---|---|---|---|---|---|---|
| LAC | 0.697 | 0.894 | 1.24 | 76.2% | 87.1% | 0.0% | 0.779 | 0.901 ± 0.014 |
| APS | 1.000 | 0.908 | 1.52 | 48.4% | 84.2% | 0.0% | 0.789 | 0.922 ± 0.011 |
| APS (randomized) | 0.894 | 0.910 | 1.40 | 61.9% | 87.8% | 0.0% | 0.812 | 0.922 ± 0.010 |
| Class-conditional LAC | 0.500, 0.672, 0.791 | 0.901 | 1.33 | 67.4% | 86.5% | 0.0% | 0.864 | 0.901 ± 0.014 |

## What the sets look like (LAC)

| Set returned | Share of test X-rays |
|---|---|
| {Bacterial} | 35.5% |
| {Normal} | 25.2% |
| {Bacterial, Viral} | 21.4% |
| {Viral} | 15.5% |
| {Normal, Viral} | 2.3% |
| {Normal, Bacterial} | 0.1% |

![composition](figures/conformal_set_composition.png)

## Coverage vs set size across α

![sweep](figures/conformal_alpha_sweep.png)

| α | Method | Coverage | Avg size | Singletons | Viral coverage |
|---|---|---|---|---|---|
| 0.01 | LAC | 0.989 | 1.85 | 22.6% | 0.995 |
| 0.01 | APS | 1.000 | 3.00 | 0.0% | 1.000 |
| 0.01 | APS (randomized) | 0.986 | 1.91 | 20.9% | 0.972 |
| 0.01 | Class-conditional LAC | 0.990 | 1.94 | 24.6% | 0.986 |
| 0.02 | LAC | 0.986 | 1.78 | 27.6% | 0.986 |
| 0.02 | APS | 1.000 | 3.00 | 0.0% | 1.000 |
| 0.02 | APS (randomized) | 0.973 | 1.82 | 26.6% | 0.962 |
| 0.02 | Class-conditional LAC | 0.983 | 1.80 | 28.6% | 0.977 |
| 0.05 | LAC | 0.956 | 1.55 | 48.0% | 0.925 |
| 0.05 | APS | 0.949 | 1.73 | 27.4% | 0.869 |
| 0.05 | APS (randomized) | 0.941 | 1.63 | 41.3% | 0.897 |
| 0.05 | Class-conditional LAC | 0.947 | 1.54 | 47.6% | 0.930 |
| 0.10 | LAC | 0.894 | 1.24 | 76.2% | 0.779 |
| 0.10 | APS | 0.908 | 1.52 | 48.4% | 0.789 |
| 0.10 | APS (randomized) | 0.910 | 1.40 | 61.9% | 0.812 |
| 0.10 | Class-conditional LAC | 0.901 | 1.33 | 67.4% | 0.864 |
| 0.15 | LAC | 0.847 | 1.08 | 90.8% | 0.690 |
| 0.15 | APS | 0.892 | 1.39 | 60.8% | 0.751 |
| 0.15 | APS (randomized) | 0.894 | 1.27 | 73.6% | 0.779 |
| 0.15 | Class-conditional LAC | 0.848 | 1.17 | 80.6% | 0.789 |
| 0.20 | LAC | 0.798 | 0.95 | 95.5% | 0.606 |
| 0.20 | APS | 0.875 | 1.28 | 72.4% | 0.714 |
| 0.20 | APS (randomized) | 0.875 | 1.19 | 81.5% | 0.732 |
| 0.20 | Class-conditional LAC | 0.805 | 1.06 | 86.1% | 0.761 |

## Every model at 90% target coverage (LAC)

Same guarantee for every model; a better model earns it with smaller sets.

| Model | Coverage | Avg set size | Singletons | Accuracy on singletons | Viral coverage |
|---|---|---|---|---|---|
| Logistic Regression (HOG) | 0.924 | 1.49 | 52.2% | 88.6% | 0.897 |
| Random Forest (HOG) | 0.902 | 1.37 | 64.8% | 87.5% | 0.700 |
| CNN (scratch) | 0.901 | 1.31 | 69.5% | 88.0% | 0.793 |
| DenseNet121 (frozen) | 0.913 | 1.38 | 64.4% | 89.4% | 0.803 |
| DenseNet121 (fine-tuned) | 0.906 | 1.23 | 76.7% | 89.4% | 0.798 |
| EfficientNetV2-B0 224px (frozen) | 0.892 | 1.36 | 64.9% | 87.1% | 0.817 |
| EfficientNetV2-B0 224px (fine-tuned) | 0.894 | 1.24 | 76.2% | 87.1% | 0.779 |
| EfficientNetV2-B0 384px (frozen) | 0.906 | 1.34 | 67.3% | 87.9% | 0.822 |
| EfficientNetV2-B0 384px (fine-tuned) | 0.892 | 1.21 | 79.7% | 87.6% | 0.742 |
| EfficientNetV2-B0 512px (frozen) | 0.896 | 1.33 | 67.9% | 87.1% | 0.793 |
| EfficientNetV2-B0 512px (fine-tuned) | 0.912 | 1.27 | 73.7% | 89.5% | 0.812 |
