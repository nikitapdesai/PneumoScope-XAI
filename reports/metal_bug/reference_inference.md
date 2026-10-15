# Reference vs Metal-graph inference

Test-set metrics of the SAME saved weights, evaluated two ways. *Metal graph* = tensorflow-metal's optimized graph (what `model.predict` did during training). *Reference* = Grappler meta-optimizer disabled, which matches CPU (column 'CPU check'). All other reports use the reference numbers.

| Model | Metal-graph acc | Reference acc | Metal-graph macro-F1 | Reference macro-F1 | Test argmax agreement | Max prob diff | CPU check (max diff) |
|---|---|---|---|---|---|---|---|
| CNN (scratch) | 0.800 | 0.804 | 0.787 | 0.788 | 96.3% | 0.315 | 2.4e-07 |
| DenseNet121 (frozen) | 0.800 | 0.804 | 0.779 | 0.785 | 95.8% | 0.355 | 8.3e-07 |
| DenseNet121 (fine-tuned) | 0.829 | 0.810 | 0.811 | 0.774 | 93.5% | 0.395 | 3.6e-07 |
| EfficientNetV2-B0 224px (frozen) | 0.772 | 0.741 | 0.759 | 0.703 | 85.7% | 0.503 | 7.2e-07 |
| EfficientNetV2-B0 224px (fine-tuned) | 0.808 | 0.773 | 0.800 | 0.781 | 88.6% | 0.468 | 6.6e-07 |
| EfficientNetV2-B0 384px (frozen) | 0.790 | 0.762 | 0.777 | 0.725 | 85.1% | 0.521 | 6.6e-07 |
| EfficientNetV2-B0 384px (fine-tuned) | 0.822 | 0.822 | 0.815 | 0.818 | 96.3% | 0.351 | 4.8e-07 |
| EfficientNetV2-B0 512px (frozen) | 0.802 | 0.795 | 0.796 | 0.792 | 94.6% | 0.488 | 8.9e-07 |
| EfficientNetV2-B0 512px (fine-tuned) | 0.812 | 0.804 | 0.803 | 0.796 | 96.4% | 0.482 | 6.3e-07 |
