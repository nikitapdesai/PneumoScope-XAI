# Grad-CAM analysis: effnetv2_b0_224_finetuned

![grid](figures/gradcam_grid_effnetv2_b0_224_finetuned.png)

## Shortcut-learning audit

*Border attention* = share of Grad-CAM mass within 12% of the image edge, where hospital text markers, arrows and tubes sit. That strip covers ~42% of the image, so a uniform heatmap would score ~0.42. Lower means the model focuses on the central chest.

Audited 837 test images. Mean border attention: **0.322**

| Group | Mean border attention |
|---|---|
| true NORMAL | 0.423 |
| true PNEUMONIA_BACTERIA | 0.247 |
| true PNEUMONIA_VIRUS | 0.357 |
| misclassified | 0.318 |
| correct | 0.323 |

If misclassified images show clearly higher border attention, the model is partly using non-anatomical cues on those cases, which is worth mentioning under limitations.
