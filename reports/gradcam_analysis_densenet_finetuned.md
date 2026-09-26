# Grad-CAM analysis: densenet_finetuned

![grid](figures/gradcam_grid_densenet_finetuned.png)

## Shortcut-learning audit

*Border attention* = share of Grad-CAM mass within 12% of the image edge, where hospital text markers, arrows and tubes sit. That strip covers ~42% of the image, so a uniform heatmap would score ~0.42. Lower means the model focuses on the central chest.

Audited 837 test images. Mean border attention: **0.324**

| Group | Mean border attention |
|---|---|
| true NORMAL | 0.354 |
| true PNEUMONIA_BACTERIA | 0.293 |
| true PNEUMONIA_VIRUS | 0.349 |
| misclassified | 0.331 |
| correct | 0.322 |

If misclassified images show clearly higher border attention, the model is partly using non-anatomical cues on those cases, which is worth mentioning under limitations.
