# ROC Curve Comparison

<iframe src="main.html" width="100%" height="950px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](main.html){: target="_blank" .md-button }

## Description

Compare ROC curves for classifiers with different performance levels.

## Learning Objectives

- Understand how ROC curves visualize classifier performance
- Interpret AUC (Area Under Curve) as a performance metric
- Compare multiple classifiers using ROC curves
- Recognize the trade-off between TPR and FPR

## How to Use

1. **Select Classifier**: Choose performance level from dropdown
2. **Show All**: Toggle to compare all classifiers simultaneously
3. **Observe**: See how AUC relates to curve position

## Key Concepts

### ROC Curve
- Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
- Shows performance at all classification thresholds
- Better classifiers curve toward top-left corner

### AUC (Area Under Curve)
- Single metric summarizing classifier performance
- Range: 0.5 (random) to 1.0 (perfect)
- Higher AUC = better overall performance

### Interpreting Performance
- **Excellent**: AUC > 0.9
- **Good**: AUC 0.8-0.9
- **Fair**: AUC 0.7-0.8
- **Poor**: AUC 0.5-0.7
- **Random**: AUC = 0.5 (diagonal line)
