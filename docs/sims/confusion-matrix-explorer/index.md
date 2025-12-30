# Interactive Confusion Matrix Explorer

Explore confusion matrices and calculate classification metrics interactively.

## Learning Objectives

- Understand the structure of confusion matrices (TP, TN, FP, FN)
- Calculate accuracy, precision, recall, and F1 score from confusion matrix values
- Recognize trade-offs between different metrics
- Interpret confusion matrices for different classifier performance scenarios

## How to Use

1. **Adjust Values**: Enter custom values for TP, TN, FP, FN
2. **Select Examples**: Choose preset scenarios (good/poor/imbalanced)
3. **Observe Metrics**: See how metrics update in real-time

## Key Metrics

- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **FPR**: False positive rate

<iframe src="confusion-matrix.html" width="100%" height="900" frameborder="0"></iframe>
