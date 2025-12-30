---
title: K-Fold Cross-Validation Visualization
description: Interactive visualization demonstrating how K-fold cross-validation partitions data and cycles through different train/validation splits
---

# K-Fold Cross-Validation Visualization

<iframe src="main.html" width="100%" height="550px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[Open Fullscreen](main.html){: target="_blank" .md-button }

## About This MicroSim

This interactive visualization demonstrates how K-fold cross-validation works by showing how a dataset is partitioned into K equal-sized folds, with each fold taking turns as the validation set while the remaining folds form the training set.

### How to Use

1. **Adjust K**: Use the slider to change the number of folds (3-10)
2. **Next Fold**: Step through each fold one at a time
3. **Run All Folds**: Automatically animate through all K folds
4. **Reset**: Return to the initial state

### Key Concepts

- **Training Folds (Blue)**: Data used to train the model in each iteration
- **Validation Fold (Orange)**: Data used to evaluate the model in each iteration
- **Cross-Validation Score**: The mean accuracy across all K folds provides a more reliable performance estimate than a single train/validation split

### Educational Value

This visualization helps students understand:

- How cross-validation ensures every data point is used for both training and validation
- Why averaging across multiple folds produces more reliable performance estimates
- The trade-off between K value and computational cost (higher K = more iterations)
- How cross-validation reduces the impact of lucky or unlucky single data splits

## Learning Objectives

**Bloom's Taxonomy Level**: Understand (L2)

After using this MicroSim, students should be able to:

1. Explain how K-fold cross-validation partitions a dataset
2. Describe why cross-validation provides better performance estimates than a single split
3. Understand the meaning of "K-fold" and how K affects the validation process
4. Calculate the mean cross-validation score from individual fold results

## Technical Details

- **Library**: p5.js
- **Responsive**: Yes (adapts to container width)
- **Interactivity**: Slider control, buttons for stepping/animation
- **Data**: Simulated accuracy values for demonstration

## Integration

To embed this MicroSim in your course materials:

```html
<iframe src="https://your-site.github.io/docs/sims/kfold-cross-validation/main.html"
        width="100%" height="550px" scrolling="no"></iframe>
```
