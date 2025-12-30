---
title: Sigmoid Function Explorer
description: Interactive visualization showing how sigmoid function transforms linear outputs into probabilities for logistic regression
---

# Sigmoid Function Explorer

<iframe src="main.html" width="100%" height="680px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[Open Fullscreen](main.html){: target="_blank" .md-button }

## About This MicroSim

This interactive visualization demonstrates how the **sigmoid function** (also called logistic function) transforms a linear function z = mx + b into probabilities between 0 and 1. This transformation is the core of logistic regression for binary classification.

### How to Use

1. **Adjust Slope (m)**: Move the slider to change the slope of the linear function
2. **Adjust Intercept (b)**: Move the slider to shift the linear function up or down
3. **Observe Transformation**: Watch how the linear function (left, blue) transforms through sigmoid (right, orange)
4. **View Sample Points**: See how individual points map from linear space to probability space
5. **Reset**: Click "Reset Parameters" to return to default values (m=1, b=0)

### Key Concepts

- **Linear Function**: z = mx + b produces values from -∞ to +∞
- **Sigmoid Function**: σ(z) = 1 / (1 + e⁻ᶻ) maps z to probabilities [0, 1]
- **Decision Boundary**: Points where σ(z) = 0.5 (shown as horizontal line)
- **Slope Effect**: Larger |m| creates steeper sigmoid → more confident predictions
- **Intercept Effect**: Changing b shifts the decision boundary left or right

### Educational Value

This visualization helps students understand:

- How logistic regression transforms linear outputs into probabilities
- Why the sigmoid has an S-shaped curve
- How slope controls prediction confidence (steep = confident, flat = uncertain)
- How intercept shifts the decision threshold
- The relationship between linear decision boundaries and probabilistic predictions

## Learning Objectives

**Bloom's Taxonomy Level**: Understand (L2)

After using this MicroSim, students should be able to:

1. Explain how sigmoid function maps linear outputs to probabilities
2. Describe the effect of slope on prediction confidence
3. Understand how intercept affects the decision boundary
4. Interpret sigmoid outputs as class probabilities
5. Recognize the S-shape characteristic of the sigmoid curve

## Technical Details

- **Library**: p5.js
- **Responsive**: Fixed canvas size (800x600)
- **Interactivity**: Slider controls for slope and intercept
- **Features**: Side-by-side comparison, sample points, real-time updates

## Integration

To embed this MicroSim in your course materials:

```html
<iframe src="https://your-site.github.io/docs/sims/sigmoid-explorer/main.html"
        width="100%" height="680px" scrolling="no"></iframe>
```
