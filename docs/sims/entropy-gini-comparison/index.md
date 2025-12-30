---
title: Entropy vs Gini Impurity Comparison
description: Interactive visualization comparing entropy and Gini impurity measures for decision tree splitting criteria
---

# Entropy vs Gini Impurity Comparison

<iframe src="main.html" width="100%" height="780px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[Open Fullscreen](main.html){: target="_blank" .md-button }

## About This MicroSim

This interactive visualization demonstrates the two most common splitting criteria used in decision tree algorithms: **entropy** (information gain) and **Gini impurity**. Both measures quantify the "impurity" or "disorder" of a node based on its class distribution.

### How to Use

1. **Adjust Class Distribution**: Move the slider to change the proportion of Class 1 (0% to 100%)
2. **Observe Curves**: Watch how both entropy and Gini curves respond to the distribution
3. **Show Good Split**: Click to see an example split with high information gain
4. **Show Bad Split**: Click to see a split that doesn't improve purity
5. **Show Split Comparison**: Check the box to overlay parent and children impurities

### Key Concepts

- **Entropy**: Measures disorder using information theory, calculated as H = -Σ pᵢ log₂(pᵢ)
- **Gini Impurity**: Measures the probability of misclassification, calculated as 1 - Σ pᵢ²
- **Maximum Impurity**: Both measures peak at 50/50 class distribution (maximum uncertainty)
- **Pure Nodes**: Both measures equal 0 when all samples belong to one class (perfect certainty)
- **Information Gain**: Reduction in impurity achieved by a split (higher is better)

### Educational Value

This visualization helps students understand:

- How impurity measures quantify node purity/disorder
- Why both metrics favor balanced vs pure splits
- The relationship between class distribution and impurity
- How decision trees select optimal splits (maximize information gain)
- Why entropy and Gini often produce similar trees despite different formulas

## Learning Objectives

**Bloom's Taxonomy Level**: Analyze (L4)

After using this MicroSim, students should be able to:

1. Calculate entropy and Gini impurity for a given class distribution
2. Explain why maximum impurity occurs at 50/50 distribution
3. Analyze the difference between good and bad splits using impurity reduction
4. Understand why decision trees prefer splits with high information gain
5. Compare when to use entropy vs Gini in practice

## Technical Details

- **Library**: p5.js
- **Responsive**: Fixed canvas size (900x700)
- **Interactivity**: Slider, buttons, checkbox controls
- **Features**: Side-by-side comparison, split demonstrations, real-time calculations

## Integration

To embed this MicroSim in your course materials:

```html
<iframe src="https://your-site.github.io/docs/sims/entropy-gini-comparison/main.html"
        width="100%" height="780px" scrolling="no"></iframe>
```
