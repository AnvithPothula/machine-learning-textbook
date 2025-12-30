---
title: SVM Margin Maximization
description: Interactive visualization showing how Support Vector Machines maximize the margin between classes and identify support vectors
---

# SVM Margin Maximization

<iframe src="main.html" height="582px" width="100%" scrolling="no"></iframe>

[Run the SVM Margin Maximization MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## Description

This interactive visualization demonstrates the core principle of Support Vector Machines (SVMs): margin maximization. The simulation shows:

- **Decision Boundary**: The hyperplane (shown as a vertical line) that separates the two classes
- **Margin Boundaries**: Dashed lines showing the edges of the margin (the "widest street")
- **Class -1 Points**: Red circles on the left side
- **Class +1 Points**: Blue squares on the right side
- **Support Vectors**: Points with thick black borders that lie exactly on the margin boundaries
- **Margin Region**: Shaded blue area showing the width of the margin

### Interactive Controls

- **Margin Width Slider**: Adjust the width of the margin to see how it affects the separation. This demonstrates the concept of maximizing the margin (larger margin = better generalization).
- **Show Margin Boundaries Checkbox**: Toggle the display of margin boundary lines

### Key Concepts

1. **Maximum Margin Principle**: SVMs find the decision boundary that maximizes the distance to the nearest training examples from both classes
2. **Support Vectors**: The critical training points that lie on the margin boundaries (shown with thick borders). These points define the optimal hyperplane.
3. **Margin Width**: The perpendicular distance between the two margin boundaries, equal to 2/||w|| where w is the weight vector
4. **Sparsity**: Only support vectors matter for the decision boundary; all other points could be removed without changing the solution

### Educational Use

This visualization helps students understand:

- Why SVMs are called "maximum margin" classifiers
- The geometric meaning of the margin and its relationship to generalization
- How support vectors differ from regular training points
- Why SVM solutions are sparse (only support vectors matter)
- The concept of the "widest street" separating two classes

### Technical Details

- Built with p5.js for interactive visualization
- Width-responsive design for embedding in educational materials
- Real-time parameter updates as the margin slider is adjusted
- Support vectors automatically repositioned based on margin width

## Lesson Plan

**Learning Objective**: Students will understand the margin maximization principle of SVMs and identify the role of support vectors in defining the decision boundary.

**Prerequisites**: Basic understanding of classification, decision boundaries, and linear separability.

**Duration**: 10-15 minutes

**Activities**:
1. Observe the default configuration with support vectors highlighted
2. Adjust the margin width slider to see how margin size changes
3. Notice that only the support vectors (thick borders) touch the margin boundaries
4. Discuss why maximizing the margin leads to better generalization
5. Consider what happens if you remove non-support-vector points (nothing changes!)
6. Compare to other classifiers like logistic regression or k-NN which use all training points
7. Relate margin width to the regularization parameter C (smaller margin = larger C = less regularization)
