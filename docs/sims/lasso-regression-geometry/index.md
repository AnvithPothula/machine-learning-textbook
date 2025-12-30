---
title: Lasso Regression Geometry
description: Interactive visualization showing L1 regularization constraint diamond and how Lasso regression produces sparse solutions through feature selection
---

# Lasso Regression Geometry

<iframe src="main.html" height="582px" width="100%" scrolling="no"></iframe>

[Run the Lasso Regression Geometry MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## Description

This interactive visualization demonstrates the geometric interpretation of Lasso regression (L1 regularization). The simulation shows:

- **L1 Constraint Diamond**: The diamond-shaped constraint region |β₁| + |β₂| ≤ t that defines the feasible coefficient space
- **OLS Solution**: The unconstrained ordinary least squares solution (red point)
- **Lasso Solution**: The constrained solution where the error contour touches the L1 diamond (green point)
- **Error Contours**: Elliptical contours representing the loss function (can be toggled)
- **Feature Selection**: Visual highlighting when the solution hits a corner, setting one coefficient to exactly zero

### Interactive Controls

- **Regularization λ Slider**: Adjust the regularization strength from 0 (no penalty) to 1 (maximum penalty). As λ increases beyond 0.3, the solution tends to hit the diamond's corner, demonstrating automatic feature selection.
- **Show Error Contours Checkbox**: Toggle the display of error contour ellipses

### Key Concepts

1. **Diamond-Shaped Constraint**: The L1 penalty creates a diamond (rotated square) feasible region in coefficient space
2. **Sparsity-Inducing**: Lasso regression often drives some coefficients to exactly zero, performing automatic feature selection
3. **Corner Solutions**: The sharp corners of the diamond make it highly likely that the solution will have zero coefficients
4. **Feature Selection**: When β₂ = 0, that feature is effectively removed from the model

### Educational Use

This visualization helps students understand:

- Why Lasso regression produces sparse solutions (coefficients = 0)
- The geometric relationship between the L1 penalty and feature selection
- How the diamond shape's sharp corners differ from Ridge's smooth circle
- Why Lasso is preferred when you want to identify the most important features
- The trade-off between model complexity and fit quality

### Technical Details

- Built with p5.js for interactive visualization
- Width-responsive design for embedding in educational materials
- Real-time parameter updates as sliders are adjusted
- Visual feedback when feature selection occurs (orange highlighting)

## Lesson Plan

**Learning Objective**: Students will understand the geometric interpretation of Lasso regression and how L1 regularization induces sparsity through automatic feature selection.

**Prerequisites**: Basic understanding of linear regression, OLS estimation, regularization, and ideally Ridge regression for comparison.

**Duration**: 10-15 minutes

**Activities**:
1. Start with λ = 0 and observe the OLS solution outside the diamond
2. Gradually increase λ and watch the Lasso solution move toward the diamond
3. Observe what happens around λ = 0.3-0.4 when the solution hits the corner (β₂ = 0)
4. Discuss why the diamond's sharp corners create exact zeros
5. Compare this behavior to Ridge regression which has smooth shrinkage
6. Discuss when feature selection (Lasso) is preferable to proportional shrinkage (Ridge)
