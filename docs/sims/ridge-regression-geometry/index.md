---
title: Ridge Regression Geometry
description: Interactive visualization showing L2 regularization constraint circle and how Ridge regression shrinks coefficients toward origin
---

# Ridge Regression Geometry

<iframe src="main.html" height="582px" width="100%" scrolling="no"></iframe>

[Run the Ridge Regression Geometry MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## Description

This interactive visualization demonstrates the geometric interpretation of Ridge regression (L2 regularization). The simulation shows:

- **L2 Constraint Circle**: The circular constraint region β₁² + β₂² ≤ t that defines the feasible coefficient space
- **OLS Solution**: The unconstrained ordinary least squares solution (red point)
- **Ridge Solution**: The constrained solution where the error contour touches the L2 circle (blue point)
- **Error Contours**: Elliptical contours representing the loss function (can be toggled)
- **Shrinkage**: Visual arrow showing how Ridge pulls coefficients toward the origin

### Interactive Controls

- **Regularization λ Slider**: Adjust the regularization strength from 0 (no penalty) to 1 (maximum penalty). As λ increases, the constraint circle shrinks, pulling the Ridge solution closer to the origin.
- **Show Error Contours Checkbox**: Toggle the display of error contour ellipses

### Key Concepts

1. **Circular Constraint**: The L2 penalty creates a circular feasible region in coefficient space
2. **Smooth Shrinkage**: Ridge regression smoothly shrinks coefficients toward zero
3. **No Sparsity**: Coefficients shrink but rarely become exactly zero (no feature selection)
4. **Tangency Condition**: The optimal Ridge solution occurs where an error contour is tangent to the constraint circle

### Educational Use

This visualization helps students understand:

- Why Ridge regression shrinks coefficients proportionally
- The geometric relationship between the penalty parameter λ and the constraint radius
- How the L2 penalty affects the solution path compared to unconstrained OLS
- Why Ridge regression doesn't produce sparse solutions (coefficients don't reach zero)

### Technical Details

- Built with p5.js for interactive visualization
- Width-responsive design for embedding in educational materials
- Real-time parameter updates as sliders are adjusted

## Lesson Plan

**Learning Objective**: Students will understand the geometric interpretation of Ridge regression and how L2 regularization constrains the coefficient space.

**Prerequisites**: Basic understanding of linear regression, OLS estimation, and the regularization concept.

**Duration**: 10-15 minutes

**Activities**:
1. Start with λ = 0 and observe the OLS solution
2. Gradually increase λ and watch the Ridge solution move toward the origin
3. Discuss why the circular constraint creates proportional shrinkage
4. Compare the behavior to Lasso regression (L1 penalty) which uses a diamond-shaped constraint
