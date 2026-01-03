---
title: Regularization Techniques
description: Preventing overfitting through L1 and L2 regularization, Ridge and Lasso regression
generated_by: claude skill chapter-content-generator
date: 2025-12-28
version: 0.03
---

# Regularization Techniques

## Summary

This chapter focuses on regularization methods that prevent overfitting by constraining model complexity. Students will learn how L1 (Lasso) and L2 (Ridge) regularization add penalty terms to the loss function to discourage large parameter values, understand the geometric interpretation of these constraints, and discover how L1 regularization can perform automatic feature selection by driving some weights to exactly zero. The chapter demonstrates practical applications of Ridge and Lasso regression and explains how to select appropriate regularization strength through cross-validation. These techniques are fundamental for building models that generalize well to unseen data.

## Concepts Covered

This chapter covers the following 5 concepts from the learning graph:

1. Regularization
2. L1 Regularization
3. L2 Regularization
4. Ridge Regression
5. Lasso Regression

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Machine Learning Fundamentals](../01-intro-to-ml-fundamentals/index.md)
- [Chapter 3: Decision Trees and Tree-Based Learning](../03-decision-trees/index.md)

---

## The Problem of Overfitting

Machine learning models face a fundamental challenge: they must learn patterns from training data while maintaining the ability to generalize to new, unseen examples. When models become too complex, they can memorize the training data—including its noise and peculiarities—rather than learning the underlying patterns. This phenomenon, called **overfitting**, results in excellent training performance but poor performance on test data.

Consider a linear regression problem where we predict automobile fuel efficiency (mpg) from various features. As we learned in previous chapters, linear regression finds coefficients that minimize the sum of squared errors:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \boldsymbol{\beta}^T \mathbf{x}_i)^2$$

When we have many features relative to the number of training examples, or when features are highly correlated, the model can fit the training data almost perfectly by assigning very large positive and negative coefficients. These extreme coefficients capture noise rather than signal, leading to poor generalization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load automobile dataset
Auto = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/Auto.csv')

# Create polynomial features to demonstrate overfitting
X = Auto[["weight"]].values
y = Auto["mpg"].values

# Add polynomial features up to degree 10
X_poly = np.column_stack([X**i for i in range(1, 11)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Fit without regularization
model = LinearRegression()
model.fit(X_train, y_train)

print("Training R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))
print("\nCoefficients (first 5):", model.coef_[:5])
print("Coefficient magnitudes range:", np.min(np.abs(model.coef_)), "to", np.max(np.abs(model.coef_)))
```

In this example, the high-degree polynomial features allow the model to fit training data extremely well, but the large coefficient magnitudes indicate overfitting. The model has learned to memorize training examples rather than discover generalizable patterns.

**Regularization** provides a principled solution to overfitting by adding a penalty term to the loss function that discourages large coefficient values. This forces the model to find simpler explanations that are more likely to generalize.

## L2 Regularization and Ridge Regression

**L2 regularization**, also called **Ridge regularization**, adds a penalty proportional to the sum of squared coefficients to the loss function. The modified objective becomes:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \boldsymbol{\beta}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

where:

- The first term is the standard sum of squared errors (residual sum of squares)
- The second term is the **L2 penalty**: $\lambda \|\boldsymbol{\beta}\|_2^2 = \lambda \sum_{j=1}^{p} \beta_j^2$
- $\lambda \geq 0$ is the **regularization parameter** controlling penalty strength
- $p$ is the number of features

### Understanding the L2 Penalty

The L2 penalty term $\sum_{j=1}^{p} \beta_j^2$ grows quadratically with coefficient magnitude. This creates several important effects:

1. **Shrinkage**: Coefficients are "shrunk" toward zero, but rarely become exactly zero
2. **Smooth solutions**: The quadratic penalty is differentiable everywhere, leading to stable optimization
3. **Correlated features**: When features are correlated, Ridge tends to assign similar weights to them rather than arbitrarily choosing one

The regularization parameter $\lambda$ controls the trade-off:

- $\lambda = 0$: No regularization, equivalent to ordinary least squares
- Small $\lambda$: Weak penalty, model can use large coefficients
- Large $\lambda$: Strong penalty, coefficients shrink toward zero, potentially underfitting

### Ridge Regression Implementation

**Ridge regression** applies L2 regularization to linear regression. Scikit-learn provides the `Ridge` class for this purpose:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Ridge regression with different alpha values
# Note: scikit-learn uses 'alpha' instead of 'lambda'
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)

    train_score = ridge.score(X_train_scaled, y_train)
    test_score = ridge.score(X_test_scaled, y_test)
    max_coef = np.max(np.abs(ridge.coef_))

    print(f"Alpha={alpha:6.2f}: Train R²={train_score:.3f}, Test R²={test_score:.3f}, Max|coef|={max_coef:.2e}")
```

!!! note "Feature Scaling for Regularization"
    Always standardize features before applying regularization! The penalty term $\sum \beta_j^2$ treats all coefficients equally, but features on different scales lead to coefficients of different magnitudes. Standardization ensures fair penalization across all features.

### Geometric Interpretation of Ridge Regression

We can visualize Ridge regression geometrically in coefficient space. For two coefficients $\beta_1$ and $\beta_2$, the constraint $\beta_1^2 + \beta_2^2 \leq t$ defines a circle (in higher dimensions, a hypersphere).

The Ridge solution is the point where the smallest error contour (ellipse from the squared error term) touches this circular constraint region. The smooth circular boundary means the solution typically lies in the interior, not at a boundary where coefficients are exactly zero.

#### Ridge Regression Geometry

<iframe src="../../sims/ridge-regression-geometry/main.html" width="100%" height="582" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](../../sims/ridge-regression-geometry/main.html){: target="_blank" .md-button } | [Documentation](../../sims/ridge-regression-geometry/index.md)

### Visualizing Coefficient Paths

A powerful way to understand Ridge regression is to plot how coefficients change as $\lambda$ increases:

```python
# Compute Ridge solutions for a range of alpha values
alphas_range = np.logspace(-2, 3, 100)
coefs = []

for alpha in alphas_range:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

# Plot coefficient paths
plt.figure(figsize=(12, 6))
plt.plot(alphas_range, coefs)
plt.xscale('log')
plt.xlabel('Alpha (λ)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Ridge Regression: Coefficient Paths', fontsize=14)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.show()
```

This **regularization path** plot shows that as $\lambda$ increases, all coefficients shrink smoothly toward zero. Unlike L1 regularization (which we'll see next), coefficients approach zero asymptotically but never reach exactly zero.

## L1 Regularization and Lasso Regression

**L1 regularization**, also called **Lasso regularization** (Least Absolute Shrinkage and Selection Operator), replaces the squared penalty with an absolute value penalty:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \boldsymbol{\beta}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

The L1 penalty term is $\lambda \|\boldsymbol{\beta}\|_1 = \lambda \sum_{j=1}^{p} |\beta_j|$, the sum of absolute coefficient values.

### The Power of L1: Automatic Feature Selection

The most remarkable property of L1 regularization is that it can drive coefficients to **exactly zero**, effectively removing features from the model. This provides automatic **feature selection**: the model itself decides which features are most important.

Why does L1 produce exact zeros while L2 doesn't? The difference lies in the geometry:

- **L2 penalty** ($\beta^2$): Smooth and differentiable everywhere, gradient approaches zero as $\beta$ approaches zero
- **L1 penalty** ($|\beta|$): Has a "corner" at zero with constant gradient, allowing coefficients to hit exactly zero

This makes Lasso particularly valuable when:

1. You have many features and suspect only a subset are truly predictive
2. You want an interpretable model with fewer features
3. You need to reduce model complexity for deployment or computational efficiency

### Lasso Regression Implementation

Scikit-learn provides the `Lasso` class for L1-regularized regression:

```python
from sklearn.linear_model import Lasso

# Fit Lasso regression with different alpha values
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    train_score = lasso.score(X_train_scaled, y_train)
    test_score = lasso.score(X_test_scaled, y_test)
    n_nonzero = np.sum(lasso.coef_ != 0)

    print(f"Alpha={alpha:6.2f}: Train R²={train_score:.3f}, Test R²={test_score:.3f}, Non-zero coefs={n_nonzero}")
```

Notice how the number of non-zero coefficients decreases as $\lambda$ increases. Lasso is performing feature selection automatically!

### Geometric Interpretation of Lasso Regression

For two coefficients, the L1 constraint $|\beta_1| + |\beta_2| \leq t$ defines a diamond (in higher dimensions, a hypercube rotated 45°). The diamond has corners along the coordinate axes.

When the error contour touches the constraint region, it's more likely to touch at a corner where one or more coefficients are exactly zero. This is why Lasso produces sparse solutions.

#### Lasso Regression Geometry

<iframe src="../../sims/lasso-regression-geometry/main.html" width="100%" height="582" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](../../sims/lasso-regression-geometry/main.html){: target="_blank" .md-button } | [Documentation](../../sims/lasso-regression-geometry/index.md)

### Lasso Coefficient Paths

Plotting Lasso coefficient paths reveals the feature selection behavior:

```python
# Compute Lasso solutions for a range of alpha values
alphas_range = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas_range:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs.append(lasso.coef_)

# Plot coefficient paths
plt.figure(figsize=(12, 6))
plt.plot(alphas_range, coefs)
plt.xscale('log')
plt.xlabel('Alpha (λ)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Lasso Regression: Coefficient Paths', fontsize=14)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.show()
```

The Lasso paths show coefficients hitting exactly zero at different values of $\lambda$. The order in which coefficients become zero indicates their relative importance: features whose coefficients remain non-zero at higher $\lambda$ values are more predictive.

## Comparing Ridge and Lasso

Both Ridge and Lasso address overfitting through regularization, but they have distinct characteristics:

| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|------------|
| **Penalty** | $\lambda \sum \beta_j^2$ | $\lambda \sum \|\beta_j\|$ |
| **Coefficient shrinkage** | Smooth, asymptotic to zero | Can reach exactly zero |
| **Feature selection** | No (all features retained) | Yes (automatic) |
| **Correlated features** | Assigns similar weights | Arbitrarily selects one |
| **Solution uniqueness** | Always unique | May have multiple solutions |
| **Computational cost** | Fast (closed form) | Slower (iterative optimization) |
| **Interpretability** | All features contribute | Sparse model, easier to interpret |

### When to Use Ridge vs Lasso

**Use Ridge when:**

- All features are potentially relevant
- Features are highly correlated (multicollinearity)
- You want stable, unique solutions
- Computational speed is critical

**Use Lasso when:**

- You suspect many features are irrelevant
- You need automatic feature selection
- Interpretability is important (fewer features)
- You want a sparse model for deployment

**Use both (Elastic Net) when:**

- You want a balance between Ridge and Lasso properties
- You have groups of correlated features and want to select groups
- You're unsure which regularization type is better

### Elastic Net: Combining L1 and L2

**Elastic Net** combines both L1 and L2 penalties:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \boldsymbol{\beta}^T \mathbf{x}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

Alternatively, it can be parameterized with a mixing parameter $\alpha \in [0, 1]$:

$$\text{Penalty} = \lambda \left[ \alpha \|\boldsymbol{\beta}\|_1 + (1-\alpha) \|\boldsymbol{\beta}\|_2^2 \right]$$

where $\alpha = 0$ gives Ridge, $\alpha = 1$ gives Lasso, and intermediate values blend the two.

```python
from sklearn.linear_model import ElasticNet

# Elastic Net with balanced L1 and L2
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train_scaled, y_train)

print("Training R²:", elastic.score(X_train_scaled, y_train))
print("Test R²:", elastic.score(X_test_scaled, y_test))
print("Non-zero coefficients:", np.sum(elastic.coef_ != 0))
```

## Selecting the Regularization Parameter

Choosing the optimal $\lambda$ is critical: too small allows overfitting, too large causes underfitting. **Cross-validation** provides a principled method for selecting $\lambda$.

### Cross-Validation for Lambda Selection

We evaluate model performance across a range of $\lambda$ values using k-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score

# Test range of alpha values for Ridge
alphas = np.logspace(-2, 3, 50)
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
    ridge_scores.append(scores.mean())

# Find optimal alpha
optimal_alpha_ridge = alphas[np.argmax(ridge_scores)]

# Plot cross-validation curve
plt.figure(figsize=(12, 6))
plt.plot(alphas, ridge_scores, 'b-', linewidth=2, label='Ridge')
plt.axvline(optimal_alpha_ridge, color='blue', linestyle='--', label=f'Optimal α={optimal_alpha_ridge:.2f}')
plt.xscale('log')
plt.xlabel('Alpha (λ)', fontsize=12)
plt.ylabel('Cross-Validation R²', fontsize=12)
plt.title('Ridge Regression: Cross-Validation Score vs Regularization Strength', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal alpha for Ridge: {optimal_alpha_ridge:.3f}")
```

The cross-validation curve typically shows:

1. **Left side (small λ)**: High variance, potential overfitting
2. **Middle (optimal λ)**: Best bias-variance trade-off
3. **Right side (large λ)**: High bias, underfitting

### Automated Hyperparameter Tuning

Scikit-learn provides `RidgeCV` and `LassoCV` for automatic cross-validated $\lambda$ selection:

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge with automatic alpha selection
ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print("Optimal Ridge alpha:", ridge_cv.alpha_)
print("Test R²:", ridge_cv.score(X_test_scaled, y_test))

# Lasso with automatic alpha selection
lasso_cv = LassoCV(alphas=np.logspace(-2, 2, 100), cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print("\nOptimal Lasso alpha:", lasso_cv.alpha_)
print("Test R²:", lasso_cv.score(X_test_scaled, y_test))
print("Non-zero coefficients:", np.sum(lasso_cv.coef_ != 0))
```

These cross-validated variants automatically search over the specified alpha values and select the one with the best cross-validation performance.

## Regularization in Classification

Regularization applies to classification algorithms as well. Logistic regression, SVMs, and neural networks all benefit from L1 and L2 penalties.

### Regularized Logistic Regression

Scikit-learn's `LogisticRegression` includes L2 regularization by default, controlled by the `C` parameter (note: `C = 1/λ`, so smaller `C` means stronger regularization):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different regularization strengths
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

for C in C_values:
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    train_acc = lr.score(X_train_scaled, y_train)
    test_acc = lr.score(X_test_scaled, y_test)

    print(f"C={C:6.2f} (λ={1/C:6.2f}): Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
```

For L1 regularization in logistic regression, specify `penalty='l1'` and use the `saga` or `liblinear` solver:

```python
# L1-regularized logistic regression
lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=10000, random_state=42)
lr_l1.fit(X_train_scaled, y_train)

print("Test Accuracy:", lr_l1.score(X_test_scaled, y_test))
print("Non-zero coefficients per class:")
for i, coef in enumerate(lr_l1.coef_):
    print(f"  Class {i}: {np.sum(coef != 0)} features")
```

## Practical Considerations

### Always Standardize Features

Regularization penalizes coefficient magnitudes, so feature scaling is essential:

```python
# BAD: Regularization without scaling
ridge_bad = Ridge(alpha=1.0)
ridge_bad.fit(X_train, y_train)  # Features have different scales!

# GOOD: Standardize first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_good = Ridge(alpha=1.0)
ridge_good.fit(X_train_scaled, y_train)
```

Without standardization, features with larger natural scales dominate the penalty term, leading to unfair shrinkage.

### Intercept Term

Typically, we **do not** regularize the intercept term $\beta_0$. Scikit-learn handles this automatically with `fit_intercept=True` (the default).

### Regularization Path Algorithms

For Lasso and Elastic Net, specialized algorithms compute the entire regularization path (solutions for all $\lambda$ values) efficiently. Scikit-learn uses these algorithms internally in `LassoCV` and `ElasticNetCV`.

### Convergence and Tolerance

Lasso optimization uses iterative algorithms that may not converge with default settings for some problems. Increase `max_iter` or adjust `tol` if you see convergence warnings:

```python
lasso = Lasso(alpha=1.0, max_iter=10000, tol=1e-4)
```

## Ridge vs Lasso: Key Differences

The fundamental difference between Ridge and Lasso regularization lies in their behavior as $\lambda$ increases:

| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|------------|
| **Constraint Shape** | Circle: $\beta_1^2 + \beta_2^2 \leq t$ | Diamond: $\|\beta_1\| + \|\beta_2\| \leq t$ |
| **Coefficient Shrinkage** | Smooth, asymptotic to zero | Can reach exactly zero |
| **Feature Selection** | No (all coefficients remain) | Yes (automatic) |
| **Best When** | All features relevant | Many irrelevant features |
| **Handling Multicollinearity** | Excellent | Picks one feature arbitrarily |
| **Sparsity** | Dense solutions | Sparse solutions |
| **Computational Cost** | Closed-form solution | Iterative (coordinate descent) |

**When to Use:**
- **Ridge**: You believe most features contribute to the prediction, want stable coefficients, or have multicollinear features
- **Lasso**: You have many features and suspect only a subset are important, want an interpretable model, or need automatic feature selection
- **Elastic Net**: Combines both L1 and L2, balancing feature selection with handling multicollinearity

## Summary

Regularization is an essential technique for building machine learning models that generalize well beyond their training data. By adding penalty terms to the loss function, we constrain model complexity and prevent overfitting.

**L2 regularization** (Ridge) adds a penalty proportional to the sum of squared coefficients, shrinking them smoothly toward zero. Ridge is stable, fast, and works well when all features contribute to the prediction.

**L1 regularization** (Lasso) adds a penalty proportional to the sum of absolute coefficient values, driving some coefficients to exactly zero. Lasso performs automatic feature selection, producing sparse, interpretable models.

The choice between Ridge and Lasso depends on your problem characteristics and goals. Cross-validation provides a principled method for selecting the regularization parameter $\lambda$, balancing the bias-variance trade-off to optimize generalization performance.

These regularization techniques extend beyond linear regression to classification (logistic regression, SVMs) and deep learning (neural networks), making them fundamental tools in every machine learning practitioner's toolkit.

## Key Takeaways

1. **Regularization** prevents overfitting by adding a penalty term that discourages large coefficients
2. **L2 regularization** uses a squared penalty ($\sum \beta_j^2$) and shrinks coefficients smoothly toward zero
3. **Ridge regression** applies L2 regularization to linear regression, providing stable solutions
4. **L1 regularization** uses an absolute value penalty ($\sum |\beta_j|$) and can set coefficients to exactly zero
5. **Lasso regression** applies L1 regularization, performing automatic feature selection
6. **Geometric interpretation**: L2 creates circular constraints, L1 creates diamond constraints with corners on axes
7. **Cross-validation** is the standard method for selecting the optimal regularization strength $\lambda$
8. **Feature standardization** is essential before applying regularization to ensure fair penalization
9. **Ridge** is preferred when all features are relevant; **Lasso** when many features are irrelevant
10. **Elastic Net** combines L1 and L2 to balance their properties

## Further Reading

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 3: Linear Methods for Regression, Section 3.4)
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning* (Chapter 6: Linear Model Selection and Regularization)
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- Scikit-learn documentation: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

## Exercises

1. **Coefficient Paths**: Generate a synthetic dataset with 20 features where only 5 are truly predictive (others are noise). Fit Ridge and Lasso with a range of $\lambda$ values and plot coefficient paths. Which method correctly identifies the true features?

2. **Bias-Variance Decomposition**: Implement a simulation that computes bias and variance of Ridge predictions for different $\lambda$ values. Plot bias, variance, and total error vs $\lambda$ to visualize the bias-variance trade-off.

3. **Multicollinearity**: Create a dataset where two features are highly correlated ($r > 0.9$). Compare how Ridge and Lasso handle these correlated features as $\lambda$ increases.

4. **Cross-Validation Implementation**: Implement k-fold cross-validation from scratch to select the optimal $\lambda$ for Ridge regression. Compare your results to scikit-learn's `RidgeCV`.

5. **Regularization in Classification**: Apply L1 and L2 regularized logistic regression to a high-dimensional classification dataset (e.g., text classification with bag-of-words features). Analyze which features are selected by Lasso and their interpretation.

6. **Elastic Net Tuning**: Use grid search to find optimal values of both $\lambda$ and the L1/L2 mixing parameter for Elastic Net on a real dataset. Visualize the 2D grid of cross-validation scores.
