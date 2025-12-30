# Quiz: Regularization Techniques

Test your understanding of regularization methods with these questions.

---

#### 1. What is the primary purpose of regularization in machine learning?

<div class="upper-alpha" markdown>
1. To speed up model training time
2. To prevent overfitting by penalizing model complexity
3. To increase the number of features in the model
4. To eliminate the need for cross-validation
</div>

??? question "Show Answer"
    The correct answer is **B**. Regularization prevents overfitting by adding a penalty term to the loss function that discourages large coefficient values. This forces the model to find simpler explanations that are more likely to generalize to unseen data. Without regularization, complex models can memorize training data including noise, leading to excellent training performance but poor test performance.

    **Concept Tested:** Regularization

---

#### 2. What is the mathematical form of the L2 penalty term in Ridge regression?

<div class="upper-alpha" markdown>
1. λ Σ |βⱼ|
2. λ Σ βⱼ
3. λ Σ log(βⱼ)
4. λ Σ βⱼ²
</div>

??? question "Show Answer"
    The correct answer is **D**. The L2 penalty in Ridge regression is λ Σ βⱼ², the sum of squared coefficients multiplied by the regularization parameter λ. This quadratic penalty grows rapidly with coefficient magnitude, encouraging the model to use smaller, more distributed weights. The squared term ensures the penalty is always positive and differentiable everywhere, leading to smooth, stable optimization.

    **Concept Tested:** L2 Regularization

---

#### 3. What distinguishes L1 regularization (Lasso) from L2 regularization (Ridge)?

<div class="upper-alpha" markdown>
1. L1 can drive coefficients to exactly zero, performing automatic feature selection
2. L1 always trains faster than L2
3. L1 uses squared penalties while L2 uses absolute value penalties
4. L1 requires fewer training examples than L2
</div>

??? question "Show Answer"
    The correct answer is **A**. L1 regularization (Lasso) uses an absolute value penalty λ Σ |βⱼ| that can drive coefficients to exactly zero, effectively removing features from the model. This automatic feature selection makes Lasso valuable for high-dimensional problems with many irrelevant features. In contrast, L2 (Ridge) uses squared penalties that shrink coefficients smoothly toward zero but rarely reach exactly zero.

    **Concept Tested:** L1 Regularization

---

#### 4. In scikit-learn's Ridge and Lasso classes, what does the alpha parameter control?

<div class="upper-alpha" markdown>
1. The learning rate for gradient descent
2. The number of features to select
3. The regularization strength (λ)
4. The train-test split ratio
</div>

??? question "Show Answer"
    The correct answer is **C**. The `alpha` parameter in scikit-learn's Ridge and Lasso classes directly controls the regularization strength λ. Larger alpha values apply stronger regularization (more penalty on large coefficients), leading to simpler models with smaller coefficient magnitudes. Alpha=0 corresponds to ordinary least squares with no regularization, while very large alpha values shrink coefficients heavily toward zero, potentially causing underfitting.

    **Concept Tested:** Ridge Regression

---

#### 5. Why is feature standardization essential before applying regularization?

<div class="upper-alpha" markdown>
1. It speeds up the optimization algorithm
2. Regularization penalizes coefficient magnitudes, and features on different scales lead to unfair penalization
3. Standardization is not necessary for regularization
4. It reduces the number of features needed
</div>

??? question "Show Answer"
    The correct answer is **B**. Regularization penalties like λ Σ βⱼ² treat all coefficients equally, but features with larger natural scales require larger coefficients to have the same predictive impact. Without standardization, features with large scales (e.g., income in dollars) would have their coefficients penalized more heavily than features with small scales (e.g., age in decades), even if they're equally important. Standardizing ensures all features contribute fairly to the penalty term.

    **Concept Tested:** Regularization

---

#### 6. Given a dataset with 100 features where you suspect only 10 are truly predictive, which regularization method would be most appropriate?

<div class="upper-alpha" markdown>
1. Lasso (L1) because it performs automatic feature selection
2. Ridge (L2) because it handles all features equally
3. No regularization because it would remove important features
4. L2 because it's computationally faster
</div>

??? question "Show Answer"
    The correct answer is **A**. Lasso regression is ideal when you suspect many features are irrelevant because its L1 penalty drives coefficients of unimportant features to exactly zero. This automatic feature selection would identify the approximately 10 truly predictive features while eliminating the 90 noise features, resulting in a sparse, interpretable model. Ridge would keep all 100 features with small but non-zero coefficients, which doesn't solve the feature selection problem.

    **Concept Tested:** Lasso Regression

---

#### 7. In the geometric interpretation of Ridge regression with two coefficients, what shape does the L2 constraint region form?

<div class="upper-alpha" markdown>
1. A square
2. A diamond
3. A triangle
4. A circle
</div>

??? question "Show Answer"
    The correct answer is **D**. The L2 constraint β₁² + β₂² ≤ t defines a circle (in higher dimensions, a hypersphere) centered at the origin. The Ridge solution occurs where the smallest error contour ellipse touches this circular constraint region. Because circles have smooth boundaries with no corners, the solution typically doesn't lie exactly on an axis, which is why Ridge rarely sets coefficients to exactly zero.

    **Concept Tested:** L2 Regularization

---

#### 8. What shape does the L1 constraint region form in two-dimensional coefficient space?

<div class="upper-alpha" markdown>
1. A circle
2. An ellipse
3. A diamond (rotated square)
4. A hexagon
</div>

??? question "Show Answer"
    The correct answer is **C**. The L1 constraint |β₁| + |β₂| ≤ t defines a diamond shape (a square rotated 45 degrees) with corners aligned on the coordinate axes at points like (±t, 0) and (0, ±t). When error contours touch this diamond-shaped constraint region, they frequently contact at a corner where one coefficient is exactly zero. This geometric property explains why Lasso performs automatic feature selection—the corners correspond to sparse solutions.

    **Concept Tested:** L1 Regularization

---

#### 9. You fit Ridge regression with alpha values [0.01, 0.1, 1.0, 10.0, 100.0] and observe the following test R² scores: [0.72, 0.78, 0.82, 0.79, 0.65]. What does this pattern suggest?

<div class="upper-alpha" markdown>
1. Alpha should be increased further to improve performance
2. The optimal alpha is around 1.0, balancing bias and variance
3. Regularization is not helping this problem
4. The model is underfitting at all alpha values
</div>

??? question "Show Answer"
    The correct answer is **B**. The test R² scores peak at alpha=1.0 (R²=0.82) and decline for both smaller and larger alpha values. This indicates that alpha=1.0 provides the optimal bias-variance trade-off: smaller alpha values (0.01, 0.1) underregularize and allow overfitting, while larger values (10.0, 100.0) overregularize and cause underfitting. The cross-validation curve shows the classic U-shape (inverted for R²), with the optimal alpha balancing model complexity and generalization.

    **Concept Tested:** Ridge Regression

---

#### 10. In scikit-learn's LogisticRegression, the C parameter is the inverse of regularization strength. If C=0.1, what does this imply?

<div class="upper-alpha" markdown>
1. Strong regularization (equivalent to large λ), encouraging simpler models
2. Weak regularization (equivalent to small λ), allowing complex models
3. No regularization is applied
4. The model will automatically select 10% of features
</div>

??? question "Show Answer"
    The correct answer is **A**. Since C = 1/λ in scikit-learn's LogisticRegression, a small C value like 0.1 corresponds to large λ (λ=10), applying strong regularization. This heavily penalizes large coefficients, forcing the model toward simpler solutions with smaller weights. Strong regularization helps prevent overfitting but risks underfitting if too strong. Typical C values range from 0.001 (very strong regularization) to 100 (very weak regularization).

    **Concept Tested:** Regularization
