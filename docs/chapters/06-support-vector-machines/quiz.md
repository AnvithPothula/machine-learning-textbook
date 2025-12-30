# Quiz: Support Vector Machines

Test your understanding of Support Vector Machines with these questions.

---

#### 1. What is the fundamental principle that SVMs use to find the optimal decision boundary?

<div class="upper-alpha" markdown>
1. Minimize the number of misclassified training examples
2. Minimize the total distance to all training points
3. Maximize the margin between classes
4. Maximize the probability of correct classification
</div>

??? question "Show Answer"
    The correct answer is **C**. SVMs find the optimal decision boundary by maximizing the margin—the distance to the nearest training examples from each class. This principle of margin maximization leads to classifiers that generalize well and have strong theoretical guarantees. The margin represents the "widest street" that separates the two classes, and SVMs choose the unique hyperplane that lies exactly in the middle of this gap.

    **Concept Tested:** Support Vector Machine

---

#### 2. In d-dimensional feature space, what is a hyperplane?

<div class="upper-alpha" markdown>
1. A d-dimensional curved surface
2. A (d-1)-dimensional flat subspace defined by w^T x + b = 0
3. A point at the center of the data
4. The set of all support vectors
</div>

??? question "Show Answer"
    The correct answer is **B**. A hyperplane in d-dimensional space is a (d-1)-dimensional flat subspace defined by the equation w^T x + b = 0, where w is the weight vector normal to the hyperplane and b is the bias term. For 2D data, a hyperplane is a line; for 3D data, it's a plane. The hyperplane divides the space into two half-spaces, with one side classified as positive (w^T x + b > 0) and the other as negative.

    **Concept Tested:** Hyperplane

---

#### 3. What are support vectors in an SVM classifier?

<div class="upper-alpha" markdown>
1. All training data points used to train the model
2. The weight vector perpendicular to the decision boundary
3. Points in the test set that are difficult to classify
4. Training points that lie exactly on the margin boundaries
</div>

??? question "Show Answer"
    The correct answer is **D**. Support vectors are the training points that lie exactly on the margin boundaries—the data points closest to the decision boundary. These critical points define the optimal hyperplane. Remarkably, the SVM solution depends only on support vectors; all other training points could be removed without changing the decision boundary. This sparsity property makes SVMs computationally efficient and resistant to outliers far from the boundary.

    **Concept Tested:** Support Vectors

---

#### 4. For a hyperplane defined by weight vector w, what is the formula for the margin width?

<div class="upper-alpha" markdown>
1. 2 / ||w||
2. ||w|| / 2
3. 2 × ||w||
4. 1 / ||w||²
</div>

??? question "Show Answer"
    The correct answer is **A**. The margin width is 2 / ||w||, where ||w|| is the magnitude (norm) of the weight vector. The distance from a single point to the hyperplane is |w^T x + b| / ||w||, and the margin encompasses points on both sides of the decision boundary, so the total width is twice this distance. SVMs maximize the margin by minimizing ||w||², which is equivalent to maximizing 2 / ||w||.

    **Concept Tested:** Margin

---

#### 5. What is the main limitation of hard margin SVMs that soft margin SVMs address?

<div class="upper-alpha" markdown>
1. Hard margin SVMs train too slowly
2. Hard margin SVMs cannot use kernel functions
3. Hard margin SVMs require perfect linear separability and cannot tolerate any misclassifications
4. Hard margin SVMs only work with two features
</div>

??? question "Show Answer"
    The correct answer is **C**. Hard margin SVMs require the data to be perfectly linearly separable and enforce that all training points be on the correct side of the margin with no violations. If even one point cannot be correctly classified with a linear boundary, no solution exists. Soft margin SVMs relax this constraint by introducing slack variables that allow some points to violate the margin or be misclassified, making them practical for real-world noisy data.

    **Concept Tested:** Hard Margin SVM

---

#### 6. In soft margin SVMs, what do slack variables (ξᵢ) represent?

<div class="upper-alpha" markdown>
1. The distance between support vectors
2. The degree of margin violation or misclassification for each point
3. The kernel function parameters
4. The regularization strength
</div>

??? question "Show Answer"
    The correct answer is **B**. Slack variables ξᵢ ≥ 0 represent the degree of margin violation for each training point i. When ξᵢ = 0, the point is on or outside the correct margin boundary (no violation). When 0 < ξᵢ < 1, the point is inside the margin but correctly classified. When ξᵢ ≥ 1, the point is misclassified. The soft margin objective minimizes ½||w||² + C Σξᵢ, balancing margin width against violations controlled by parameter C.

    **Concept Tested:** Slack Variables

---

#### 7. How does the C parameter in soft margin SVMs affect the model?

<div class="upper-alpha" markdown>
1. Larger C creates wider margins and more tolerance for violations
2. C controls the number of support vectors directly
3. C has no effect on model performance
4. Larger C heavily penalizes violations, prioritizing correct classification but risking overfitting
</div>

??? question "Show Answer"
    The correct answer is **D**. The parameter C > 0 controls the trade-off between margin width and margin violations. Large C heavily penalizes violations, forcing the model to prioritize correct classification of training points, which can lead to overfitting with narrow, complex margins. Small C tolerates violations, prioritizing a large margin over perfect classification, which can lead to underfitting but better generalization. The C parameter is analogous to inverse regularization strength.

    **Concept Tested:** Soft Margin SVM

---

#### 8. What is the fundamental advantage of the kernel trick in SVMs?

<div class="upper-alpha" markdown>
1. It allows SVMs to learn nonlinear decision boundaries by implicitly mapping data to higher-dimensional spaces
2. It reduces the number of support vectors needed
3. It eliminates the need for the C parameter
4. It makes training faster than linear SVMs
</div>

??? question "Show Answer"
    The correct answer is **A**. The kernel trick allows SVMs to learn complex nonlinear decision boundaries by implicitly mapping data to higher-dimensional spaces where they become linearly separable. Remarkably, this is done without explicitly computing the transformation φ(x)—instead, kernel functions K(x, z) compute inner products in the transformed space directly. This enables SVMs to handle XOR patterns, concentric circles, and other nonlinear problems that linear classifiers cannot solve.

    **Concept Tested:** Kernel Trick

---

#### 9. What type of decision boundary does the linear kernel K(x, z) = x^T z create?

<div class="upper-alpha" markdown>
1. Polynomial curves
2. Radial boundaries around support vectors
3. Linear hyperplanes with no transformation
4. Exponential boundaries
</div>

??? question "Show Answer"
    The correct answer is **C**. The linear kernel K(x, z) = x^T z is equivalent to the standard inner product with no transformation. It creates straight-line decision boundaries (hyperplanes) in the original feature space. Use the linear kernel when data is linearly separable or nearly so. It's the simplest kernel and should be tried first as a baseline before considering more complex nonlinear kernels.

    **Concept Tested:** Linear Kernel

---

#### 10. In the RBF (Gaussian) kernel, what effect does the γ (gamma) parameter have on the decision boundary?

<div class="upper-alpha" markdown>
1. γ controls the number of support vectors only
2. Large γ makes each point influence only nearby regions, creating complex boundaries that risk overfitting
3. γ has no effect on the decision boundary shape
4. Small γ always produces better accuracy than large γ
</div>

??? question "Show Answer"
    The correct answer is **B**. The RBF kernel K(x, z) = exp(-γ ||x - z||²) uses γ to control the influence radius of training points. Large γ means each point influences only its immediate neighborhood, creating highly complex, wiggly boundaries that can overfit to training data. Small γ means each point influences broader regions, creating smoother, simpler boundaries. The γ parameter (also called kernel size or bandwidth) must be tuned via cross-validation to balance complexity and generalization.

    **Concept Tested:** Radial Basis Function
