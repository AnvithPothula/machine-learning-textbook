# Quiz: Logistic Regression and Classification

Test your understanding of logistic regression and classification with these questions.

---

#### 1. What fundamental problem does the sigmoid function solve when using linear models for classification?

<div class="upper-alpha" markdown>
1. It speeds up the training process
2. It reduces the number of features needed
3. It constrains predictions to the interval [0, 1] for probabilistic interpretation
4. It eliminates the need for regularization
</div>

??? question "Show Answer"
    The correct answer is **C**. The sigmoid function transforms any real-valued output from a linear model into a value between 0 and 1, which can be interpreted as a probability. Linear regression produces unconstrained outputs (like 1.3 or -0.2) that are meaningless as probabilities or class labels. The sigmoid's S-shaped curve ensures all predictions fall within the valid probability range [0, 1].

    **Concept Tested:** Sigmoid Function

---

#### 2. In logistic regression for binary classification, what does the model output P(y = 1 | x) represent?

<div class="upper-alpha" markdown>
1. The probability that the instance belongs to class 1 given its features
2. The distance from the decision boundary
3. The number of training examples in class 1
4. The coefficient value for the first feature
</div>

??? question "Show Answer"
    The correct answer is **A**. Logistic regression models the probability P(y = 1 | x) that an instance with features x belongs to class 1. This probabilistic interpretation is a key advantage of logistic regression—it provides not just a class prediction but also a confidence measure. A decision threshold (typically 0.5) converts this probability into a binary class prediction.

    **Concept Tested:** Binary Classification

---

#### 3. What is the mathematical form of the sigmoid function?

<div class="upper-alpha" markdown>
1. σ(z) = z / (1 + z)
2. σ(z) = e^z / z
3. σ(z) = 1 / (1 + z²)
4. σ(z) = 1 / (1 + e^(-z))
</div>

??? question "Show Answer"
    The correct answer is **D**. The sigmoid (logistic) function is defined as σ(z) = 1 / (1 + e^(-z)), where z is any real number. This function has the crucial properties of being monotonic, differentiable, and constrained to the range (0, 1). As z approaches positive infinity, σ(z) approaches 1; as z approaches negative infinity, σ(z) approaches 0; and when z = 0, σ(z) = 0.5.

    **Concept Tested:** Sigmoid Function

---

#### 4. What loss function does logistic regression minimize during training?

<div class="upper-alpha" markdown>
1. Mean squared error
2. Log-loss (binary cross-entropy)
3. Absolute error
4. Hinge loss
</div>

??? question "Show Answer"
    The correct answer is **B**. Logistic regression minimizes log-loss, also called binary cross-entropy, defined as -[y log(p) + (1-y) log(1-p)] averaged over all instances. Log-loss heavily penalizes confident wrong predictions—if the true label is 1 and the model predicts probability near 0, the loss approaches infinity. This is derived from maximum likelihood estimation, making it the theoretically principled objective for probabilistic classification.

    **Concept Tested:** Log-Loss

---

#### 5. Given a true label of 1 and a predicted probability of 0.1, why does log-loss assign a large penalty?

<div class="upper-alpha" markdown>
1. The model is confidently wrong, predicting low probability for the actual positive class
2. The predicted value is less than 0.5
3. The model requires more training iterations
4. The feature values are not normalized
</div>

??? question "Show Answer"
    The correct answer is **A**. Log-loss for this case is -log(0.1) ≈ 2.303, which is quite large. The model assigned only 10% probability to class 1 when the true label was actually 1, meaning it was very confident in the wrong prediction. Log-loss grows exponentially as predicted probability moves away from the true label, encouraging the model to avoid overconfident mistakes. In contrast, predicting 0.9 for a true label of 1 yields a small loss of only -log(0.9) ≈ 0.105.

    **Concept Tested:** Log-Loss

---

#### 6. How many binary classifiers must be trained for a 5-class problem using the one-vs-one strategy?

<div class="upper-alpha" markdown>
1. 5 classifiers
2. 4 classifiers
3. 10 classifiers
4. 25 classifiers
</div>

??? question "Show Answer"
    The correct answer is **C**. One-vs-one trains a binary classifier for every pair of classes, requiring C(K,2) = K(K-1)/2 classifiers for K classes. For 5 classes, this is 5×4/2 = 10 classifiers. Each classifier learns to distinguish one specific pair of classes (e.g., class 1 vs class 2, class 1 vs class 3, etc.). During prediction, all 10 classifiers vote, and the class with the most votes wins. This grows quadratically—10 classes require 45 classifiers, making one-vs-one expensive for large K.

    **Concept Tested:** One-vs-One

---

#### 7. What is the primary advantage of the one-vs-all (one-vs-rest) multiclass strategy?

<div class="upper-alpha" markdown>
1. It produces perfectly balanced datasets for each classifier
2. It requires training only K binary classifiers for K classes
3. It always achieves higher accuracy than other strategies
4. It eliminates the need for probability calibration
</div>

??? question "Show Answer"
    The correct answer is **B**. One-vs-all trains exactly K binary classifiers for a K-class problem—one classifier per class that learns to distinguish "this class" from "all other classes." This is computationally more efficient than one-vs-one, which requires K(K-1)/2 classifiers. For example, with 10 classes, one-vs-all needs only 10 classifiers while one-vs-one needs 45. The main disadvantage is class imbalance, as each classifier sees one positive class versus many negatives combined.

    **Concept Tested:** One-vs-All

---

#### 8. What property does the softmax function guarantee for multiclass probability predictions?

<div class="upper-alpha" markdown>
1. All probabilities are negative
2. The highest probability is always exactly 1.0
3. Probabilities are evenly distributed across classes
4. All probabilities are positive and sum to exactly 1
</div>

??? question "Show Answer"
    The correct answer is **D**. The softmax function P(y = k | x) = e^(z_k) / Σe^(z_j) ensures that (1) all probabilities are positive due to the exponential, and (2) they sum to exactly 1 across all K classes due to the normalization by the sum in the denominator. This creates a valid probability distribution over the classes. The class with the highest linear score z_k receives the highest probability, but it's not necessarily 1.0—all classes receive non-zero probability based on their relative scores.

    **Concept Tested:** Softmax Function

---

#### 9. You're training logistic regression on a dataset with 1,000 features and 500 examples. The model achieves 99% training accuracy but only 65% test accuracy. What is the most likely cause and solution?

<div class="upper-alpha" markdown>
1. Overfitting due to high dimensionality; decrease the C parameter to increase regularization
2. Underfitting; increase the number of features
3. The model needs more training iterations
4. The sigmoid function is not appropriate for this problem
</div>

??? question "Show Answer"
    The correct answer is **A**. The large gap between training accuracy (99%) and test accuracy (65%) is a classic sign of overfitting. With more features (1,000) than examples (500), the model has high capacity to memorize the training set. Decreasing C in scikit-learn's LogisticRegression increases regularization strength (C is the inverse of regularization), which penalizes large coefficient values and encourages simpler models that generalize better. Typical values to try would be C = 0.1, 0.01, or 0.001.

    **Concept Tested:** Logistic Regression

---

#### 10. In the context of logistic regression and neural networks, what role does the sigmoid play as an activation function?

<div class="upper-alpha" markdown>
1. It reduces computation time during training
2. It stores the weights and biases
3. It introduces nonlinearity by transforming weighted sums into probability-like outputs
4. It automatically selects the most important features
</div>

??? question "Show Answer"
    The correct answer is **C**. The sigmoid activation function transforms the linear weighted sum of inputs (z = w₁x₁ + w₂x₂ + ... + b) into a nonlinear, bounded output in the range (0, 1). This nonlinearity is crucial—without it, stacking multiple linear transformations would still produce only linear models. In neural networks, sigmoid units were historically used to introduce this nonlinearity, though modern networks often prefer ReLU or other activations. The S-shaped curve enables the network to learn complex, nonlinear decision boundaries.

    **Concept Tested:** Sigmoid Activation
