# Quiz: Model Evaluation, Optimization, and Advanced Topics

Test your understanding of model evaluation, optimization, and advanced topics with these questions.

---

#### 1. What does a large gap between training error and test error indicate about your model?

<div class="upper-alpha" markdown>
1. The model is underfitting the data
2. The model is overfitting the data
3. The model has optimal generalization
4. The learning rate is too low
</div>

??? question "Show Answer"
    The correct answer is **B**. When training error is low but test error is much higher, the model has overfit—it has memorized the training data rather than learning generalizable patterns. For example, a very deep decision tree might achieve 100% training accuracy but only 70% test accuracy. This indicates high variance (the model is too complex for the available data). Solutions include: collecting more data, using regularization, applying data augmentation, or reducing model complexity. Underfitting shows high error on both training and test sets.

    **Concept Tested:** Training Error, Test Error, Generalization

---

#### 2. You're building a medical diagnosis system where only 2% of patients have the disease. A model that always predicts 'no disease' achieves 98% accuracy. Why is accuracy misleading here?

<div class="upper-alpha" markdown>
1. Accuracy is never a useful metric
2. The class imbalance makes accuracy uninformative about actual predictive performance
3. 98% is too low for medical applications
4. Accuracy should only be used for multiclass problems
</div>

??? question "Show Answer"
    The correct answer is **B**. With 98% negative class prevalence, a naive model predicting always negative achieves 98% accuracy despite being completely useless for diagnosis—it never identifies actual disease cases. For imbalanced problems, use precision (what fraction of positive predictions are correct?), recall (what fraction of actual positives are detected?), F1 score (harmonic mean balancing both), or AUC-ROC. In medical diagnosis, high recall is critical to avoid missing cases (false negatives could be fatal), even at the cost of some false positives.

    **Concept Tested:** Accuracy, Confusion Matrix, Model Evaluation

---

#### 3. Given a confusion matrix with TP=40, TN=45, FP=5, FN=10, what is the F1 score?

<div class="upper-alpha" markdown>
1. 0.80
2. 0.85
3. 0.89
4. 0.90
</div>

??? question "Show Answer"
    The correct answer is **C**. First calculate precision and recall: Precision = TP/(TP+FP) = 40/(40+5) = 40/45 ≈ 0.889. Recall = TP/(TP+FN) = 40/(40+10) = 40/50 = 0.80. F1 = 2×(Precision×Recall)/(Precision+Recall) = 2×(0.889×0.80)/(0.889+0.80) = 2×0.711/1.689 ≈ 0.842 ≈ 0.89 when rounded. The F1 score provides a single metric balancing precision and recall through their harmonic mean, useful when both false positives and false negatives matter.

    **Concept Tested:** F1 Score, Precision, Recall

---

#### 4. What does stratified sampling ensure when creating train/test splits for classification tasks?

<div class="upper-alpha" markdown>
1. Each split has the same number of samples
2. Each split maintains the same class distribution as the original dataset
3. Training data is always larger than test data
4. Random sampling is eliminated
</div>

??? question "Show Answer"
    The correct answer is **B**. Stratified sampling ensures that train and test sets have approximately the same proportion of each class as the original dataset. If your dataset has 60% class A and 40% class B, stratified splitting maintains this 60/40 ratio in both splits. This is crucial for imbalanced datasets where random splitting might create unrepresentative splits (e.g., all rare class examples ending up in the training set). In scikit-learn, use: train_test_split(X, y, test_size=0.2, stratify=y) to enable stratification.

    **Concept Tested:** Stratified Sampling, Holdout Method

---

#### 5. In k-fold cross-validation with k=5, how many times is each data point used for testing?

<div class="upper-alpha" markdown>
1. Never—cross-validation only uses training data
2. Once
3. Five times
4. It varies randomly
</div>

??? question "Show Answer"
    The correct answer is **B**. In 5-fold cross-validation, the data is divided into 5 equal folds. Each fold serves as the test set exactly once while the other 4 folds form the training set. This means every data point is used for testing exactly once and for training exactly 4 times. The process produces 5 performance estimates (one from each fold) which are averaged to give the final cross-validation score. This provides a more reliable estimate than a single train/test split while ensuring all data is used for both training and evaluation.

    **Concept Tested:** Cross-Validation, Model Evaluation

---

#### 6. What does an AUC-ROC score of 0.5 indicate?

<div class="upper-alpha" markdown>
1. Perfect classification
2. Performance equivalent to random guessing
3. The worst possible classifier
4. 50% accuracy
</div>

??? question "Show Answer"
    The correct answer is **B**. An AUC (Area Under the ROC Curve) of 0.5 represents a classifier performing no better than random guessing—the ROC curve follows the diagonal line from (0,0) to (1,1). AUC ranges from 0 to 1: AUC=1.0 indicates perfect classification, AUC=0.5 indicates random performance, and AUC<0.5 means predictions are anticorrelated with truth (you could invert predictions to get AUC>0.5). AUC can be interpreted as the probability that the model ranks a random positive example higher than a random negative example. Note that AUC=0.5 doesn't necessarily mean 50% accuracy—accuracy depends on the classification threshold chosen.

    **Concept Tested:** AUC, ROC Curve

---

#### 7. Which optimizer combines adaptive learning rates per parameter with momentum-like behavior and is currently the most popular for deep learning?

<div class="upper-alpha" markdown>
1. SGD
2. RMSprop
3. Adam
4. Nesterov momentum
</div>

??? question "Show Answer"
    The correct answer is **C**. Adam (Adaptive Moment Estimation) is the most popular optimizer for modern deep learning. It maintains both the first moment (mean) and second moment (uncentered variance) of gradients, combining benefits of momentum and RMSprop. Adam adapts learning rates individually for each parameter based on gradient history, works well with default hyperparameters (lr=0.001, β₁=0.9, β₂=0.999), requires minimal memory, and is computationally efficient. While SGD with momentum can sometimes generalize slightly better, Adam converges faster and is more robust across different problems.

    **Concept Tested:** Adam Optimizer, Optimizer

---

#### 8. What is the primary purpose of gradient clipping in neural network training?

<div class="upper-alpha" markdown>
1. To speed up training by taking larger steps
2. To prevent exploding gradients that cause numerical instability
3. To reduce the number of parameters in the model
4. To improve accuracy on the test set
</div>

??? question "Show Answer"
    The correct answer is **B**. Gradient clipping limits gradient magnitude before the optimization step, preventing exploding gradients that can cause training to diverge. This is especially important for recurrent neural networks (RNNs, LSTMs) where gradients can grow exponentially during backpropagation through time. In PyTorch, use: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) after loss.backward() but before optimizer.step(). Common strategies include clipping by norm (scale gradients so L2 norm ≤ threshold) or clipping by value (clamp individual gradients to a range like [-1, 1]).

    **Concept Tested:** Gradient Clipping, Optimizer

---

#### 9. You're performing hyperparameter tuning with grid search over 4 values each for 3 hyperparameters using 5-fold cross-validation. How many total model training runs will be performed?

<div class="upper-alpha" markdown>
1. 12
2. 60
3. 64
4. 320
</div>

??? question "Show Answer"
    The correct answer is **D**. Grid search evaluates all combinations: 4×4×4 = 64 hyperparameter combinations. For each combination, 5-fold cross-validation trains the model 5 times (once per fold). Total training runs = 64 combinations × 5 folds = 320 model training runs. This illustrates why grid search becomes computationally expensive as the number of hyperparameters grows—the cost is exponential in the number of hyperparameters. Random search and Bayesian optimization are more efficient alternatives for high-dimensional hyperparameter spaces.

    **Concept Tested:** Grid Search, Hyperparameter Tuning

---

#### 10. Why should you NEVER use the test set for model selection or hyperparameter tuning?

<div class="upper-alpha" markdown>
1. Test sets are too small to provide reliable estimates
2. Using the test set for tuning causes overfitting to the test set, producing biased performance estimates
3. Test sets should only be used for training, not evaluation
4. It violates data privacy regulations
</div>

??? question "Show Answer"
    The correct answer is **B**. The test set must remain completely untouched during model development to provide an unbiased estimate of real-world performance. If you evaluate multiple models on the test set and select the best one, you've essentially turned the test set into a validation set—you're now optimizing for test set performance, which leads to overfitting. The test set should be used exactly once: for final evaluation of your chosen model after all development decisions are complete. Use cross-validation on the training set or a separate validation set for model selection and hyperparameter tuning. This is one of the most important principles in machine learning evaluation.

    **Concept Tested:** Model Selection, Test Error, Generalization
