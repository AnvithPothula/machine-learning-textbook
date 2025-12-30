# Quiz: Introduction to Machine Learning Fundamentals

Test your understanding of machine learning fundamentals with these questions.

---

#### 1. What is the fundamental characteristic that distinguishes machine learning from traditional programming?

<div class="upper-alpha" markdown>
1. Machine learning uses faster algorithms
2. Machine learning learns patterns from data rather than following explicit rules
3. Machine learning requires more memory
4. Machine learning only works with numerical data
</div>

??? question "Show Answer"
    The correct answer is **B**. Machine learning enables systems to learn patterns from data without being explicitly programmed for every scenario. Traditional programming relies on predefined rules, while machine learning algorithms discover statistical relationships and build models that generalize to new examples.

    **Concept Tested:** Machine Learning

---

#### 2. Which statement best describes supervised learning?

<div class="upper-alpha" markdown>
1. The algorithm discovers patterns in data without any labels
2. The algorithm learns from labeled examples to predict outputs for new inputs
3. The algorithm groups similar data points together
4. The algorithm reduces the number of features in a dataset
</div>

??? question "Show Answer"
    The correct answer is **B**. Supervised learning algorithms learn from labeled examples, where each data point includes both input features and a known output (label). The algorithm's objective is to discover a mapping function that accurately predicts labels for new, unseen inputs. Options A, C, and D describe unsupervised learning tasks.

    **Concept Tested:** Supervised Learning

---

#### 3. What is the primary difference between classification and regression?

<div class="upper-alpha" markdown>
1. Classification predicts categorical labels, while regression predicts continuous numerical values
2. Classification is faster than regression
3. Classification requires more training data than regression
4. Classification can only handle binary outcomes
</div>

??? question "Show Answer"
    The correct answer is **A**. Classification predicts discrete categorical labels (classes) from input features, while regression predicts continuous numerical values. For example, predicting whether a tumor is malignant or benign is classification, while predicting house prices in dollars is regression. Both are supervised learning tasks but differ in output type.

    **Concept Tested:** Classification vs Regression

---

#### 4. In machine learning, what is a feature?

<div class="upper-alpha" markdown>
1. The final prediction made by a model
2. A measurable property or characteristic used as input to an algorithm
3. The error rate of a trained model
4. A technique for reducing overfitting
</div>

??? question "Show Answer"
    The correct answer is **B**. A feature (also called an attribute or variable) is a measurable property or characteristic of the data used as input to a machine learning algorithm. For example, in house price prediction, features might include square footage, number of bedrooms, and neighborhood. Features form the input X in supervised learning.

    **Concept Tested:** Feature

---

#### 5. Why is it crucial to maintain separate training and test datasets?

<div class="upper-alpha" markdown>
1. To reduce the computational cost of training
2. To ensure models are evaluated on data they haven't seen during training
3. To increase the amount of available data
4. To speed up the training process
</div>

??? question "Show Answer"
    The correct answer is **B**. Maintaining separate training and test datasets ensures that models are evaluated on data they have never seen during training. This separation provides an unbiased estimate of how the model will perform on real-world, unseen data and helps detect overfitting. If we tested on training data, we'd get overly optimistic performance estimates.

    **Concept Tested:** Training Data vs Test Data

---

#### 6. What distinguishes a continuous feature from a categorical feature?

<div class="upper-alpha" markdown>
1. Continuous features can take any value within a range, while categorical features represent discrete categories
2. Continuous features are always more important than categorical features
3. Continuous features require less storage space
4. Categorical features can only be binary
</div>

??? question "Show Answer"
    The correct answer is **A**. Continuous features are numerical values that can take any value within a range (like temperature, height, or price) and have infinite possible values. Categorical features represent discrete categories or classes (like color, country, or yes/no responses) with a finite set of possible values and no inherent numerical meaning.

    **Concept Tested:** Continuous Features vs Categorical Features

---

#### 7. Given a dataset with 1,000 examples, you need to train a model and evaluate it. What would be a typical data split strategy?

<div class="upper-alpha" markdown>
1. Use all 1,000 examples for training, then create new data for testing
2. Split into 60% training, 20% validation, 20% test
3. Use 50% for training and 50% for testing
4. Randomly select examples during training without any fixed split
</div>

??? question "Show Answer"
    The correct answer is **B**. A typical split is 60% training (for fitting model parameters), 20% validation (for tuning hyperparameters and model selection), and 20% test (for final unbiased evaluation). Other common splits are 70/15/15 or 80/10/10. The key is maintaining three separate partitions: training data teaches the model, validation data helps choose between models, and test data gives honest final performance.

    **Concept Tested:** Training, Validation, and Test Data

---

#### 8. What is the primary purpose of k-fold cross-validation?

<div class="upper-alpha" markdown>
1. To reduce the size of the training dataset
2. To provide more reliable performance estimates by training and evaluating on multiple data splits
3. To eliminate the need for a test set
4. To automatically select the best machine learning algorithm
</div>

??? question "Show Answer"
    The correct answer is **B**. K-fold cross-validation provides more reliable performance estimates by repeatedly splitting data into training and validation sets using different partitions, training a model on each split, and averaging validation performance. This reduces variance in estimates from a single random split and makes efficient use of limited data. It's used on training data only—the test set remains separate.

    **Concept Tested:** K-Fold Cross-Validation

---

#### 9. What is the key distinction between model parameters and hyperparameters?

<div class="upper-alpha" markdown>
1. Parameters are set before training, hyperparameters are learned during training
2. Parameters are learned from data during training, hyperparameters are configuration settings specified before training
3. Parameters are only used in neural networks
4. Hyperparameters determine the size of the dataset
</div>

??? question "Show Answer"
    The correct answer is **B**. Model parameters (like weights in linear regression) are learned from data during training by the optimization algorithm. Hyperparameters (like learning rate, number of layers, regularization strength) are configuration settings that control the learning process but are specified before training begins and must be tuned through experimentation.

    **Concept Tested:** Hyperparameter

---

#### 10. In the context of unsupervised learning, what does clustering accomplish?

<div class="upper-alpha" markdown>
1. It predicts numerical values from input features
2. It groups similar data points together based on their characteristics
3. It removes outliers from the dataset
4. It converts categorical features to numerical features
</div>

??? question "Show Answer"
    The correct answer is **B**. Clustering is an unsupervised learning task that groups similar data points together based on their characteristics without using predefined labels. For example, clustering customer purchase data might identify distinct customer segments (budget shoppers, luxury buyers) based solely on purchasing patterns, discovering structure that wasn't explicitly labeled.

    **Concept Tested:** Unsupervised Learning (Clustering)
