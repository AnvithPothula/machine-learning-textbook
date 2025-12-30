# Quiz: Data Preprocessing and Feature Engineering

Test your understanding of data preprocessing and feature engineering with these questions.

---

#### 1. What is the primary difference between min-max scaling and z-score standardization?

<div class="upper-alpha" markdown>
1. Min-max scaling transforms features to have mean 0, while z-score standardization transforms to range [0,1]
2. Min-max scaling transforms to range [0,1], while z-score standardization transforms to mean 0 and standard deviation 1
3. Min-max scaling is only for categorical data, while z-score standardization is for numerical data
4. Min-max scaling removes outliers, while z-score standardization preserves them
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Min-max scaling (normalization) transforms features to a bounded range, typically [0, 1], using the formula $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$. Z-score standardization transforms features to have mean 0 and standard deviation 1 using $x' = \frac{x - \mu}{\sigma}$. Min-max scaling preserves the original distribution shape but is sensitive to outliers, while z-score standardization is less sensitive to outliers and is preferred when features are approximately Gaussian.

    **Concept Tested:** Min-Max Scaling, Z-Score Normalization, Normalization, Standardization

---

#### 2. Why should a scaler be fit only on training data and then applied to test data?

<div class="upper-alpha" markdown>
1. To reduce computational cost during testing
2. To prevent data leakage from test set into the training process
3. To ensure test data has exactly the same range as training data
4. To make the test set predictions more accurate
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Fitting the scaler on the test set would cause data leakage—information from the test set (its mean, standard deviation, min, max) would influence the model, violating the principle that test data should be completely unseen during training. The correct approach is: `scaler.fit_transform(X_train)` to compute statistics from training data, then `scaler.transform(X_test)` to apply those same transformations to test data. This simulates real-world deployment where new data must be preprocessed using only information available during training.

    **Concept Tested:** Data Preprocessing, Standardization, Train-Test Split

---

#### 3. A dataset contains a categorical variable 'education_level' with values: ['High School', 'Bachelor', 'Master', 'PhD']. Which encoding method would be most appropriate for this variable when using logistic regression?

<div class="upper-alpha" markdown>
1. Label encoding, because education level has a natural ordering
2. One-hot encoding, because logistic regression cannot handle ordinal data
3. Either could work, but one-hot encoding is safer if the spacing between levels is unknown
4. No encoding needed, logistic regression can directly use text labels
</div>

??? question "Show Answer"
    The correct answer is **C**.

    Education level is an ordinal variable with natural ordering (High School < Bachelor < Master < PhD). Label encoding (0, 1, 2, 3) could work if we assume equal spacing between levels. However, the "distance" from High School to Bachelor may not equal the distance from Master to PhD. One-hot encoding is safer because it makes no assumptions about spacing and allows the model to learn appropriate weights for each level independently. For logistic regression and most algorithms, one-hot encoding is the more robust choice unless you're certain about the ordinal relationship.

    **Concept Tested:** Label Encoding, One-Hot Encoding, Categorical Data

---

#### 4. When using one-hot encoding with `drop_first=True`, what problem does this parameter setting prevent?

<div class="upper-alpha" markdown>
1. Excessive memory usage from too many features
2. The dummy variable trap (perfect multicollinearity)
3. Loss of information about the reference category
4. Incorrect predictions for the first category
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Setting `drop_first=True` removes one binary column per categorical variable, preventing the dummy variable trap. For example, with species [Adelie, Chinstrap, Gentoo], if you know Chinstrap=0 and Gentoo=0, you can deduce Adelie=1. This makes one column redundant and creates perfect multicollinearity where one feature is perfectly predictable from others. This can cause numerical instability in some algorithms (especially those that invert matrices). Using $k-1$ columns for $k$ categories avoids this issue while preserving all information.

    **Concept Tested:** One-Hot Encoding, Data Preprocessing

---

#### 5. A machine learning engineer creates polynomial features of degree 3 from 5 original features. Approximately how many features will result (excluding the bias term)?

<div class="upper-alpha" markdown>
1. 15 features
2. 35 features
3. 56 features
4. 125 features
</div>

??? question "Show Answer"
    The correct answer is **C**.

    For $d$ original features and polynomial degree $p$, the number of polynomial features (including interactions) is $\binom{d+p}{p} - 1$ (excluding bias). For $d=5$ and $p=3$, this is $\binom{8}{3} - 1 = 56 - 1 = 55$ features. These include: 5 original features ($x_1, \ldots, x_5$), 10 pairwise interactions ($x_1 x_2, x_1 x_3, \ldots$), 10 degree-2 terms ($x_1^2, \ldots, x_5^2$), 10 triple interactions, 10 mixed degree-2 and degree-1 interactions, and 5 degree-3 terms. The rapid growth in features illustrates why high-degree polynomials can lead to overfitting.

    **Concept Tested:** Feature Engineering, Polynomial Features, Dimensionality

---

#### 6. Which feature selection method evaluates features based on their statistical properties independent of any machine learning model?

<div class="upper-alpha" markdown>
1. Wrapper methods
2. Filter methods
3. Embedded methods
4. Recursive feature elimination
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Filter methods select features based on statistical tests (like correlation, chi-square, ANOVA F-statistic) computed independently of the learning algorithm. They're fast and model-agnostic. Wrapper methods (option A) use the model itself to evaluate feature subsets, such as Recursive Feature Elimination (option D). Embedded methods (option C) perform feature selection as part of model training, like Lasso regression's L1 penalty that drives coefficients to zero. Filter methods are fastest but may miss feature interactions that models could exploit.

    **Concept Tested:** Feature Selection, Data Preprocessing

---

#### 7. In Principal Component Analysis (PCA), the first principal component represents:

<div class="upper-alpha" markdown>
1. The original feature with highest variance
2. The direction in feature space that captures maximum variance
3. The linear combination of features that best predicts the target variable
4. The feature most correlated with the target variable
</div>

??? question "Show Answer"
    The correct answer is **B**.

    PCA is an unsupervised dimensionality reduction technique that finds orthogonal directions (principal components) in feature space ordered by variance. The first principal component is the linear combination of original features that captures the maximum variance in the data. Subsequent components capture remaining variance while being orthogonal to previous components. PCA doesn't use target variable information (options C and D), making it different from supervised feature selection. It creates new features as combinations rather than selecting existing ones (option A).

    **Concept Tested:** Dimensionality Reduction, PCA, Feature Engineering

---

#### 8. A dataset has 1,000,000 samples and 100 features. What is the time complexity of standardizing all features using z-score normalization?

<div class="upper-alpha" markdown>
1. $O(n)$ where $n$ is the number of samples
2. $O(d)$ where $d$ is the number of features
3. $O(nd)$ where $n$ is samples and $d$ is features
4. $O(n^2 d)$
</div>

??? question "Show Answer"
    The correct answer is **C**.

    Z-score standardization requires computing the mean and standard deviation for each feature (one pass through the data: $O(nd)$), then transforming each value using $x' = \frac{x - \mu}{\sigma}$ (another pass: $O(nd)$). The total time complexity is $O(nd) + O(nd) = O(nd)$, which is linear in both dimensions. For this dataset, that's approximately $100,000,000$ operations. This linear scaling makes standardization computationally efficient even for large datasets, unlike polynomial feature generation or PCA which have higher complexity.

    **Concept Tested:** Time Complexity, Standardization, Computational Complexity

---

#### 9. Which data augmentation technique would be inappropriate for a digit classification task (0-9)?

<div class="upper-alpha" markdown>
1. Adding small amounts of random noise
2. Rotating images by 180 degrees
3. Slightly shifting images horizontally and vertically
4. Applying small random scaling
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Data augmentation should create variations that preserve the true label. Rotating digits by 180 degrees would change a '6' into a '9' or a '1' into an upside-down '1' that might be unrecognizable, fundamentally altering the meaning. Small rotations (like ±15 degrees), noise, shifts, and scaling are appropriate because they simulate real-world variations (handwriting angle, position, size) without changing what digit is represented. Augmentation guidelines: only apply transformations that reflect realistic variations and don't change the label.

    **Concept Tested:** Data Augmentation, Data Preprocessing

---

#### 10. A data scientist notices that K-Nearest Neighbors performs poorly on a dataset but a Random Forest performs well. The only difference in preprocessing was that features were standardized for KNN but not for Random Forest. Which explanation is most likely correct?

<div class="upper-alpha" markdown>
1. Random Forest requires unstandardized features to work properly
2. The standardization was done incorrectly for KNN
3. KNN is distance-based and needed standardization, while Random Forest is scale-invariant
4. Random Forest automatically standardizes features internally
</div>

??? question "Show Answer"
    The correct answer is **C**.

    KNN is a distance-based algorithm where features with larger scales dominate distance calculations, making standardization essential. Random Forest uses decision trees that split on threshold values—whether a feature is $x > 100$ or $x > 0.5$ doesn't fundamentally change the splits' effectiveness. Tree-based algorithms are scale-invariant. If both algorithms performed well, that would indicate the preprocessing was done correctly (ruling out option B). Random Forest doesn't automatically standardize (option D), and it certainly doesn't require unstandardized features (option A)—it simply doesn't care about scale.

    **Concept Tested:** Data Preprocessing, Feature Scaling, Algorithm Properties
