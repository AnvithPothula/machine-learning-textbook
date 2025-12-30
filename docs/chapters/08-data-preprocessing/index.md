---
title: Data Preprocessing and Feature Engineering
description: Transforming raw data into effective representations for machine learning algorithms
generated_by: claude skill chapter-content-generator
date: 2025-12-28
version: 0.03
---

# Data Preprocessing and Feature Engineering

## Summary

This chapter covers essential data preprocessing and feature engineering techniques that are critical for successful machine learning applications. Students will learn how to prepare raw data for machine learning algorithms through normalization and standardization (min-max scaling, z-score normalization), understand encoding strategies for categorical variables (one-hot encoding, label encoding), and explore feature engineering methods to create more informative representations. The chapter also introduces dimensionality reduction concepts and data augmentation techniques particularly useful for neural networks. These preprocessing skills are fundamental for building effective machine learning pipelines in real-world applications.

## Concepts Covered

This chapter covers the following 15 concepts from the learning graph:

1. Data Preprocessing
2. Normalization
3. Standardization
4. Min-Max Scaling
5. Z-Score Normalization
6. One-Hot Encoding
7. Label Encoding
8. Feature Engineering
9. Feature Selection
10. Dimensionality Reduction
11. Data Augmentation
12. Computational Complexity
13. Time Complexity
14. Space Complexity
15. Scalability

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Machine Learning Fundamentals](../01-intro-to-ml-fundamentals/index.md)
- [Chapter 2: K-Nearest Neighbors Algorithm](../02-k-nearest-neighbors/index.md)

---

## The Importance of Data Preprocessing

Raw data rarely comes in a form suitable for direct use in machine learning algorithms. Real-world datasets contain missing values, inconsistent formats, features on vastly different scales, categorical variables that need numerical encoding, and irrelevant or redundant information. **Data preprocessing** encompasses all the transformations applied to raw data to make it suitable for machine learning algorithms.

Effective preprocessing can mean the difference between a model that fails to learn and one that achieves state-of-the-art performance. Consider these scenarios:

- A k-nearest neighbors classifier using features measured in kilometers and millimeters—the distance metric becomes dominated by the kilometer-scale feature
- A neural network trying to learn from categorical data encoded as arbitrary numbers (e.g., "red"=1, "green"=2, "blue"=3)—the model incorrectly assumes "green" is somehow between "red" and "blue"
- A decision tree with 10,000 features where only 10 are truly predictive—training is slow and the model overfits

Preprocessing addresses these issues systematically, transforming raw data into representations that algorithms can effectively learn from.

## Feature Scaling: Normalization and Standardization

Many machine learning algorithms are sensitive to the scale of features. Distance-based algorithms (k-NN, SVMs, k-means) and gradient-based optimization (neural networks, logistic regression) can perform poorly when features have vastly different ranges.

### The Scale Problem

Consider a dataset with two features:
- **Feature 1**: Annual income (range: $20,000 - $200,000)
- **Feature 2**: Age (range: 18 - 65 years)

When computing Euclidean distance for k-NN, a $1,000 difference in income contributes far more to the distance than a 1-year difference in age, even though age might be equally or more predictive. Similarly, in gradient descent optimization, features with larger scales can dominate the gradient updates, slowing convergence or preventing the algorithm from finding optimal solutions.

**Feature scaling** addresses this by transforming features to comparable ranges or distributions.

### Min-Max Scaling (Normalization)

**Min-max scaling** (also called **normalization**) transforms features to a specific range, typically [0, 1]:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

where:
- $x$ is the original feature value
- $x_{\min}$ and $x_{\max}$ are the minimum and maximum values in the feature
- $x'$ is the scaled value

**Properties:**
- All scaled values fall in [0, 1]
- Preserves the original distribution shape
- Sensitive to outliers (extreme values affect $x_{\min}$ and $x_{\max}$)
- Useful when you need a bounded range or when the distribution isn't Gaussian

**Example:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample data: house prices and square footage
data = pd.DataFrame({
    'price': [150000, 200000, 180000, 250000, 300000],
    'sqft': [1200, 1500, 1350, 1800, 2000],
    'age': [5, 10, 8, 3, 15]
})

print("Original data:")
print(data)
print("\nData statistics:")
print(data.describe())

# Apply min-max scaling
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Convert back to DataFrame for display
data_normalized_df = pd.DataFrame(
    data_normalized,
    columns=data.columns,
    index=data.index
)

print("\nNormalized data (0-1 range):")
print(data_normalized_df)
print("\nNormalized data statistics:")
print(data_normalized_df.describe())
```

After min-max scaling, all features range from 0 to 1, making them directly comparable.

### Z-Score Normalization (Standardization)

**Z-score normalization** (also called **standardization**) transforms features to have mean 0 and standard deviation 1:

$$x' = \frac{x - \mu}{\sigma}$$

where:
- $x$ is the original feature value
- $\mu$ is the mean of the feature
- $\sigma$ is the standard deviation of the feature
- $x'$ is the standardized value (z-score)

**Properties:**
- Scaled values have mean 0 and standard deviation 1
- No bounded range (values can be negative or greater than 1)
- Less sensitive to outliers than min-max scaling
- Assumes or creates approximately Gaussian distribution
- Preferred for most machine learning algorithms, especially those assuming Gaussian-distributed features

**Example:**

```python
from sklearn.preprocessing import StandardScaler

# Apply standardization
std_scaler = StandardScaler()
data_standardized = std_scaler.fit_transform(data)

# Convert to DataFrame
data_standardized_df = pd.DataFrame(
    data_standardized,
    columns=data.columns,
    index=data.index
)

print("Standardized data (mean=0, std=1):")
print(data_standardized_df)
print("\nStandardized data statistics:")
print(data_standardized_df.describe())

# Verify mean ≈ 0 and std ≈ 1
print("\nMeans:", data_standardized_df.mean())
print("Standard deviations:", data_standardized_df.std())
```

After standardization, features have mean 0 and standard deviation 1, making them comparable while preserving information about the variability within each feature.

### Comparing Normalization and Standardization

| Property | Min-Max Scaling | Z-Score Standardization |
|----------|----------------|------------------------|
| **Output range** | [0, 1] (or custom) | Unbounded |
| **Mean** | Depends on data | 0 |
| **Std deviation** | Depends on data | 1 |
| **Outlier sensitivity** | High | Lower |
| **Use when** | Need bounded range | Features are approximately Gaussian |
| **Good for** | Neural networks, image data | SVM, logistic regression, k-NN |

### When to Scale Features

**Always scale** for:
- K-nearest neighbors (distance-based)
- Support vector machines (distance-based)
- Neural networks (gradient-based optimization)
- K-means clustering (distance-based)
- Principal Component Analysis (variance-based)
- Regularized models (L1/L2 penalties assume comparable scales)

**No need to scale** for:
- Decision trees and random forests (split points are scale-invariant)
- Naive Bayes (works with probabilities)

!!! warning "Fit on Training Data Only"
    Always fit the scaler on training data and apply the same transformation to test data:

    ```python
    # Correct: Fit on training, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use training statistics

    # Incorrect: Fitting separately causes data leakage
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)  # Wrong!
    ```

## Encoding Categorical Variables

Machine learning algorithms require numerical input, but real-world data often contains categorical variables (text labels like "red," "green," "blue" or "small," "medium," "large"). **Encoding** transforms categorical variables into numerical representations.

### Label Encoding

**Label encoding** assigns each unique category an integer label. For a variable with $k$ categories, labels range from 0 to $k-1$.

**Example:**

```python
import pandas as pd

# Create sample data with categorical features
penguins = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/penguins.csv')

print("Original species data:")
print(penguins['species'].head(10))

# Apply label encoding using factorize
labels, unique_species = pd.factorize(penguins['species'])

print("\nLabel encoded species:")
print(labels[:10])
print("\nMapping:")
for i, species in enumerate(unique_species):
    print(f"  {species} → {i}")
```

**Advantages:**
- Simple and memory-efficient
- Preserves single-column structure
- Works well for ordinal variables (e.g., "small" < "medium" < "large")

**Disadvantages:**
- Introduces artificial ordering for nominal variables
- The model might incorrectly assume category 2 is "between" categories 1 and 3
- Not suitable for most algorithms with nominal categorical data

**When to use:**
- Target variable in classification (y labels)
- Ordinal categorical variables with natural ordering
- Tree-based algorithms (they learn to split on categorical values)

### One-Hot Encoding

**One-hot encoding** creates a binary column for each category, with 1 indicating presence and 0 indicating absence. For $k$ categories, this creates $k$ new binary features (or $k-1$ with `drop_first=True` to avoid multicollinearity).

**Example:**

```python
# Apply one-hot encoding
penguins_encoded = pd.get_dummies(penguins, columns=['species', 'island', 'sex'],
                                   drop_first=True, dtype='int')

print("Original penguins data shape:", penguins.shape)
print("\nOne-hot encoded shape:", penguins_encoded.shape)
print("\nEncoded columns:")
print(penguins_encoded.columns.tolist())

# Display first few rows
print("\nFirst 5 rows of encoded data:")
print(penguins_encoded.head())
```

**Interpretation:**
- `species_Chinstrap=1, species_Gentoo=0` → Chinstrap penguin
- `species_Chinstrap=0, species_Gentoo=1` → Gentoo penguin
- `species_Chinstrap=0, species_Gentoo=0` → Adelie penguin (reference category)

**Advantages:**
- No artificial ordering imposed
- Works with all machine learning algorithms
- Clear interpretation: each column represents presence/absence of a category

**Disadvantages:**
- Increases dimensionality significantly (creates $k$ or $k-1$ columns per categorical feature)
- Sparse representation for high-cardinality features (many categories)
- Can lead to multicollinearity if `drop_first=False`

**When to use:**
- Nominal categorical variables (no natural ordering)
- Linear models, neural networks, k-NN, SVM
- When number of categories is manageable (<50)

### drop_first Parameter

Setting `drop_first=True` removes one binary column per categorical variable:

```python
# Without drop_first: k columns for k categories
penguins_full = pd.get_dummies(penguins, columns=['species'], dtype='int')
print("Without drop_first:", penguins_full.filter(like='species').columns.tolist())

# With drop_first: k-1 columns for k categories
penguins_reduced = pd.get_dummies(penguins, columns=['species'],
                                   drop_first=True, dtype='int')
print("With drop_first:", penguins_reduced.filter(like='species').columns.tolist())
```

Using `drop_first=True` avoids the **dummy variable trap** (perfect multicollinearity) where one column is perfectly predictable from the others. For instance, if `species_Chinstrap=0` and `species_Gentoo=0`, then we know the species must be Adelie.

!!! note "Regularized Models and drop_first"
    For models with regularization (Ridge, Lasso, Elastic Net), the dummy variable trap is less critical because regularization handles multicollinearity. However, using `drop_first=True` is still recommended for interpretability and reduced dimensionality.

## Feature Engineering

**Feature engineering** is the process of creating new features from existing ones to better represent the underlying patterns in data. Good features can dramatically improve model performance, often more than sophisticated algorithms or extensive hyperparameter tuning.

### Domain-Driven Features

Domain knowledge guides the creation of meaningful features:

```python
# Example: Engineering features for house price prediction
houses = pd.DataFrame({
    'bedrooms': [3, 4, 2, 5, 3],
    'bathrooms': [2, 3, 1, 3, 2],
    'sqft': [1500, 2000, 1200, 2500, 1800],
    'lot_size': [5000, 6000, 4000, 8000, 5500],
    'year_built': [1990, 2005, 1985, 2010, 1995],
    'price': [250000, 350000, 180000, 450000, 290000]
})

# Create engineered features
houses['sqft_per_bedroom'] = houses['sqft'] / houses['bedrooms']
houses['bathroom_bedroom_ratio'] = houses['bathrooms'] / houses['bedrooms']
houses['age'] = 2025 - houses['year_built']
houses['total_rooms'] = houses['bedrooms'] + houses['bathrooms']
houses['sqft_per_lot'] = houses['sqft'] / houses['lot_size']

print("Original features:")
print(houses[['bedrooms', 'sqft', 'price']].head())
print("\nEngineered features:")
print(houses[['sqft_per_bedroom', 'age', 'sqft_per_lot']].head())
```

These engineered features capture relationships that might be more predictive than raw features alone.

### Polynomial Features

For linear models, polynomial features create nonlinear relationships:

```python
from sklearn.preprocessing import PolynomialFeatures

# Original features
X = np.array([[2, 3], [3, 4], [4, 5]])

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features:", X.shape)
print(X)
print("\nPolynomial features (degree 2):", X_poly.shape)
print(X_poly)
print("\nFeature names:")
print(poly.get_feature_names_out(['x1', 'x2']))
```

For input features $[x_1, x_2]$, degree-2 polynomial features include $[x_1, x_2, x_1^2, x_1 x_2, x_2^2]$, enabling linear models to learn nonlinear decision boundaries.

### Interaction Features

Interaction features capture relationships between pairs (or higher-order combinations) of features:

```python
# Create interaction features
houses['sqft_age_interaction'] = houses['sqft'] * houses['age']
houses['bedrooms_sqft_interaction'] = houses['bedrooms'] * houses['sqft']

print("Interaction features:")
print(houses[['sqft', 'age', 'sqft_age_interaction']].head())
```

These might reveal that the relationship between square footage and price depends on the age of the house.

### Binning/Discretization

Converting continuous features to categorical bins can help models learn non-smooth relationships:

```python
# Create age bins
houses['age_category'] = pd.cut(houses['age'],
                                 bins=[0, 10, 20, 50],
                                 labels=['New', 'Recent', 'Old'])

print("Age binning:")
print(houses[['age', 'age_category']].head())

# One-hot encode the bins
houses_binned = pd.get_dummies(houses, columns=['age_category'], prefix='age')
print("\nOne-hot encoded age bins:")
print(houses_binned.filter(like='age_').head())
```

### Log Transformations

For skewed distributions, log transformations can make relationships more linear:

```python
# Skewed price distribution
print("Original price statistics:")
print(houses['price'].describe())

# Log transformation
houses['log_price'] = np.log(houses['price'])

print("\nLog-transformed price statistics:")
print(houses['log_price'].describe())

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(houses['price'], bins=10, edgecolor='black')
axes[0].set_title('Original Price Distribution')
axes[0].set_xlabel('Price')

axes[1].hist(houses['log_price'], bins=10, edgecolor='black')
axes[1].set_title('Log-Transformed Price Distribution')
axes[1].set_xlabel('Log(Price)')

plt.tight_layout()
plt.show()
```

Log transformations are particularly useful for features spanning multiple orders of magnitude (income, population, counts).

## Feature Selection

With many features, models can overfit, training becomes slow, and interpretation becomes difficult. **Feature selection** identifies the most relevant features, removing redundant or irrelevant ones.

### Why Feature Selection Matters

1. **Reduces overfitting**: Fewer features mean less opportunity to memorize noise
2. **Improves performance**: Removing irrelevant features helps models focus on signal
3. **Speeds up training**: Fewer features mean faster computation
4. **Enhances interpretability**: Simpler models with fewer features are easier to understand

### Filter Methods

Filter methods select features based on statistical properties, independent of the learning algorithm:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print("Original features:", X.shape)

# Select top 2 features using ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print("Selected features:", X_selected.shape)

# Show which features were selected
selected_indices = selector.get_support(indices=True)
selected_names = [iris.feature_names[i] for i in selected_indices]
print("Selected feature names:", selected_names)

# Show feature scores
scores = selector.scores_
for name, score in zip(iris.feature_names, scores):
    print(f"  {name}: {score:.2f}")
```

### Wrapper Methods

Wrapper methods use the model itself to evaluate feature subsets:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)

print("RFE selected features:")
selected_features = [name for name, selected in zip(iris.feature_names, rfe.support_) if selected]
print(selected_features)

# Feature ranking (1 = selected, 2+ = eliminated in order)
print("\nFeature rankings:")
for name, rank in zip(iris.feature_names, rfe.ranking_):
    print(f"  {name}: rank {rank}")
```

### Embedded Methods

Regularization-based models perform feature selection as part of training:

```python
from sklearn.linear_model import LassoCV

# Lasso for feature selection (L1 drives coefficients to zero)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

print("Lasso coefficients:")
for name, coef in zip(iris.feature_names, lasso.coef_):
    print(f"  {name}: {coef:.3f}")

# Select features with non-zero coefficients
selected_features = [name for name, coef in zip(iris.feature_names, lasso.coef_) if abs(coef) > 0.01]
print("\nSelected features (|coef| > 0.01):")
print(selected_features)
```

## Dimensionality Reduction

**Dimensionality reduction** transforms high-dimensional data into lower-dimensional representations while preserving important information. Unlike feature selection (which discards features), dimensionality reduction creates new features as combinations of original ones.

### Principal Component Analysis (PCA)

PCA finds orthogonal directions (principal components) that capture maximum variance:

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original data shape:", X.shape)
print("PCA-reduced data shape:", X_pca.shape)

# Explained variance
print("\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.3f}")

# Visualize in 2D
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA: Iris Dataset Projected to 2D')
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)
plt.show()
```

PCA is particularly useful for:
- Visualization (reducing to 2-3 dimensions)
- Speeding up training (fewer dimensions)
- Removing multicollinearity (PCs are orthogonal)
- Noise reduction (discarding low-variance components)

### Choosing the Number of Components

```python
# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X)

# Plot explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Explained Variance vs Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")
```

## Data Augmentation

**Data augmentation** artificially expands training data by creating modified versions of existing examples. This is particularly important for deep learning, where large datasets are crucial.

### Image Augmentation

For image data, common augmentations include:

```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Load sample digit
digits = load_digits()
sample_image = digits.images[0]

# Define augmentation functions
def rotate(image, angle):
    from scipy.ndimage import rotate as scipy_rotate
    return scipy_rotate(image, angle, reshape=False)

def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 16)

def shift(image, dx, dy):
    from scipy.ndimage import shift as scipy_shift
    return scipy_shift(image, [dy, dx], mode='constant', cval=0)

# Create augmented versions
augmented = [
    ("Original", sample_image),
    ("Rotated 15°", rotate(sample_image, 15)),
    ("Noisy", add_noise(sample_image)),
    ("Shifted", shift(sample_image, 2, -1))
]

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, (title, img) in zip(axes, augmented):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Common image augmentations:
- **Geometric**: Rotation, flipping, scaling, translation, shearing
- **Color**: Brightness, contrast, saturation adjustments
- **Noise**: Gaussian noise, dropout
- **Cropping**: Random crops, center crops

### Text Augmentation

For text data:
- **Synonym replacement**: Replace words with synonyms
- **Random insertion/deletion**: Add or remove words
- **Back-translation**: Translate to another language and back
- **Paraphrasing**: Generate paraphrases using language models

### Benefits of Augmentation

1. **Increases effective dataset size** without collecting new data
2. **Improves generalization** by exposing models to variations
3. **Reduces overfitting** by regularizing the learning process
4. **Makes models robust** to real-world variations

!!! warning "Augmentation Guidelines"
    - Apply augmentations that reflect real-world variations
    - Don't augment in ways that change the label (e.g., rotating a "6" to look like a "9")
    - Augment training data only, not validation/test data
    - Be mindful of computational cost (augment on-the-fly during training)

## Computational Complexity Considerations

Preprocessing choices affect computational and space requirements. Understanding complexity helps build scalable pipelines.

### Time Complexity

**Time complexity** describes how processing time grows with data size $n$ or feature count $d$:

- **Standardization**: $O(nd)$ - one pass to compute statistics, one to transform
- **One-hot encoding**: $O(nd)$ - scan data to find categories, create columns
- **Polynomial features (degree $p$)**: $O(nd^p)$ - exponential in degree
- **PCA**: $O(nd^2 + d^3)$ - covariance matrix computation and eigendecomposition
- **Feature selection (filter methods)**: $O(nd)$ - compute statistics per feature
- **Feature selection (wrapper methods)**: $O(nd \cdot m)$ - $m$ model training iterations

### Space Complexity

**Space complexity** describes memory requirements:

- **Original data**: $O(nd)$
- **Standardization**: $O(d)$ - store mean and std for each feature
- **One-hot encoding**: $O(nk)$ where $k$ is total number of categories across all features
- **Polynomial features (degree $p$)**: $O(n \cdot d^p)$ - exponential growth
- **PCA**: $O(nd + d^2)$ - store transformed data and transformation matrix

### Scalability

**Scalability** refers to how well preprocessing handles increasing data size:

```python
import time

# Compare scaling approaches on different data sizes
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    X_large = np.random.randn(size, 10)

    start = time.time()
    scaler = StandardScaler()
    scaler.fit_transform(X_large)
    elapsed = time.time() - start

    times.append(elapsed)
    print(f"Size {size:7d}: {elapsed:.4f} seconds")

# Plot scaling
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'bo-')
plt.xlabel('Dataset Size')
plt.ylabel('Time (seconds)')
plt.title('Standardization: Time vs Dataset Size')
plt.grid(True, alpha=0.3)
plt.show()
```

For large datasets, consider:
- **Incremental processing**: Process data in batches
- **Distributed computing**: Use tools like Spark or Dask
- **Approximate methods**: Trade accuracy for speed (e.g., random projections instead of PCA)

## Complete Preprocessing Pipeline

Combining multiple preprocessing steps into a pipeline ensures consistency and prevents data leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Sample data with mixed types
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 35],
    'income': [50000, 60000, 55000, 80000, np.nan],
    'education': ['BS', 'MS', 'BS', 'PhD', 'MS'],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA']
})

# Define preprocessing for different column types
numeric_features = ['age', 'income']
categorical_features = ['education', 'city']

# Numeric pipeline: impute missing values, then standardize
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', pd.get_dummies)
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform
data_preprocessed = preprocessor.fit_transform(data)

print("Original data:")
print(data)
print("\nPreprocessed data shape:", data_preprocessed.shape)
```

Pipelines ensure that:
- All preprocessing steps are applied consistently
- Transformations fitted on training data are applied to test data
- No data leakage occurs between training and testing

## Interactive Visualization: Feature Scaling Comparison

Explore how min-max scaling and z-score standardization transform data differently:

<iframe src="../../sims/feature-scaling-visualizer/feature-scaling-visualizer.html" width="100%" height="750" frameborder="0"></iframe>

## Interactive Visualization: One-Hot Encoding Explorer

Compare how label encoding and one-hot encoding transform categorical variables:

<iframe src="../../sims/categorical-encoding-explorer/categorical-encoding-explorer.html" width="100%" height="850" frameborder="0"></iframe>

## Summary

Data preprocessing transforms raw data into representations suitable for machine learning algorithms. Effective preprocessing is often more impactful than algorithm choice or hyperparameter tuning.

**Normalization** and **standardization** scale features to comparable ranges. **Min-max scaling** transforms to [0, 1], while **z-score normalization** creates distributions with mean 0 and standard deviation 1. Always fit scalers on training data only to avoid data leakage.

**Label encoding** assigns integer labels to categories, suitable for ordinal variables or tree-based algorithms. **One-hot encoding** creates binary columns for each category, necessary for nominal variables in most algorithms. The `drop_first` parameter prevents multicollinearity by removing one redundant column per feature.

**Feature engineering** creates informative features through domain knowledge, polynomial transformations, interactions, binning, and mathematical transformations like logarithms. **Feature selection** identifies relevant features using filter, wrapper, or embedded methods, reducing overfitting and improving interpretability.

**Dimensionality reduction** transforms high-dimensional data to lower dimensions while preserving information. PCA finds directions of maximum variance, enabling visualization and noise reduction. **Data augmentation** artificially expands datasets by creating modified versions of existing examples, particularly valuable for deep learning.

**Computational complexity** considerations guide scalable preprocessing. Time complexity ranges from linear ($O(nd)$ for standardization) to polynomial ($O(nd^p)$ for degree-$p$ polynomial features) to exponential (PCA's $O(nd^2 + d^3)$). Space complexity grows with features and transformations, requiring careful memory management for large datasets.

Building preprocessing pipelines ensures consistent transformations across training and test data, preventing data leakage and simplifying deployment.

## Key Takeaways

1. **Data preprocessing** transforms raw data into representations suitable for machine learning
2. **Min-max scaling** normalizes features to [0, 1]; **z-score normalization** standardizes to mean=0, std=1
3. **Feature scaling** is essential for distance-based and gradient-based algorithms
4. **Label encoding** assigns integer labels; **one-hot encoding** creates binary columns per category
5. **One-hot encoding** is preferred for nominal variables in most algorithms
6. **Feature engineering** creates new features from existing ones using domain knowledge
7. **Feature selection** reduces dimensionality by identifying relevant features
8. **Dimensionality reduction** (PCA) transforms to lower dimensions while preserving information
9. **Data augmentation** expands datasets by creating modified examples
10. Always fit preprocessing on training data and apply to test data to prevent leakage
11. **Time complexity** and **space complexity** affect scalability of preprocessing methods
12. Preprocessing pipelines ensure consistency and prevent data leakage

## Further Reading

- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Chapter 2: End-to-End Machine Learning Project)
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling* (Chapter 3: Data Pre-Processing)
- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*
- Scikit-learn documentation: [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- Scikit-learn documentation: [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

## Exercises

1. **Scaling Comparison**: Create a synthetic dataset with features on vastly different scales (e.g., 0-1, 0-1000, 0-1000000). Train a k-NN classifier with and without standardization. Compare accuracy and decision boundaries.

2. **Encoding Impact**: Take a dataset with categorical variables (e.g., mushroom classification). Compare model performance using label encoding vs one-hot encoding. Which encoding works better for logistic regression? For random forest?

3. **Feature Engineering**: Create polynomial features up to degree 5 for a simple regression problem. Plot training and test error vs polynomial degree to observe the bias-variance trade-off.

4. **Dimensionality Curse**: Generate high-dimensional random data (1000 features) where only 10 features are predictive. Apply different feature selection methods and compare which ones successfully identify the true features.

5. **PCA Visualization**: Apply PCA to the MNIST digit dataset. Visualize digits in 2D PCA space. How many components are needed to retain 95% of variance? Reconstruct images using different numbers of components.

6. **Preprocessing Pipeline**: Build a complete preprocessing pipeline for a real-world dataset with missing values, mixed feature types, and different scales. Use cross-validation to evaluate a model with and without preprocessing.

7. **Computational Scaling**: Implement standardization from scratch and compare its runtime to scikit-learn's implementation on datasets of increasing size. Plot time vs dataset size to verify $O(nd)$ complexity.
