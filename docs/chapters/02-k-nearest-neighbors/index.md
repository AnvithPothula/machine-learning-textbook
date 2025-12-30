# K-Nearest Neighbors Algorithm

---
title: K-Nearest Neighbors Algorithm
description: Introduction to KNN for classification and regression, distance metrics, k selection, decision boundaries, and the curse of dimensionality
generated_by: claude skill chapter-content-generator
date: 2025-12-28
version: 0.03
---

## Summary

This chapter introduces the K-Nearest Neighbors (KNN) algorithm, one of the most intuitive machine learning algorithms that serves as an excellent starting point for understanding classification and regression. Students will explore how KNN makes predictions by finding similar examples in the training data, learn about different distance metrics (Euclidean and Manhattan), and understand the importance of selecting an appropriate value of k. The chapter covers the geometric interpretation of decision boundaries and Voronoi diagrams, addresses the curse of dimensionality, and demonstrates how KNN operates as a lazy learning algorithm that requires no explicit training phase.

## Concepts Covered

This chapter covers the following 11 concepts from the learning graph:

1. K-Nearest Neighbors
2. Distance Metric
3. Euclidean Distance
4. Manhattan Distance
5. K Selection
6. Decision Boundary
7. Voronoi Diagram
8. Curse of Dimensionality
9. KNN for Classification
10. KNN for Regression
11. Lazy Learning

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Machine Learning Fundamentals](../01-intro-to-ml-fundamentals/index.md)

---

## The Intuition Behind K-Nearest Neighbors

Imagine you've just moved to a new city and want to find a good restaurant. You ask your five nearest neighbors for recommendations, and four of them suggest Italian restaurants while one suggests Chinese food. Using the "majority vote" principle, you'd choose an Italian restaurant. This is precisely how the K-Nearest Neighbors algorithm works—it makes predictions based on the most common outcome among the k closest training examples.

K-Nearest Neighbors (KNN) is an instance-based learning algorithm that classifies new data points by examining the k training instances that are most similar (nearest) to the query point. Unlike algorithms that build an explicit model during training, KNN simply stores all training examples and defers computation until prediction time. This makes KNN remarkably simple yet surprisingly effective for many real-world problems.

The algorithm's elegance lies in its core principle: **similar inputs should produce similar outputs**. If you want to predict whether a flower is an Iris setosa, versicolor, or virginica based on its petal and sepal measurements, KNN finds the k flowers in your training data with the most similar measurements and assigns the most common species among those neighbors.

### How KNN Works: A Step-by-Step Example

Let's walk through a concrete classification example using the classic Iris dataset, which contains measurements of 150 iris flowers across three species.

**Training Phase:**
1. Store all training examples (no actual "learning" happens—this is why KNN is called a "lazy" algorithm)

**Prediction Phase:**
1. Given a new flower to classify, calculate its distance to all training examples
2. Identify the k nearest neighbors based on these distances
3. For classification: Count the class labels among these k neighbors
4. Predict the majority class (for regression, predict the average value)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=setosa, 1=versicolor, 2=virginica

# Create a DataFrame for easier exploration
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("Iris Dataset Shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nClass distribution:")
print(iris_df['species'].value_counts())
```

This dataset provides an ideal testbed for KNN because the three species form distinct clusters in feature space, as we'll see when we visualize the data.

```python
# Visualize the data with pairplot
plt.figure(figsize=(12, 10))
sns.pairplot(iris_df, vars=iris_df.columns[:-1], hue="species",
             markers=["o", "s", "D"], palette="Set2")
plt.suptitle("Iris Dataset: Feature Relationships by Species", y=1.02)
plt.show()
```

!!! note "Interpreting Pairplots"
    Each subplot shows the relationship between two features, with points colored by species. Notice how the three species form separable clusters, particularly when plotting petal length vs petal width. This visual separation suggests KNN will perform well on this dataset.

## Distance Metrics: Measuring Similarity

The foundation of KNN is measuring how "close" or "similar" two data points are. This requires defining a distance metric—a mathematical function that quantifies the dissimilarity between feature vectors. The choice of distance metric profoundly affects KNN's performance, as it determines which neighbors are considered "nearest."

### Euclidean Distance

The most common distance metric is **Euclidean distance**, which measures the straight-line distance between two points in feature space. For two points $\mathbf{x} = (x_1, x_2, ..., x_n)$ and $\mathbf{y} = (y_1, y_2, ..., y_n)$, Euclidean distance is:

$$d_{Euclidean}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

This corresponds to our everyday notion of distance—imagine drawing a straight line between two points and measuring its length. For a 2D example with points (1, 2) and (4, 6):

$$d = \sqrt{(4-1)^2 + (6-2)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

Euclidean distance works well when all features are on similar scales and continuous. However, it can be dominated by features with large numerical ranges, making feature scaling essential (we'll address this shortly).

### Manhattan Distance

**Manhattan distance** (also called taxicab or L1 distance) measures the distance traveled along axis-aligned paths, like navigating city blocks in Manhattan where you can only move horizontally or vertically. The formula is:

$$d_{Manhattan}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$$

For the same example points (1, 2) and (4, 6):

$$d = |4-1| + |6-2| = 3 + 4 = 7$$

Manhattan distance can be more robust to outliers than Euclidean distance because it doesn't square the differences. It's particularly useful when features represent independent dimensions (like grid coordinates) rather than components of a unified measurement.

```python
# Demonstrate distance calculations
point1 = np.array([1, 2])
point2 = np.array([4, 6])

# Euclidean distance
euclidean = np.sqrt(np.sum((point2 - point1)**2))
print(f"Euclidean distance: {euclidean:.3f}")

# Manhattan distance
manhattan = np.sum(np.abs(point2 - point1))
print(f"Manhattan distance: {manhattan:.3f}")

# Scikit-learn's KNN allows specifying the distance metric
from sklearn.neighbors import KNeighborsClassifier

# KNN with Euclidean distance (default)
knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# KNN with Manhattan distance
knn_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
```

The table below compares these distance metrics:

| Aspect | Euclidean Distance | Manhattan Distance |
|--------|-------------------|-------------------|
| **Formula** | $\sqrt{\sum (x_i - y_i)^2}$ | $\sum \|x_i - y_i\|$ |
| **Geometry** | Straight-line distance | Grid-path distance |
| **Sensitivity to outliers** | Higher (squares differences) | Lower (absolute differences) |
| **Best for** | Continuous features on similar scales | Independent dimensions, city-block problems |
| **Computational cost** | Moderate (square root) | Lower (just absolute values) |

#### Distance Metrics Visualization

<iframe src="../../sims/distance-metrics/main.html" width="100%" height="680px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](../../sims/distance-metrics/main.html){: target="_blank" .md-button } | [Documentation](../../sims/distance-metrics/index.md)

This interactive visualization compares Euclidean distance (green straight line) with Manhattan distance (orange grid path). Drag the blue point to see how both metrics change. Notice that Manhattan distance always equals or exceeds Euclidean distance, with the ratio reaching √2 when points align diagonally.

## Implementing KNN for Classification

Let's build a complete KNN classifier for the Iris dataset, following best practices for machine learning pipelines.

```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create and train KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on test set
y_pred = knn.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy with k=3: {accuracy:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix for KNN (k=3)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

The confusion matrix reveals which species are most easily confused. Typically, setosa is perfectly separated, while versicolor and virginica have some overlap in feature space.

!!! tip "Why Stratified Splitting?"
    When we use `stratify=y` in `train_test_split`, we ensure that both training and test sets maintain the same class proportions as the original dataset. For Iris, this means each set contains roughly equal numbers of all three species, preventing biased evaluation.

## K Selection: Choosing the Right Number of Neighbors

The value of k—the number of neighbors to consider—is KNN's most important hyperparameter. Choosing k involves a fundamental tradeoff between bias and variance:

- **Small k (e.g., k=1)**: Low bias, high variance
  - Decision boundary closely fits training data
  - Sensitive to noise and outliers
  - Risk of overfitting

- **Large k**: High bias, low variance
  - Smoother decision boundary
  - More robust to noise
  - Risk of underfitting by averaging over too many neighbors

### Finding the Optimal k

We can systematically search for the best k value using cross-validation, which provides more reliable performance estimates than a single train-test split.

```python
from sklearn.model_selection import cross_val_score

# Test different k values
k_range = range(1, 50, 2)  # Test k = 1, 3, 5, 7, ..., 49
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 10-fold cross-validation
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find optimal k
optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best cross-validation accuracy: {max(cv_scores):.3f}")

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Performance vs k Value')
plt.axvline(x=optimal_k, color='red', linestyle='--',
            label=f'Optimal k={optimal_k}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

The accuracy typically increases initially as k grows (reducing overfitting to noise), reaches a peak, then declines as k becomes too large (the model becomes too simple). The optimal k often falls between 3 and 15 for many datasets, though this depends on dataset size and complexity.

Explore how different k values affect decision boundaries and predictions:

<iframe src="../../sims/k-selection-simulator/k-selection-simulator.html" width="100%" height="850" frameborder="0"></iframe>

## Decision Boundaries and Voronoi Diagrams

A **decision boundary** is the line (or surface in higher dimensions) that separates regions assigned to different classes. Understanding decision boundaries provides geometric intuition for how classification algorithms partition feature space.

For KNN, the decision boundary is determined by the distribution of training points and the value of k. When k=1, the decision boundary creates **Voronoi diagrams**—each training point has a cell consisting of all locations closer to it than to any other training point. Points within a Voronoi cell are classified according to that cell's training point.

As k increases, decision boundaries become smoother. Instead of sharp Voronoi cells, boundaries become influenced by groups of neighbors, creating more gradual transitions between classes. This smoothing reduces sensitivity to individual training points but may obscure genuine fine-grained patterns in the data.

```python
# Visualize decision boundaries for different k values
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

# Use only two features for 2D visualization
X_2d = iris.data[:, [2, 3]]  # Petal length and width
y_2d = iris.target

# Standardize features
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# Split data
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d_scaled, y_2d, test_size=0.2, random_state=42, stratify=y_2d
)

# Create mesh for plotting decision boundaries
h = 0.02  # Step size in mesh
x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot decision boundaries for k=1, 5, 15
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
k_values = [1, 5, 15]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_2d, y_train_2d)

    # Predict for each point in mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    axes[idx].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    axes[idx].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d,
                     cmap=cmap_bold, edgecolor='black', s=50)
    axes[idx].set_title(f'KNN Decision Boundary (k={k})')
    axes[idx].set_xlabel('Petal Length (scaled)')
    axes[idx].set_ylabel('Petal Width (scaled)')

plt.tight_layout()
plt.show()
```

Notice how k=1 creates jagged, complex boundaries that tightly wrap training points, while k=15 creates smooth, generalized boundaries. The optimal k typically produces boundaries that capture true patterns while ignoring noise.

## KNN for Regression

While we've focused on classification, KNN also performs regression by predicting the average (or weighted average) of the k nearest neighbors' target values instead of voting on classes.

For a regression problem, given a query point $\mathbf{x}$ and its k nearest neighbors $\mathcal{N}_k(\mathbf{x})$, the KNN regression prediction is:

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i$$

Optionally, we can weight neighbors by inverse distance, giving closer neighbors more influence:

$$\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}, \quad \text{where} \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing

# Load California housing dataset (regression task)
housing = fetch_california_housing()
X_housing = housing.data[:1000]  # Use subset for speed
y_housing = housing.target[:1000]

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# KNN regression with k=5
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_h, y_train_h)

# Evaluate with R² score (proportion of variance explained)
from sklearn.metrics import r2_score, mean_squared_error

y_pred_h = knn_reg.predict(X_test_h)
r2 = r2_score(y_test_h, y_pred_h)
mse = mean_squared_error(y_test_h, y_pred_h)

print(f"KNN Regression R² score: {r2:.3f}")
print(f"Mean Squared Error: {mse:.3f}")

# Compare actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_h, y_pred_h, alpha=0.6)
plt.plot([y_test_h.min(), y_test_h.max()],
         [y_test_h.min(), y_test_h.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('KNN Regression: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

KNN regression works well when the relationship between features and target is complex and nonlinear, as KNN makes no assumptions about the underlying functional form. However, it performs poorly when data is sparse or high-dimensional (as we'll discuss next).

## Lazy Learning: No Explicit Training Phase

KNN is a **lazy learning** (or instance-based) algorithm, meaning it defers all computation until prediction time rather than building a model during training. This contrasts with eager learning algorithms like decision trees or neural networks that construct explicit models from training data.

Characteristics of lazy learning in KNN:

- **Training phase**: Simply store all training examples (O(1) time complexity)
- **Prediction phase**: Compute distances to all training points and find k nearest neighbors (O(n) time complexity for n training examples)
- **Memory requirements**: Must store entire training dataset
- **Adaptability**: Easy to add new training data without retraining

This lazy approach has important implications:

**Advantages:**
- No training time—can immediately use new data
- Naturally handles complex decision boundaries
- No assumptions about data distribution
- Adapts locally to different regions of feature space

**Disadvantages:**
- Slow predictions (must compute distances to all training points)
- High memory requirements (stores all training data)
- Requires careful selection of k and distance metric
- Doesn't provide interpretable model or feature importance

```python
import time

# Measure prediction time for different training set sizes
sizes = [100, 500, 1000, 5000, 10000]
prediction_times = []

for size in sizes:
    # Create synthetic dataset of specified size
    X_synth = np.random.randn(size, 10)
    y_synth = (X_synth[:, 0] > 0).astype(int)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_synth, y_synth)

    # Time prediction for 100 test points
    X_test_synth = np.random.randn(100, 10)
    start = time.time()
    knn.predict(X_test_synth)
    elapsed = time.time() - start
    prediction_times.append(elapsed)

# Plot scaling behavior
plt.figure(figsize=(10, 6))
plt.plot(sizes, prediction_times, marker='o', linewidth=2)
plt.xlabel('Training Set Size')
plt.ylabel('Prediction Time (seconds)')
plt.title('KNN Prediction Time Scales Linearly with Training Set Size')
plt.grid(True, alpha=0.3)
plt.show()

print("Prediction time grows linearly with training data size.")
print("This is a fundamental limitation of lazy learning algorithms.")
```

For large datasets, consider using approximate nearest neighbor algorithms (like locality-sensitive hashing or KD-trees) to speed up predictions, though these trade exactness for efficiency.

## The Curse of Dimensionality

One of KNN's most significant limitations is the **curse of dimensionality**—as the number of features increases, the distance between all points becomes increasingly similar, making the notion of "nearest neighbors" less meaningful.

In high-dimensional spaces:

1. **Volume grows exponentially**: A unit hypercube in d dimensions has volume $1^d = 1$, but most of this volume concentrates near the surface. Points become sparse even with millions of examples.

2. **Distances become uniform**: The ratio of the farthest to nearest neighbor approaches 1 as dimensionality increases, making all points roughly equidistant.

3. **Concentration phenomenon**: The distance between a random point and its nearest neighbor grows as $\sqrt{d}$, where d is dimensionality.

Mathematically, for uniformly distributed random points in a d-dimensional unit hypercube, the expected distance to the nearest neighbor is approximately:

$$\mathbb{E}[\text{dist}_{nearest}] \approx \left(\frac{1}{n}\right)^{1/d}$$

This means we need exponentially more data to maintain the same density as dimensions increase. To keep the same neighbor density going from 2D to 10D requires roughly $n^{10/2} = n^5$ times as much data!

```python
# Demonstrate curse of dimensionality
from sklearn.metrics.pairwise import euclidean_distances

dimensions = [2, 5, 10, 20, 50, 100]
avg_nearest_dist = []
avg_farthest_dist = []

np.random.seed(42)

for d in dimensions:
    # Generate 1000 random points in d dimensions
    X = np.random.rand(1000, d)

    # Compute pairwise distances
    distances = euclidean_distances(X, X)

    # For each point, find nearest and farthest neighbor (excluding itself)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1).mean()
    np.fill_diagonal(distances, 0)
    farthest = distances.max(axis=1).mean()

    avg_nearest_dist.append(nearest)
    avg_farthest_dist.append(farthest)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(dimensions, avg_nearest_dist, marker='o', label='Avg Nearest Neighbor Dist', linewidth=2)
plt.plot(dimensions, avg_farthest_dist, marker='s', label='Avg Farthest Neighbor Dist', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Distance')
plt.title('Curse of Dimensionality: Distances Become Similar in High Dimensions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compute ratio of farthest to nearest
ratio = np.array(avg_farthest_dist) / np.array(avg_nearest_dist)
print("\nRatio of farthest to nearest neighbor distance:")
for d, r in zip(dimensions, ratio):
    print(f"  {d}D: {r:.2f}")
```

As dimensionality increases, the ratio approaches 1, meaning "nearest" and "farthest" neighbors become indistinguishable. This severely degrades KNN performance in high dimensions.

**Mitigation strategies:**

- **Feature selection**: Remove irrelevant features
- **Dimensionality reduction**: Use PCA or other techniques (Chapter 8)
- **Distance weighting**: Weight features by importance
- **Local distance metrics**: Use distance metrics that adapt to local data structure

!!! warning "High-Dimensional Data Warning"
    For datasets with >20 features, KNN often performs poorly unless combined with dimensionality reduction. Modern deep learning methods (Chapters 9-11) handle high-dimensional data more effectively by learning feature representations.

## Best Practices for Using KNN

To maximize KNN performance, follow these guidelines:

**1. Feature Scaling is Essential**

KNN uses distances, so features with large ranges dominate those with small ranges. Always standardize or normalize features:

```python
from sklearn.preprocessing import StandardScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

**2. Choose k Using Cross-Validation**

Never guess k—systematically search for the optimal value using cross-validation.

**3. Consider the Distance Metric**

Euclidean distance is default but not always best. Try Manhattan distance or others (Minkowski, cosine similarity) based on data characteristics.

**4. Handle Imbalanced Classes**

For imbalanced datasets, consider weighted KNN that gives more importance to closer neighbors.

**5. Reduce Dimensionality**

For high-dimensional data, apply PCA or feature selection before KNN.

**6. Use Approximate Methods for Large Datasets**

For very large datasets, use tree-based methods (Ball Tree, KD-Tree) or approximate nearest neighbor algorithms.

## Key Takeaways

This chapter explored the K-Nearest Neighbors algorithm, a simple yet powerful instance-based learning approach:

- **KNN makes predictions** by finding the k training examples most similar to a query point, then voting (classification) or averaging (regression) their labels

- **Distance metrics** like Euclidean and Manhattan distance quantify similarity; the choice of metric affects which neighbors are considered "nearest"

- **K selection** involves a bias-variance tradeoff: small k fits training data closely but is sensitive to noise, while large k creates smoother boundaries but may underfit

- **Decision boundaries** visualize how KNN partitions feature space; when k=1, these form Voronoi diagrams

- **Lazy learning** means KNN stores training data and defers all computation until prediction time, making training fast but predictions slow

- **The curse of dimensionality** causes KNN to fail in high-dimensional spaces where distances become meaningless and all points appear equidistant

- **Best practices** include feature scaling, cross-validation for k selection, and dimensionality reduction for high-dimensional data

KNN provides an excellent introduction to machine learning because it's intuitive, requires no training, and performs well on many real-world problems. However, its limitations—particularly computational cost and sensitivity to dimensionality—motivate the more sophisticated algorithms we'll explore in subsequent chapters, starting with Decision Trees in Chapter 3.