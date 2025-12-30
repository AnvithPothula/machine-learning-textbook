# Quiz: K-Nearest Neighbors Algorithm

Test your understanding of the K-Nearest Neighbors algorithm with these questions.

---

#### 1. What does the "k" in K-Nearest Neighbors represent?

<div class="upper-alpha" markdown>
1. The number of features in the dataset
2. The number of nearest training examples used for prediction
3. The distance threshold for neighbors
4. The number of classes in classification
</div>

??? question "Show Answer"
    The correct answer is **B**. The "k" in K-Nearest Neighbors represents the number of nearest training examples (neighbors) used to make a prediction. For example, in 5-NN, the algorithm finds the 5 closest training examples and uses their labels to predict the class (via majority vote) or value (via average).

    **Concept Tested:** K-Nearest Neighbors

---

#### 2. What is the primary reason KNN is called a "lazy learning" algorithm?

<div class="upper-alpha" markdown>
1. It takes a long time to make predictions
2. It stores all training data and defers computation until prediction time
3. It requires minimal memory
4. It only works with small datasets
</div>

??? question "Show Answer"
    The correct answer is **B**. KNN is called "lazy learning" because it doesn't build an explicit model during training—it simply stores all training examples. All computation is deferred until prediction time, when it must calculate distances to all training points. This contrasts with "eager" learners that build models during training.

    **Concept Tested:** Lazy Learning

---

#### 3. Given a query point in 2D space at (3, 4) and a training point at (6, 8), what is the Euclidean distance between them?

<div class="upper-alpha" markdown>
1. 3.0
2. 4.0
3. 5.0
4. 7.0
</div>

??? question "Show Answer"
    The correct answer is **C**. The Euclidean distance is calculated as sqrt((6-3)² + (8-4)²) = sqrt(9 + 16) = sqrt(25) = 5.0. Euclidean distance is the straight-line distance between two points and is the most common distance metric used in KNN.

    **Concept Tested:** Euclidean Distance

---

#### 4. How does Manhattan distance differ from Euclidean distance?

<div class="upper-alpha" markdown>
1. Manhattan distance is always larger than Euclidean distance
2. Manhattan distance sums absolute differences while Euclidean distance uses squared differences
3. Manhattan distance only works in 2D space
4. Manhattan distance requires normalized features
</div>

??? question "Show Answer"
    The correct answer is **B**. Manhattan distance (L1) sums the absolute differences across all dimensions: |x₁-y₁| + |x₂-y₂| + ..., while Euclidean distance (L2) uses squared differences under a square root: sqrt((x₁-y₁)² + (x₂-y₂)² + ...). Manhattan distance represents the distance if you could only travel along grid lines, like navigating city blocks.

    **Concept Tested:** Manhattan Distance

---

#### 5. What happens when k=1 in K-Nearest Neighbors classification?

<div class="upper-alpha" markdown>
1. The algorithm predicts the most common class in the entire dataset
2. The prediction is based solely on the single nearest training example
3. The algorithm cannot make predictions
4. All training examples contribute equally to the prediction
</div>

??? question "Show Answer"
    The correct answer is **B**. When k=1, the algorithm finds the single nearest training example and assigns its label to the query point. This makes the model highly sensitive to noise and outliers, as each prediction is based on just one neighbor, often leading to overfitting with complex, irregular decision boundaries.

    **Concept Tested:** K Selection

---

#### 6. Why does KNN performance typically degrade in high-dimensional spaces?

<div class="upper-alpha" markdown>
1. Computers cannot process many dimensions
2. The curse of dimensionality makes distances less meaningful as dimensionality increases
3. KNN can only use up to 10 dimensions
4. High dimensions require more neighbors
</div>

??? question "Show Answer"
    The correct answer is **B**. In high-dimensional spaces, the curse of dimensionality causes all points to become approximately equidistant from each other. Data becomes increasingly sparse, distances lose their discriminative power, and the nearest and farthest neighbors become nearly the same distance away, making KNN's distance-based predictions unreliable.

    **Concept Tested:** Curse of Dimensionality

---

#### 7. For a KNN regression problem with k=5 and neighbor values [10, 12, 11, 13, 14], what would be the predicted value?

<div class="upper-alpha" markdown>
1. 10
2. 12
3. 13
4. 14
</div>

??? question "Show Answer"
    The correct answer is **B**. For KNN regression, the predicted value is the average of the k nearest neighbors' values: (10 + 12 + 11 + 13 + 14) / 5 = 60 / 5 = 12. Unlike classification which uses majority voting, regression predicts the mean (or sometimes median) of neighbor values.

    **Concept Tested:** KNN for Regression

---

#### 8. What is the main computational bottleneck of KNN during prediction time?

<div class="upper-alpha" markdown>
1. Storing the training data
2. Computing distances to all training examples
3. Sorting the class labels
4. Calculating the majority vote
</div>

??? question "Show Answer"
    The correct answer is **B**. During prediction, KNN must compute the distance from the query point to every training example, which has O(n) complexity where n is the number of training examples. This becomes expensive for large datasets. Data structures like k-d trees or ball trees can reduce this to O(log n) in lower dimensions.

    **Concept Tested:** K-Nearest Neighbors (computational complexity)

---

#### 9. In a binary classification problem, why might choosing an even value for k be problematic?

<div class="upper-alpha" markdown>
1. Even values are computationally more expensive
2. It can lead to ties in majority voting
3. Even values always cause overfitting
4. The algorithm only works with odd k values
</div>

??? question "Show Answer"
    The correct answer is **B**. With even k values in binary classification, it's possible to get a tie (e.g., k=4 with 2 votes for each class). While this can be resolved with strategies like choosing the label of the nearest neighbor or random selection, odd k values naturally avoid ties and are generally preferred for binary classification.

    **Concept Tested:** K Selection

---

#### 10. What is a Voronoi diagram in the context of KNN?

<div class="upper-alpha" markdown>
1. A visualization showing decision boundaries when k=1
2. A graph showing the relationship between k and accuracy
3. A plot of training data points in feature space
4. A diagram showing the distance between all pairs of points
</div>

??? question "Show Answer"
    The correct answer is **A**. A Voronoi diagram partitions the feature space into regions where each region contains all points closest to a particular training example. For 1-NN classification, the Voronoi diagram exactly represents the decision boundaries, as any point in a region is classified with the label of the training point in that region.

    **Concept Tested:** Voronoi Diagram
