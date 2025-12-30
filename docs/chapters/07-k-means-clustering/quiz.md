# Quiz: K-Means Clustering

Test your understanding of k-means clustering with these questions.

---

#### 1. What is the primary objective function that k-means clustering aims to minimize?

<div class="upper-alpha" markdown>
1. The sum of distances between all pairs of data points
2. The sum of squared distances from each point to its assigned cluster centroid
3. The maximum distance between any point and its cluster centroid
4. The number of clusters needed to separate the data
</div>

??? question "Show Answer"
    The correct answer is **B**.

    K-means clustering minimizes the within-cluster variance (also called inertia), which is mathematically defined as the sum of squared Euclidean distances from each point to its assigned cluster centroid: $J = \sum_{i=1}^{n} \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\|^2$. This objective function measures how tightly grouped the clusters are—smaller values indicate more compact, well-separated clusters. The algorithm iteratively refines cluster assignments and centroids to reduce this value until convergence.

    **Concept Tested:** K-Means Clustering, Within-Cluster Variance, Inertia

---

#### 2. In the k-means algorithm, what happens during the cluster assignment step?

<div class="upper-alpha" markdown>
1. Each centroid is recomputed as the mean of all points assigned to that cluster
2. Each data point is assigned to the cluster with the nearest centroid
3. The algorithm checks if convergence criteria have been met
4. New random centroids are initialized to avoid local optima
</div>

??? question "Show Answer"
    The correct answer is **B**.

    The k-means algorithm alternates between two steps. During the cluster assignment step, each data point $\mathbf{x}_i$ is assigned to the cluster whose centroid is closest, mathematically expressed as $c_i \leftarrow \arg\min_{j \in \{1,\ldots,k\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$. This step uses distance calculations to determine which cluster "claims" each point. The cluster update step (option A) comes after assignment, while convergence checking (option C) happens after both steps are complete.

    **Concept Tested:** Cluster Assignment, K-Means Clustering

---

#### 3. Which initialization method chooses subsequent centroids with probability proportional to their squared distance from already-chosen centroids?

<div class="upper-alpha" markdown>
1. Random initialization
2. K-means++ initialization
3. Hierarchical initialization
4. Grid-based initialization
</div>

??? question "Show Answer"
    The correct answer is **B**.

    K-means++ initialization improves upon random initialization by spreading initial centroids far apart. After choosing the first centroid randomly, each subsequent centroid is selected from the remaining data points with probability proportional to $D(\mathbf{x}_i)^2$, where $D(\mathbf{x}_i)$ is the distance to the nearest already-chosen centroid. This makes distant points more likely to be selected as centroids, leading to better initial coverage of the data space and more consistent final results.

    **Concept Tested:** K-Means++ Initialization, K-Means Initialization

---

#### 4. A data scientist applies k-means clustering to customer purchase data without standardizing the features. One feature is 'annual spending' (range: $100-$50,000) and another is 'number of purchases' (range: 1-100). What problem is likely to occur?

<div class="upper-alpha" markdown>
1. The algorithm will fail to converge
2. The distance calculations will be dominated by the annual spending feature
3. The number of clusters will automatically increase
4. The centroids will all converge to the same location
</div>

??? question "Show Answer"
    The correct answer is **B**.

    K-means uses Euclidean distance to assign points to clusters. When features have vastly different scales, the feature with the larger range dominates the distance calculation. In this case, a $1,000 difference in annual spending contributes far more to the distance than a difference of 10 purchases, even though purchases might be equally informative. This causes the clustering to essentially ignore the smaller-scale feature. The solution is to standardize all features before applying k-means.

    **Concept Tested:** K-Means Clustering, Data Preprocessing, Feature Scaling

---

#### 5. What does the 'elbow' in the elbow method represent when choosing the number of clusters?

<div class="upper-alpha" markdown>
1. The point where the data becomes linearly separable
2. The maximum possible number of clusters for the dataset
3. The point where adding more clusters provides diminishing returns in reducing inertia
4. The number of dimensions in the original feature space
</div>

??? question "Show Answer"
    The correct answer is **C**.

    The elbow method plots inertia (within-cluster sum of squares) as a function of $k$. Inertia always decreases as $k$ increases, but the rate of decrease slows down. The "elbow" is the point where the curve bends sharply—beyond this point, additional clusters provide only marginal improvement in fit quality. This represents a good trade-off between model complexity (number of clusters) and clustering quality. However, the elbow isn't always clearly defined, which is why silhouette scores provide a complementary evaluation method.

    **Concept Tested:** Elbow Method, Inertia, K-Means Clustering

---

#### 6. For a single data point, a silhouette coefficient close to -1 indicates that:

<div class="upper-alpha" markdown>
1. The point is perfectly clustered at the center of its cluster
2. The point is on the boundary between two clusters
3. The point might be assigned to the wrong cluster
4. The point is an outlier far from all clusters
</div>

??? question "Show Answer"
    The correct answer is **C**.

    The silhouette coefficient for point $i$ is calculated as $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, where $a(i)$ is the average distance to other points in the same cluster and $b(i)$ is the average distance to points in the nearest other cluster. A value close to -1 means $a(i) > b(i)$—the point is closer on average to points in a different cluster than to points in its own cluster, suggesting misclassification. Values near +1 indicate good clustering, while values near 0 indicate borderline cases.

    **Concept Tested:** Silhouette Score, K-Means Clustering

---

#### 7. Which of the following is NOT a convergence criterion commonly used in k-means clustering?

<div class="upper-alpha" markdown>
1. Cluster assignments don't change between iterations
2. Centroids move less than a threshold distance
3. The silhouette score reaches a maximum value
4. Maximum number of iterations is reached
</div>

??? question "Show Answer"
    The correct answer is **C**.

    K-means convergence criteria focus on detecting when the algorithm has stabilized: no reassignments of points between clusters (option A), minimal centroid movement (option B), minimal change in the objective function (inertia), or reaching a maximum iteration limit (option D). The silhouette score is an external evaluation metric used to assess clustering quality and choose $k$, but it is not computed during the iterative k-means process and therefore cannot be used as a convergence criterion.

    **Concept Tested:** Convergence Criteria, K-Means Clustering

---

#### 8. A researcher runs k-means with random initialization 10 times and gets 10 different final inertia values ranging from 145.2 to 289.7. What does this variability indicate?

<div class="upper-alpha" markdown>
1. The data is not suitable for clustering
2. The algorithm is converging to different local optima
3. The number of clusters $k$ is too large
4. The features need to be standardized
</div>

??? question "Show Answer"
    The correct answer is **B**.

    K-means is not guaranteed to find the global minimum of the objective function—it can get stuck in local optima depending on the initial centroid positions. The large range in final inertia values (145.2 to 289.7) indicates that different random initializations are leading to significantly different solutions, some much better than others. This is exactly why k-means++ initialization was developed—it provides more consistent results by carefully seeding initial centroids. Best practice is to run k-means multiple times and select the solution with lowest inertia.

    **Concept Tested:** Random Initialization, K-Means Clustering, Inertia

---

#### 9. Which limitation of k-means clustering is demonstrated when the algorithm fails to correctly identify two concentric circular clusters?

<div class="upper-alpha" markdown>
1. Sensitivity to outliers
2. Need to specify $k$ in advance
3. Assumption of spherical clusters
4. Sensitivity to feature scales
</div>

??? question "Show Answer"
    The correct answer is **C**.

    K-means uses Euclidean distance and assigns points to the nearest centroid, which implicitly assumes clusters are roughly spherical (or at least convex) and of similar size. Concentric circles represent a non-spherical, nested cluster structure that violates this assumption. K-means would likely split the rings inappropriately or merge them incorrectly because it cannot capture the circular geometry. Other algorithms like DBSCAN or spectral clustering can handle such non-spherical cluster shapes.

    **Concept Tested:** K-Means Clustering, Cluster Structure Assumptions

---

#### 10. Given a dataset with 500 samples, 10 features, and $k=4$ clusters, approximately how many parameters (centroid coordinates) does the k-means model store?

<div class="upper-alpha" markdown>
1. 10 parameters
2. 40 parameters
3. 500 parameters
4. 2000 parameters
</div>

??? question "Show Answer"
    The correct answer is **B**.

    K-means stores $k$ centroids, where each centroid is a point in the $d$-dimensional feature space. With $k=4$ clusters and $d=10$ features, the model stores $4 \times 10 = 40$ centroid coordinates. These centroids fully define the k-means model—to predict which cluster a new point belongs to, we only need the centroids to compute distances. The number of samples (500) doesn't affect the model size, making k-means very memory-efficient compared to instance-based methods like k-NN.

    **Concept Tested:** K-Means Clustering, Centroid, Model Complexity
