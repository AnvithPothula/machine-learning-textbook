# Quiz: Decision Trees

Test your understanding of decision tree algorithms with these questions.

---

#### 1. What is the primary advantage of decision trees compared to other machine learning algorithms?

<div class="upper-alpha" markdown>
1. They always achieve the highest accuracy
2. They require the least amount of training data
3. They are highly interpretable and easy to visualize
4. They train faster than any other algorithm
</div>

??? question "Show Answer"
    The correct answer is **C**. Decision trees are highly interpretable—you can follow the tree structure to understand exactly how predictions are made. Each path from root to leaf represents a clear decision rule. This interpretability makes them valuable when explanations are important, though they may not always achieve the highest accuracy compared to ensemble methods or neural networks.

    **Concept Tested:** Decision Tree

---

#### 2. What does entropy measure in the context of decision trees?

<div class="upper-alpha" markdown>
1. The impurity or disorder in a set of labels
2. The depth of the tree
3. The number of features in the dataset
4. The accuracy of predictions
</div>

??? question "Show Answer"
    The correct answer is **A**. Entropy measures the impurity or disorder in a set of labels. It's calculated as H = -Σ p_i log₂(p_i), where p_i is the proportion of class i. Entropy is 0 when all examples belong to one class (pure, perfectly ordered) and maximum when classes are evenly distributed (maximum disorder). Decision trees use entropy to select the best features for splitting.

    **Concept Tested:** Entropy

---

#### 3. How is information gain calculated when evaluating a potential split?

<div class="upper-alpha" markdown>
1. By counting the number of examples in each child node
2. By measuring the reduction in entropy before and after the split
3. By calculating the average depth of child nodes
4. By subtracting parent entropy from child entropy
</div>

??? question "Show Answer"
    The correct answer is **D**. Information gain measures the reduction in entropy achieved by splitting on a feature: IG = H(parent) - Σ (|child_i|/|parent|) × H(child_i). Decision trees greedily select the feature that maximizes information gain at each node, choosing splits that best separate classes and reduce uncertainty.

    **Concept Tested:** Information Gain

---

#### 4. What is the Gini impurity formula?

<div class="upper-alpha" markdown>
1. Gini = Σ p_i log(p_i)
2. Gini = 1 - Σ p_i²
3. Gini = -Σ p_i² log(p_i)
4. Gini = 1 + Σ p_i
</div>

??? question "Show Answer"
    The correct answer is **B**. Gini impurity is calculated as Gini = 1 - Σ p_i², where p_i is the proportion of class i. Like entropy, it measures node impurity, ranging from 0 (pure node, all one class) to 0.5 for binary classification (maximum impurity with 50/50 split). Gini is often preferred over entropy because it's computationally faster (no logarithm).

    **Concept Tested:** Gini Impurity

---

#### 5. What problem does pruning address in decision trees?

<div class="upper-alpha" markdown>
1. Overfitting by removing branches that don't improve validation performance
2. Slow training time by reducing the number of features
3. Underfitting by adding more tree depth
4. Memory usage by compressing tree nodes
</div>

??? question "Show Answer"
    The correct answer is **A**. Pruning addresses overfitting by removing branches that don't significantly improve performance on validation data. Unpruned trees can grow very deep, creating leaves for nearly every training example and memorizing noise. Pruning creates simpler trees that generalize better by removing statistically insignificant branches.

    **Concept Tested:** Pruning

---

#### 6. What is the primary difference between a tree node and a leaf node?

<div class="upper-alpha" markdown>
1. Tree nodes contain data while leaf nodes contain predictions
2. Tree nodes store features while leaf nodes store class probabilities
3. Tree nodes perform splits based on features while leaf nodes make final predictions
4. There is no difference; they are the same thing
</div>

??? question "Show Answer"
    The correct answer is **C**. Tree nodes (internal nodes) perform splits by testing a feature against a threshold, directing examples left or right based on the result. Leaf nodes (terminal nodes) are at the ends of branches and make final predictions—either a class label for classification or a numerical value for regression. The path from root to leaf represents the decision logic.

    **Concept Tested:** Tree Node vs Leaf Node

---

#### 7. How does a decision tree handle continuous features?

<div class="upper-alpha" markdown>
1. It converts them to categorical features first
2. It finds optimal threshold values to split the feature into two groups
3. It cannot use continuous features
4. It rounds them to the nearest integer
</div>

??? question "Show Answer"
    The correct answer is **B**. For continuous features, decision trees find optimal threshold values (like "age ≤ 35") that maximize information gain or minimize Gini impurity. The algorithm tests many potential thresholds (often midpoints between adjacent sorted values) and selects the one that best separates classes, creating a binary split at each node.

    **Concept Tested:** Continuous Features (in Decision Trees)

---

#### 8. What does limiting the maximum depth of a decision tree help prevent?

<div class="upper-alpha" markdown>
1. Underfitting by forcing deeper trees
2. Training errors by simplifying splits
3. Memory usage by reducing the dataset size
4. Overfitting by constraining tree complexity
</div>

??? question "Show Answer"
    The correct answer is **D**. Limiting maximum depth (max_depth hyperparameter) prevents overfitting by constraining tree complexity. Deep trees can create very specific rules for individual training examples, memorizing noise. Shallow trees are simpler and more likely to generalize. Common max_depth values range from 3-10, balancing model capacity with generalization.

    **Concept Tested:** Tree Depth (overfitting prevention)

---

#### 9. In a decision tree for predicting house prices (regression), what would a leaf node contain?

<div class="upper-alpha" markdown>
1. The average price of all training examples that reached that leaf
2. The most expensive house in the training set
3. A formula for calculating price
4. The total number of houses in the dataset
</div>

??? question "Show Answer"
    The correct answer is **A**. In regression trees, leaf nodes contain the average (mean) of the target values for all training examples that reached that leaf. For example, if 20 houses with prices [$200K, $210K, $205K, ...] reach a leaf, that leaf would predict their average (say $207K) for any new example following the same path.

    **Concept Tested:** Decision Tree (regression)

---

#### 10. What makes decision trees prone to overfitting without regularization?

<div class="upper-alpha" markdown>
1. They use too few features
2. They are too simple to capture patterns
3. They can create arbitrarily complex trees that memorize training data
4. They cannot handle categorical features
</div>

??? question "Show Answer"
    The correct answer is **C**. Without constraints, decision trees can grow arbitrarily deep, creating leaves for nearly every training example. This results in perfect training accuracy but poor generalization—the tree has memorized specific examples rather than learning general patterns. Regularization techniques (max_depth, min_samples_split, pruning) prevent this by limiting tree complexity.

    **Concept Tested:** Overfitting (in Decision Trees)
