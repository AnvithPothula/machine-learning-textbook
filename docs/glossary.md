# Glossary of Terms

#### Accuracy

The proportion of correct predictions (both true positives and true negatives) among all predictions made by a classification model.

Accuracy is calculated as (TP + TN) / (TP + TN + FP + FN), where TP is true positives, TN is true negatives, FP is false positives, and FN is false negatives. While accuracy is intuitive, it can be misleading for imbalanced datasets where one class dominates.

**Example:** A model that correctly classifies 95 out of 100 iris flowers has 95% accuracy.

#### Activation Function

A mathematical function applied to a neuron's weighted sum of inputs to introduce non-linearity into the neural network.

**Example:** The ReLU activation function outputs max(0, x), passing positive values unchanged while zeroing negative values.

#### Adam Optimizer

An adaptive learning rate optimization algorithm that combines momentum and RMSprop by maintaining exponential moving averages of both gradients and squared gradients.

**Example:** Adam with default parameters (lr=0.001, β₁=0.9, β₂=0.999) is the most common optimizer for training deep neural networks.

#### Algorithm

A step-by-step procedure for solving a problem or performing a computation in machine learning.

**Example:** The k-nearest neighbors algorithm finds the k training examples closest to a query point and predicts based on their labels.

#### AlexNet

A deep convolutional neural network architecture that won the ImageNet 2012 competition, featuring 8 layers (5 convolutional, 3 fully connected) and popularizing ReLU activation and dropout.

**Example:** AlexNet reduced ImageNet top-5 error from 26% to 15.3%, demonstrating the power of deep CNNs for image classification.

#### Artificial Neuron

A computational unit that takes weighted inputs, sums them with a bias term, and applies an activation function to produce an output.

**Example:** A neuron computes output = σ(w₁x₁ + w₂x₂ + b), where σ is the activation function, w are weights, x are inputs, and b is the bias.

#### AUC

Area Under the Curve, measuring the area under the ROC curve to summarize classifier performance across all classification thresholds with a single value between 0 and 1.

**Example:** An AUC of 0.95 indicates the model has a 95% chance of ranking a random positive example higher than a random negative example.

#### Average Pooling

A pooling operation that computes the average value within each pooling window to downsample feature maps in convolutional neural networks.

**Example:** Average pooling over a 2×2 window containing values [1, 3, 2, 4] produces output value 2.5.

#### Backpropagation

An algorithm for computing gradients of the loss function with respect to network weights by applying the chain rule backward through the network layers.

**Example:** In a 3-layer network, backpropagation starts from the output layer loss and propagates gradients backward through hidden layers to the input layer.

#### Batch Processing

Processing multiple data instances simultaneously in groups (batches) rather than one at a time to improve computational efficiency.

**Example:** Training a neural network on batches of 32 images at a time instead of processing images individually reduces training time by parallelizing computations.

#### Batch Size

The number of training examples processed together in one forward and backward pass during neural network training.

**Example:** A batch size of 64 means the model processes 64 images before updating weights, balancing memory usage and gradient stability.

#### Bayesian Optimization

A sequential model-based optimization approach that builds a probabilistic model of the objective function to intelligently select hyperparameters for evaluation.

**Example:** Bayesian optimization uses a Gaussian process to model validation accuracy as a function of learning rate and regularization strength, selecting promising hyperparameters to try next.

#### Bias

A learnable parameter added to the weighted sum of inputs in a neuron before applying the activation function, allowing the neuron to fit patterns that don't pass through the origin.

**Example:** In the linear function y = mx + b, the bias term b shifts the line vertically.

#### Bias-Variance Tradeoff

The fundamental tradeoff in machine learning where reducing model bias (underfitting) tends to increase variance (overfitting), and vice versa.

Simple models have high bias (systematic errors) but low variance (stable predictions), while complex models have low bias but high variance (sensitivity to training data variations). The optimal model balances both sources of error.

**Example:** A linear model on nonlinear data has high bias (underfits), while a 50-layer decision tree has high variance (overfits to noise).

#### Binary Classification

A supervised learning task where the goal is to assign each instance to one of exactly two classes or categories.

**Example:** Email spam detection classifies each email as either "spam" or "not spam."

#### Categorical Features

Input variables that take values from a discrete, finite set of categories or groups without inherent numerical ordering.

**Example:** Color (red, blue, green), country (USA, Canada, Mexico), and email provider (Gmail, Yahoo, Outlook) are categorical features.

#### Centroid

The center point of a cluster in k-means clustering, calculated as the mean of all data points assigned to that cluster.

**Example:** For cluster points (1,2), (3,4), and (5,6), the centroid is ((1+3+5)/3, (2+4+6)/3) = (3, 4).

#### Classification

A supervised learning task where the goal is to predict a discrete class label for each input instance from a predefined set of categories.

**Example:** Classifying iris flowers into species (setosa, versicolor, virginica) based on petal and sepal measurements.

#### Cluster Assignment

The step in k-means clustering where each data point is assigned to the nearest centroid based on distance.

**Example:** In iteration 3 of k-means with 3 clusters, each of 150 data points is assigned to whichever of the 3 centroids is closest.

#### Cluster Update

The step in k-means clustering where centroid positions are recalculated as the mean of all points assigned to each cluster.

**Example:** After assigning points to clusters, the centroid for cluster 1 moves from (2, 3) to (2.5, 3.2) based on the new mean position.

#### CNN Architecture

The overall design and layer organization of a convolutional neural network, specifying the sequence and configuration of convolutional, pooling, and fully connected layers.

**Example:** A typical CNN architecture: Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Softmax.

#### Computational Complexity

A measure of the resources (time and memory) required to execute an algorithm as a function of input size.

**Example:** K-nearest neighbors has O(nd) time complexity for n training examples with d features when making a single prediction.

#### Confusion Matrix

A table showing the counts of true positives, false positives, true negatives, and false negatives for a classification model's predictions.

**Example:** A 2×2 confusion matrix for binary classification shows actual classes in rows and predicted classes in columns, with diagonal entries representing correct predictions.

#### Continuous Features

Input variables that can take any numerical value within a range, typically representing measurements on a continuous scale.

**Example:** Height (175.3 cm), temperature (72.5°F), and income ($45,250.00) are continuous features.

#### Convergence Criteria

The conditions that determine when an iterative optimization algorithm should stop, typically based on change in loss or parameters falling below a threshold.

**Example:** K-means stops when centroids move less than 0.001 units between iterations or after 300 iterations, whichever comes first.

#### Convolution Operation

A mathematical operation that slides a filter (kernel) across an input, computing element-wise products and summing the results at each position to produce a feature map.

**Example:** Convolving a 3×3 edge detection filter across a 28×28 image produces a 26×26 feature map highlighting edges.

#### Convolutional Neural Network

A type of deep neural network specialized for processing grid-structured data (like images) that uses convolution operations to learn hierarchical spatial features.

**Example:** A CNN for image classification might use 5 convolutional layers followed by 2 fully connected layers to classify objects in photos.

#### Cross-Entropy Loss

A loss function measuring the difference between predicted probability distributions and true distributions, commonly used for classification tasks.

**Example:** For binary classification, cross-entropy loss is -[y log(p) + (1-y) log(1-p)], where y is the true label and p is the predicted probability.

#### Cross-Validation

A model evaluation technique that partitions data into multiple subsets, training on some subsets while testing on others, then averaging results across all partitions.

**Example:** 5-fold cross-validation splits data into 5 parts, trains on 4 parts and tests on the remaining part, repeating 5 times with different test folds.

#### Curse of Dimensionality

The phenomenon where data becomes increasingly sparse and distances become less meaningful as the number of features (dimensions) increases.

**Example:** In 10 dimensions, you need exponentially more data points than in 2 dimensions to maintain the same data density for k-NN to work effectively.

#### Data Augmentation

Artificially expanding a training dataset by applying transformations that preserve labels while creating new variations of existing examples.

**Example:** For image classification, rotating, flipping, cropping, and adjusting brightness of training images creates additional training examples without collecting new data.

#### Data Preprocessing

The process of transforming raw data into a format suitable for machine learning algorithms through cleaning, scaling, encoding, and feature engineering.

**Example:** Preprocessing includes removing missing values, scaling features to [0,1] range, and converting categorical variables to one-hot vectors.

#### Decision Boundary

The surface or curve in feature space that separates regions belonging to different classes in a classification model.

**Example:** A linear SVM's decision boundary is a straight line (in 2D) or hyperplane (in higher dimensions) that maximally separates positive and negative examples.

#### Decision Tree

A tree-structured model that makes predictions by recursively splitting the feature space based on feature values, with decision rules at internal nodes and predictions at leaf nodes.

**Example:** A decision tree for iris classification might split first on petal length < 2.5 cm (setosa vs. others), then on petal width < 1.8 cm (versicolor vs. virginica).

#### Deep Learning

A subfield of machine learning focused on neural networks with multiple hidden layers that can learn hierarchical representations of data.

**Example:** A 50-layer ResNet that learns to recognize objects by building representations from edges to textures to parts to complete objects.

#### Dimensionality Reduction

Techniques for reducing the number of features in a dataset while preserving important information and structure.

**Example:** Principal Component Analysis (PCA) reduces 100 features to 10 principal components that capture 95% of the data variance.

#### Distance Metric

A function that quantifies the dissimilarity or separation between two data points in feature space.

**Example:** Euclidean distance between points (1, 2) and (4, 6) is √[(4-1)² + (6-2)²] = 5.

#### Domain Adaptation

Techniques for transferring knowledge from a source domain to a related but different target domain to improve learning with limited target domain data.

**Example:** A model trained on daytime outdoor images (source domain) is adapted to work on nighttime images (target domain) through fine-tuning.

#### Dropout

A regularization technique that randomly sets a fraction of neuron activations to zero during training to prevent overfitting by reducing co-adaptation between neurons.

**Example:** With dropout rate 0.5, each neuron has a 50% probability of being temporarily removed during each training iteration.

#### Dual Formulation

An alternative mathematical formulation of an optimization problem (like SVM) expressed in terms of Lagrange multipliers rather than the original primal variables.

**Example:** The dual formulation of SVM allows the kernel trick by expressing the solution entirely in terms of dot products between training examples.

#### Early Stopping

A regularization technique that halts training when validation performance stops improving, preventing overfitting that occurs from training too long.

**Example:** Training stops at epoch 47 when validation loss hasn't decreased for 10 consecutive epochs, even though training was set for 100 epochs.

#### Elbow Method

A heuristic for selecting the number of clusters in k-means by plotting inertia versus k and choosing the "elbow" point where adding more clusters yields diminishing returns.

**Example:** If inertia drops sharply from k=1 to k=4 but only slightly from k=4 to k=10, choose k=4 as the optimal number of clusters.

#### Entropy

A measure of impurity or disorder in a dataset, quantifying the average amount of information needed to identify the class of a randomly selected instance.

**Example:** A dataset with 50% positive and 50% negative examples has maximum entropy of 1.0 bit, while a pure dataset (100% one class) has zero entropy.

#### Epoch

One complete pass through the entire training dataset during neural network training.

**Example:** Training for 50 epochs on a dataset of 10,000 images means the model sees all 10,000 images 50 times during training.

#### Euclidean Distance

The straight-line distance between two points in feature space, calculated as the square root of the sum of squared differences across all dimensions.

**Example:** Euclidean distance between (1,2,3) and (4,5,6) is √[(4-1)² + (5-2)² + (6-3)²] = √27 ≈ 5.20.

#### Exploding Gradient

A numerical instability during neural network training where gradients grow exponentially large, causing weight updates that destabilize learning.

**Example:** In a 100-layer network without proper initialization, gradients might grow from 0.01 at layer 100 to 10^30 at layer 1, causing NaN values.

#### F1 Score

The harmonic mean of precision and recall, providing a single metric that balances both measures of classifier performance.

**Example:** With precision 0.8 and recall 0.6, F1 score is 2×(0.8×0.6)/(0.8+0.6) = 0.686.

#### False Negative

An instance where the model incorrectly predicts the negative class when the true class is positive (Type II error).

**Example:** A medical test that fails to detect a disease in a patient who actually has the disease is a false negative.

#### False Positive

An instance where the model incorrectly predicts the positive class when the true class is negative (Type I error).

**Example:** A spam filter that marks a legitimate email as spam is a false positive.

#### Feature

An individual measurable property or characteristic of an instance used as input to a machine learning model.

**Example:** In house price prediction, features include square footage, number of bedrooms, and location zip code.

#### Feature Engineering

The process of creating new features or transforming existing features to improve model performance.

**Example:** Creating a "price per square foot" feature by dividing house price by square footage, or extracting day-of-week from a timestamp.

#### Feature Extraction

Using a pre-trained model as a fixed feature extractor by freezing its weights and only training a new classification head on the extracted features.

**Example:** Using a frozen ResNet-50 (pre-trained on ImageNet) to extract 2048-dimensional feature vectors, then training only a new linear classifier on these features.

#### Feature Map

The output of applying a filter through convolution across an input, highlighting specific patterns or features detected at different spatial locations.

**Example:** A 3×3 edge detection filter applied to a 28×28 image produces a 26×26 feature map with high values where edges are detected.

#### Feature Selection

The process of choosing a subset of relevant features from the original feature set to reduce dimensionality and improve model performance.

**Example:** Using correlation analysis to select the 20 most predictive features from an original set of 100 features.

#### Feature Space Partitioning

The division of feature space into regions or subspaces, each associated with a specific prediction or decision.

**Example:** A decision tree partitions 2D feature space into rectangular regions, each corresponding to a different class prediction.

#### Feature Vector

A numerical array representing all features of a single instance, serving as input to machine learning models.

**Example:** An iris flower represented as a 4-dimensional feature vector [5.1, 3.5, 1.4, 0.2] for sepal length, sepal width, petal length, and petal width.

#### Filter

A small matrix of learnable weights that slides across input data in a convolutional layer to detect specific patterns or features.

**Example:** A 3×3 filter might learn to detect vertical edges by having weights like [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]].

#### Fine-Tuning

Continuing to train a pre-trained model on a new dataset by updating all or most of its weights with a small learning rate.

**Example:** Fine-tuning a ResNet-50 pre-trained on ImageNet by training all layers with learning rate 0.0001 on a cats-vs-dogs dataset.

#### Forward Propagation

The process of computing a neural network's output by passing input data through each layer sequentially, applying weights, biases, and activation functions.

**Example:** In forward propagation, an input image flows through convolutional layers, pooling layers, and fully connected layers to produce class probabilities.

#### Freezing Layers

Setting neural network layers to non-trainable by preventing their weights from being updated during training, commonly used in transfer learning.

**Example:** Freezing the first 40 layers of a 50-layer ResNet while fine-tuning only the last 10 layers on a new dataset.

#### Fully Connected Layer

A neural network layer where every neuron connects to every neuron in the previous layer, with each connection having a learnable weight.

**Example:** A fully connected layer with 512 input neurons and 10 output neurons has 512 × 10 = 5,120 trainable weight parameters (plus 10 biases).

#### Gaussian Kernel

A radial basis function kernel for support vector machines that measures similarity between points based on Gaussian-weighted distance, equivalent to RBF kernel.

**Example:** The Gaussian kernel K(x, y) = exp(-γ||x - y||²) with γ=0.1 assigns similarity close to 1 for nearby points and close to 0 for distant points.

#### Generalization

A model's ability to perform well on new, unseen data drawn from the same distribution as the training data.

**Example:** A model with 95% training accuracy and 94% test accuracy generalizes well, while one with 99% training accuracy and 70% test accuracy overfits.

#### Gini Impurity

A measure of impurity for decision tree splitting that quantifies the probability of incorrectly classifying a randomly chosen instance if classified according to class distribution.

**Example:** A node with 40 positive and 60 negative examples has Gini impurity = 1 - (0.4² + 0.6²) = 0.48.

#### Gradient Clipping

A technique that limits the magnitude of gradients during backpropagation to prevent exploding gradients by scaling them when they exceed a threshold.

**Example:** Clipping gradients to maximum norm 1.0 means if the gradient vector has norm 5.0, it's scaled down by a factor of 5 to have norm 1.0.

#### Gradient Descent

An iterative optimization algorithm that updates model parameters in the direction opposite to the gradient of the loss function to minimize loss.

**Example:** Starting with random weights, gradient descent repeatedly computes gradients and updates weights as w_new = w_old - learning_rate × gradient until convergence.

#### Grid Search

A hyperparameter tuning method that exhaustively evaluates all combinations of specified hyperparameter values using cross-validation.

**Example:** Grid search over learning rates [0.001, 0.01, 0.1] and regularization strengths [0.1, 1, 10] trains and evaluates 3 × 3 = 9 models.

#### Hard Margin SVM

A support vector machine that requires perfect linear separation of classes with no training errors allowed.

**Example:** Hard margin SVM works on linearly separable data but fails if even a single training example cannot be correctly classified with a linear boundary.

#### He Initialization

A weight initialization strategy for neural networks using ReLU activation that draws weights from a Gaussian distribution with variance 2/n_in, where n_in is the number of input units.

**Example:** Initializing a layer with 256 input neurons using He initialization samples weights from N(0, √(2/256)).

#### Hidden Layer

An intermediate layer in a neural network between the input and output layers that learns internal representations of the data.

**Example:** A 3-layer neural network has an input layer, one hidden layer with 128 neurons, and an output layer with 10 neurons.

#### Holdout Method

A model evaluation approach that splits data into separate training and test sets, trains on the training set, and evaluates on the held-out test set.

**Example:** Using an 80/20 split, train a model on 80% of data and evaluate on the remaining 20% that was held out.

#### Hyperparameter

A configuration setting for a learning algorithm that is set before training begins rather than learned from data.

**Example:** Learning rate, number of trees in a random forest, and k in k-nearest neighbors are all hyperparameters.

#### Hyperparameter Tuning

The process of finding optimal hyperparameter values to maximize model performance, typically using cross-validation.

**Example:** Testing learning rates from 0.0001 to 0.1 and batch sizes from 16 to 128 to find the combination that minimizes validation loss.

#### Hyperplane

A linear subspace of dimension n-1 in an n-dimensional space that divides the space into two half-spaces, used as decision boundaries in linear classifiers.

**Example:** In 3D space, a hyperplane is a 2D plane defined by equation w₁x₁ + w₂x₂ + w₃x₃ + b = 0 that separates positive and negative examples.

#### ImageNet

A large-scale image database containing 14 million labeled images across 20,000+ categories, with a subset of 1.2 million images in 1,000 categories commonly used for training deep learning models.

**Example:** Pre-trained models like ResNet-50 are trained on ImageNet's 1,000-class subset before being fine-tuned for specific tasks.

#### Inception

A CNN architecture family (including GoogLeNet, Inception-v3, Inception-v4) that uses inception modules with parallel convolutional filters of different sizes to capture multi-scale features efficiently.

**Example:** An inception module applies 1×1, 3×3, and 5×5 convolutions in parallel, concatenating their outputs to capture features at multiple scales simultaneously.

#### Inertia

The sum of squared distances from each data point to its assigned cluster centroid in k-means clustering, measuring cluster compactness.

**Example:** Lower inertia indicates tighter clusters; inertia of 150 means the average squared distance from points to their centroids is 150.

#### Information Gain

The reduction in entropy (or increase in information) achieved by splitting a dataset on a particular feature, used to select the best split in decision trees.

**Example:** If parent node entropy is 0.8 and splitting produces children with entropies 0.3 and 0.5, information gain is 0.8 - weighted_average(0.3, 0.5).

#### Input Layer

The first layer of a neural network that receives raw input features and passes them to subsequent layers.

**Example:** For 28×28 grayscale images, the input layer has 784 neurons (28 × 28), one for each pixel value.

#### Instance

A single data point, observation, or example in a dataset, consisting of feature values and optionally a label.

**Example:** One row in a dataset representing a single iris flower with measurements [5.1, 3.5, 1.4, 0.2, "setosa"] is an instance.

#### K-Fold Cross-Validation

A cross-validation technique that divides data into k equal folds, using each fold once as a test set while training on the remaining k-1 folds, then averaging performance across all k trials.

**Example:** 10-fold cross-validation on 1,000 examples creates 10 train/test splits, each using 900 examples for training and 100 for testing.

#### K-Means Clustering

An unsupervised learning algorithm that partitions data into k clusters by iteratively assigning points to nearest centroids and updating centroids as cluster means.

**Example:** K-means with k=3 on customer data might discover three natural segments: high-value, medium-value, and low-value customers.

#### K-Means Initialization

The method for setting initial centroid positions before k-means iterations begin, significantly affecting convergence and final cluster quality.

**Example:** Random initialization selects k random data points as initial centroids, while k-means++ selects centroids spread far apart to improve results.

#### K-Means++ Initialization

An improved initialization method for k-means that selects initial centroids probabilistically, favoring points far from already-chosen centroids to improve clustering.

**Example:** K-means++ first selects one random centroid, then selects each subsequent centroid with probability proportional to its squared distance from the nearest existing centroid.

#### K-Nearest Neighbors

A non-parametric algorithm that predicts a query point's label based on the majority class (classification) or average value (regression) of its k nearest training examples.

**Example:** 5-NN for iris classification finds the 5 closest training flowers to a new flower and predicts the majority species among those 5 neighbors.

#### K Selection

The process of choosing an appropriate value for k (number of neighbors) in k-nearest neighbors or k (number of clusters) in k-means.

**Example:** Testing k values from 1 to 20 using cross-validation and selecting k=7 because it achieves the lowest validation error.

#### Kernel Size

The dimensions of a convolutional filter, typically specified as height × width (and optionally depth for multi-channel inputs).

**Example:** A kernel size of 3×3 means the filter covers a 3×3 spatial region when sliding across the input.

#### Kernel Trick

A mathematical technique in SVMs that implicitly maps data to high-dimensional feature spaces using kernel functions without explicitly computing the transformation.

**Example:** The RBF kernel allows SVMs to learn non-linear decision boundaries by implicitly mapping 2D data to infinite-dimensional space while only computing dot products.

#### KNN for Classification

Applying k-nearest neighbors to classification tasks by assigning a query point the majority class among its k nearest neighbors.

**Example:** In binary classification with k=5, if a point's 5 nearest neighbors include 4 positive and 1 negative examples, predict positive class.

#### KNN for Regression

Applying k-nearest neighbors to regression tasks by predicting a query point's value as the average of its k nearest neighbors' values.

**Example:** For house price prediction with k=3, predict the price as the average of the 3 most similar houses' prices.

#### L1 Regularization

A regularization technique that adds the sum of absolute values of weights to the loss function, encouraging sparse models with many weights driven to exactly zero.

**Example:** L1 penalty λ Σ|wᵢ| with λ=0.01 penalizes large weights, often resulting in 80% of weights becoming exactly zero (feature selection).

#### L2 Regularization

A regularization technique that adds the sum of squared weights to the loss function, encouraging small but non-zero weights.

**Example:** L2 penalty λ Σwᵢ² with λ=0.01 shrinks all weights toward zero proportionally without making them exactly zero.

#### Label

The target output or ground truth value associated with an instance in supervised learning, representing the correct answer the model should learn to predict.

**Example:** In email classification, labels are "spam" or "not spam"; in house price prediction, labels are dollar amounts.

#### Label Encoding

Converting categorical variables to integers by assigning each unique category a number, creating an ordinal relationship.

**Example:** Encoding colors {red, blue, green} as {0, 1, 2}.

#### Lasso Regression

Linear regression with L1 regularization that performs automatic feature selection by driving some coefficients to exactly zero.

**Example:** Lasso regression with α=1.0 on 100 features might select only 15 non-zero coefficients, effectively performing feature selection.

#### Lazy Learning

A learning paradigm where the algorithm defers processing until prediction time rather than building an explicit model during training.

**Example:** K-nearest neighbors is lazy learning because it stores all training data and performs computation only when making predictions, unlike eager learners like decision trees.

#### Leaf Node

A terminal node in a decision tree that contains no children and makes a final prediction based on the training instances that reach it.

**Example:** A leaf node in an iris decision tree might predict "setosa" with 100% confidence based on 50 training instances that all belong to the setosa class.

#### Learning Rate

A hyperparameter controlling the step size for weight updates during gradient descent optimization.

**Example:** With learning rate 0.01, if the gradient is 5.0, the weight update is 0.01 × 5.0 = 0.05.

#### Learning Rate Scheduling

Adjusting the learning rate during training according to a predefined schedule or adaptive rule to improve convergence.

**Example:** Starting with learning rate 0.1 and multiplying by 0.1 every 10 epochs: epochs 1-10 use 0.1, epochs 11-20 use 0.01, epochs 21-30 use 0.001.

#### LeNet

An early convolutional neural network architecture designed by Yann LeCun for handwritten digit recognition, featuring 2 convolutional layers followed by 3 fully connected layers.

**Example:** LeNet-5 processes 32×32 grayscale images through Conv→Pool→Conv→Pool→FC→FC→FC layers to classify MNIST digits.

#### Linear Kernel

A kernel function for support vector machines that computes the dot product between feature vectors without transformation, suitable for linearly separable data.

**Example:** Linear kernel K(x, y) = x · y creates the same decision boundary as a linear SVM without kernels but allows the dual formulation.

#### Local Connectivity

A property of convolutional layers where each neuron connects only to a small local region of the input rather than all input units.

**Example:** In a convolutional layer, each neuron in the feature map connects to only a 3×3 patch of the previous layer instead of all pixels.

#### Log-Loss

The cross-entropy loss function for binary classification, measuring the negative log-likelihood of the true labels given the predicted probabilities.

**Example:** For true label y=1 and predicted probability p=0.9, log-loss is -log(0.9) ≈ 0.105, penalizing incorrect predictions exponentially.

#### Logistic Regression

A linear classification algorithm that models class probabilities using the logistic (sigmoid) function and estimates weights via maximum likelihood.

**Example:** Logistic regression for spam detection computes P(spam|email) = σ(w₁×word_count + w₂×link_count + b), where σ is the sigmoid function.

#### Loss Function

A function measuring the difference between a model's predictions and true labels, quantifying how well the model performs on the training data.

**Example:** Mean squared error (MSE) loss for regression: (1/n) Σ(yᵢ - ŷᵢ)², where yᵢ is true value and ŷᵢ is predicted value.

#### Machine Learning

A field of artificial intelligence focused on developing algorithms that improve automatically through experience and data without explicit programming.

**Example:** A machine learning system learns to recognize cats in photos by analyzing thousands of labeled cat and non-cat images, discovering visual patterns automatically.

#### Manhattan Distance

The distance between two points calculated as the sum of absolute differences across all dimensions, equivalent to the distance traveled along grid lines.

**Example:** Manhattan distance between (1, 2) and (4, 6) is |4-1| + |6-2| = 7.

#### Margin

The perpendicular distance from the decision boundary to the nearest training examples (support vectors) in support vector machines.

**Example:** If the decision boundary is 2x₁ + x₂ = 5 and the nearest point is at (2, 1), the margin is the perpendicular distance from this point to the line.

#### Margin Maximization

The optimization objective in support vector machines of finding the decision boundary that maximizes the margin between classes.

**Example:** Among infinitely many hyperplanes that separate positive and negative examples, SVM selects the one with maximum distance to the nearest examples on both sides.

#### Max Pooling

A pooling operation that selects the maximum value within each pooling window to downsample feature maps while retaining the strongest activations.

**Example:** Max pooling over a 2×2 window containing values [1, 3, 2, 4] outputs 4, the maximum value.

#### Maximum Likelihood

A principle for estimating model parameters by finding values that maximize the probability of observing the training data.

**Example:** In logistic regression, maximum likelihood estimation finds weights that make the observed class labels most probable given the features.

#### Mean Squared Error

A loss function for regression that computes the average of squared differences between predicted and true values.

**Example:** For predictions [2, 3, 5] and true values [1, 4, 4], MSE = [(2-1)² + (3-4)² + (5-4)²]/3 = 1.

#### Mini-Batch Gradient Descent

A gradient descent variant that computes gradients and updates weights using a randomly selected subset (mini-batch) of training examples.

**Example:** With batch size 32, mini-batch gradient descent uses 32 randomly sampled examples to compute each gradient update, balancing efficiency and gradient quality.

#### Min-Max Scaling

A normalization technique that linearly transforms features to a fixed range (typically [0, 1]) by subtracting the minimum and dividing by the range.

**Example:** Feature with values [10, 20, 30] normalized to [0, 1] range becomes [0, 0.5, 1] using formula (x - min)/(max - min).

#### Model

A mathematical representation learned from data that maps inputs to outputs, encapsulating the patterns discovered during training.

**Example:** A trained decision tree model with specific split thresholds and leaf predictions represents the learned relationship between features and labels.

#### Model Evaluation

The process of measuring a trained model's performance using appropriate metrics on test data to assess its predictive capability.

**Example:** Evaluating a classifier using accuracy, precision, recall, and F1 score on a held-out test set to quantify performance.

#### Model Selection

The process of choosing the best model architecture or algorithm from multiple candidates based on validation performance.

**Example:** Comparing k-NN, decision trees, and logistic regression using cross-validation and selecting the model with lowest validation error.

#### Model Zoo

A collection of pre-trained neural network models publicly available for download and use in transfer learning.

**Example:** PyTorch's model zoo provides pre-trained ResNet, VGG, and AlexNet models trained on ImageNet that can be downloaded and fine-tuned.

#### Momentum

An optimization technique that accelerates gradient descent by accumulating a velocity vector in the direction of persistent gradient components.

**Example:** Momentum with coefficient 0.9 makes the optimizer build up speed in directions where gradients consistently point the same way, like a ball rolling downhill.

#### Multiclass Classification

A classification task where instances must be assigned to one of three or more distinct classes.

**Example:** Classifying iris flowers into three species (setosa, versicolor, virginica) or recognizing handwritten digits (0-9) are multiclass problems.

#### Multilayer Perceptron

A feedforward neural network with one or more hidden layers between input and output layers, capable of learning non-linear relationships.

**Example:** A multilayer perceptron with architecture [784, 128, 64, 10] has an input layer (784 pixels), two hidden layers (128 and 64 neurons), and an output layer (10 classes).

#### Nesterov Momentum

An improved momentum variant that computes gradients at an approximate future position rather than the current position, often converging faster.

**Example:** Nesterov momentum looks ahead by temporarily moving in the momentum direction before computing the gradient, providing better anticipation of the loss landscape.

#### Network Architecture

The overall structure and organization of a neural network, specifying the number, types, and connections of layers.

**Example:** Architecture [Input: 784] → [Dense: 512, ReLU] → [Dropout: 0.5] → [Dense: 10, Softmax] defines a 2-layer fully connected network.

#### Neural Network

A computational model inspired by biological neural networks consisting of interconnected nodes (neurons) organized in layers that learn to map inputs to outputs.

**Example:** A neural network for image classification consists of an input layer receiving pixel values, hidden layers learning features, and an output layer producing class probabilities.

#### Normalization

Transforming features to a common scale to improve learning algorithm performance and convergence.

**Example:** Normalizing all features to the range [0, 1] using min-max scaling or to mean 0 and standard deviation 1 using standardization.

#### One-Hot Encoding

Converting categorical variables into binary vectors where exactly one element is 1 (hot) and all others are 0, creating separate features for each category.

**Example:** Encoding colors {red, blue, green} as red=[1,0,0], blue=[0,1,0], green=[0,0,1].

#### One-vs-All

A multiclass classification strategy that trains k binary classifiers (one per class) to distinguish each class from all others combined.

**Example:** For 5 classes, one-vs-all trains 5 binary classifiers: class_1 vs. others, class_2 vs. others, ..., class_5 vs. others.

#### One-vs-One

A multiclass classification strategy that trains k(k-1)/2 binary classifiers, one for each pair of classes, then aggregates their votes.

**Example:** For 5 classes, one-vs-one trains 10 binary classifiers for all pairs: (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5).

#### Online Learning

A learning paradigm where models are updated incrementally as new data arrives, rather than being trained once on a fixed dataset.

**Example:** A spam filter that updates its model every time a user marks an email as spam or not spam, continuously adapting to new spam patterns.

#### Optimizer

An algorithm that adjusts model parameters to minimize the loss function, typically variants of gradient descent.

**Example:** Adam, SGD with momentum, and RMSprop are popular optimizers for training neural networks, each with different strategies for updating weights.

#### Output Layer

The final layer of a neural network that produces predictions, with the number of neurons matching the prediction task.

**Example:** For 10-class classification, the output layer has 10 neurons with softmax activation producing probability estimates for each class.

#### Overfitting

A modeling error where a model learns patterns specific to the training data (including noise) rather than general patterns, resulting in poor performance on new data.

**Example:** A decision tree with 50 levels that achieves 100% training accuracy but only 60% test accuracy has overfit to training noise.

#### Padding

Adding extra pixels (typically zeros) around the border of an input before convolution to control the spatial dimensions of the output feature map.

**Example:** Adding 1 pixel of padding around a 5×5 image creates a 7×7 padded input, allowing a 3×3 filter to produce a 5×5 output instead of 3×3.

#### Perceptron

A single-layer neural network that learns a linear decision boundary for binary classification using a simple update rule.

**Example:** The perceptron learning algorithm adjusts weights when a training example is misclassified: w_new = w_old + learning_rate × y × x.

#### Polynomial Kernel

A kernel function for SVMs that computes polynomial combinations of features, enabling learning of polynomial decision boundaries.

**Example:** Polynomial kernel K(x, y) = (x · y + c)^d with degree d=2 allows SVMs to learn parabolic decision boundaries.

#### Pooling Layer

A downsampling layer in CNNs that reduces the spatial dimensions of feature maps while retaining important information.

**Example:** A 2×2 max pooling layer reduces a 28×28 feature map to 14×14 by taking the maximum value in each 2×2 window.

#### Precision

The proportion of positive predictions that are actually correct, measuring how many predicted positives are true positives.

**Example:** If a spam filter marks 100 emails as spam and 80 actually are spam, precision is 80/100 = 0.80.

#### Pre-Trained Model

A neural network model that has been trained on a large dataset and whose learned weights can be reused for related tasks through transfer learning.

**Example:** A ResNet-50 pre-trained on ImageNet with weights learned from 1.2 million images that can be fine-tuned for medical image classification.

#### Primal Formulation

The original form of an optimization problem expressed in terms of the primary decision variables (like weights in SVM).

**Example:** The primal SVM formulation minimizes ||w||² subject to constraints on the margin for each training example.

#### Pruning

Removing branches or nodes from a decision tree to reduce complexity and prevent overfitting.

**Example:** Post-pruning removes tree branches where validation error doesn't improve, reducing a 20-level tree to 8 levels while maintaining or improving test accuracy.

#### Radial Basis Function

A kernel function for SVMs that measures similarity between points based on their Euclidean distance, creating circular (radial) decision boundaries.

**Example:** RBF kernel K(x, y) = exp(-γ||x - y||²) with γ=0.1 creates smooth, non-linear decision boundaries that can form complex shapes.

#### Random Initialization

Starting k-means clustering with k randomly selected data points as initial centroids.

**Example:** For k=3, randomly choose 3 of the 150 training points as initial cluster centers before beginning k-means iterations.

#### Random Search

A hyperparameter tuning method that randomly samples hyperparameter combinations from specified distributions and evaluates them using cross-validation.

**Example:** Sampling 50 random combinations of learning rate from [0.0001, 0.1] and regularization from [0.001, 10] using logarithmic distributions.

#### Recall

The proportion of actual positives that are correctly identified, measuring how many true positives are captured by the model.

**Example:** If 100 emails are actually spam and a filter correctly identifies 70 of them, recall is 70/100 = 0.70 (also called sensitivity or true positive rate).

#### Receptive Field

The region of the input that influences a particular neuron's activation in a neural network, growing larger in deeper layers of CNNs.

**Example:** In a CNN with two 3×3 convolutional layers, a neuron in the second layer has a 5×5 receptive field in the original input image.

#### Regression

A supervised learning task where the goal is to predict a continuous numerical value for each input instance.

**Example:** Predicting house prices (in dollars) based on features like square footage, location, and number of bedrooms.

#### Regularization

Techniques for reducing model complexity and preventing overfitting by adding constraints or penalties to the learning process.

**Example:** Adding an L2 penalty term λ Σwᵢ² to the loss function discourages large weights, reducing overfitting in linear regression.

#### ReLU

Rectified Linear Unit activation function that outputs the input if positive, otherwise zero: f(x) = max(0, x).

**Example:** ReLU([-2, 0, 3]) = [0, 0, 3], eliminating negative values while preserving positive values unchanged.

#### ResNet

A deep CNN architecture using residual connections (skip connections) that enable training of very deep networks (50-200 layers) by addressing vanishing gradients.

**Example:** ResNet-50 uses residual blocks with skip connections, allowing gradients to flow directly through the network without diminishing over 50 layers.

#### Ridge Regression

Linear regression with L2 regularization that shrinks coefficient estimates to reduce overfitting.

**Example:** Ridge regression with α=1.0 minimizes (Σ(yᵢ - ŷᵢ)² + α Σwⱼ²), trading some training error for smaller, more stable coefficients.

#### RMSprop

An adaptive learning rate optimizer that divides the learning rate by a running average of recent gradient magnitudes.

**Example:** RMSprop adapts learning rates per parameter, using larger steps for parameters with small gradients and smaller steps for those with large gradients.

#### ROC Curve

Receiver Operating Characteristic curve plotting true positive rate against false positive rate at various classification thresholds.

**Example:** An ROC curve shows how recall and false positive rate change as you vary the threshold for classifying examples as positive, with perfect classifiers reaching point (0, 1).

#### Same Padding

A padding strategy that adds enough border pixels so the output feature map has the same spatial dimensions as the input.

**Example:** For a 5×5 input with a 3×3 filter, same padding adds 1 pixel of padding on all sides to produce a 5×5 output instead of 3×3.

#### Scalability

A system's ability to handle increasing amounts of data or computational demands by adding resources.

**Example:** An algorithm with O(n log n) time complexity is more scalable than one with O(n²) because runtime grows more slowly as dataset size n increases.

#### Sensitivity

The true positive rate or recall, measuring the proportion of actual positives correctly identified by the classifier.

**Example:** A medical test with 90% sensitivity correctly identifies 90 out of 100 patients who actually have the disease.

#### Sigmoid Activation

An activation function that maps inputs to the range (0, 1) using the formula f(x) = 1 / (1 + e^(-x)).

**Example:** Sigmoid(0) = 0.5, sigmoid(2) ≈ 0.88, sigmoid(-2) ≈ 0.12, creating a smooth S-shaped curve useful for binary classification output layers.

#### Sigmoid Function

A mathematical function that maps any real number to the range (0, 1), commonly used in logistic regression and neural network output layers for binary classification.

**Example:** The sigmoid function σ(x) = 1/(1 + e^(-x)) converts a linear combination of features into a probability estimate between 0 and 1.

#### Silhouette Score

A metric measuring how well-separated clusters are by comparing the average distance to points in the same cluster versus the average distance to points in the nearest different cluster.

**Example:** Silhouette scores range from -1 (poor clustering) to +1 (excellent clustering), with values near 0 indicating overlapping clusters.

#### Slack Variables

Variables in soft margin SVMs that allow some training examples to violate the margin constraints, enabling solutions for non-linearly separable data.

**Example:** A slack variable ξᵢ > 0 for example i indicates it lies within the margin or on the wrong side of the decision boundary, with larger values for worse violations.

#### Soft Margin SVM

A support vector machine variant that tolerates some classification errors and margin violations through slack variables and a penalty parameter C.

**Example:** Soft margin SVM with C=1.0 allows misclassifications to achieve a wider margin, balancing margin maximization with training accuracy.

#### Softmax Function

A function that converts a vector of real numbers into a probability distribution, where each element is between 0 and 1 and all elements sum to 1.

**Example:** Softmax([2, 1, 0]) = [0.66, 0.24, 0.10], converting raw scores (logits) into probabilities that sum to 1 for multiclass classification.

#### Space Complexity

The amount of memory required by an algorithm as a function of input size.

**Example:** K-nearest neighbors has O(nd) space complexity because it stores all n training examples with d features, while logistic regression has O(d) space for just the weight vector.

#### Spatial Hierarchies

The layered representation of features in CNNs where early layers detect simple patterns (edges) and deeper layers detect complex patterns (objects) by combining simpler features.

**Example:** In a CNN for face recognition, layer 1 detects edges, layer 2 detects facial features (eyes, nose), layer 3 detects face parts, layer 4 detects complete faces.

#### Specificity

The true negative rate, measuring the proportion of actual negatives correctly identified by the classifier.

**Example:** A test with 95% specificity correctly identifies 95 out of 100 people who don't have the disease as healthy (5 false positives).

#### Splitting Criterion

The rule or metric used to select features and thresholds for splitting nodes in decision tree construction.

**Example:** A decision tree might use information gain as the splitting criterion, selecting at each node the feature and threshold that maximize information gain.

#### Standardization

A normalization technique that transforms features to have mean 0 and standard deviation 1 by subtracting the mean and dividing by the standard deviation.

**Example:** Feature with values [10, 20, 30] (mean=20, std=8.165) standardized becomes [-1.22, 0, 1.22] using formula (x - μ)/σ.

#### Stochastic Gradient Descent

A gradient descent variant that computes gradients and updates weights using one randomly selected training example at a time.

**Example:** Instead of computing gradients over all 10,000 training examples, SGD randomly samples one example, computes its gradient, updates weights, then repeats with another random example.

#### Stratified Sampling

A sampling strategy that preserves the proportion of each class when creating train/test splits, particularly important for imbalanced datasets.

**Example:** For data with 80% class A and 20% class B, stratified sampling ensures both training and test sets maintain the same 80/20 split.

#### Stride

The step size by which a convolutional filter moves across the input at each position.

**Example:** A stride of 2 means the filter moves 2 pixels at a time, reducing the output size by half in each dimension compared to stride 1.

#### Supervised Learning

A machine learning paradigm where models learn from labeled training data to predict labels for new, unseen instances.

**Example:** Training a model on 1,000 labeled images of cats and dogs (supervised data) to predict whether new images contain cats or dogs.

#### Support Vector Machine

A supervised learning algorithm that finds the optimal hyperplane maximizing the margin between classes, optionally using kernel functions for non-linear decision boundaries.

**Example:** SVM with RBF kernel separates non-linearly separable data by implicitly mapping it to a higher-dimensional space where a linear boundary works.

#### Support Vectors

The training examples closest to the decision boundary in an SVM that determine the position and orientation of the optimal hyperplane.

**Example:** In a dataset of 1,000 examples, only 50 might be support vectors (points lying on the margin), and removing non-support vectors doesn't change the decision boundary.

#### Tanh

Hyperbolic tangent activation function that maps inputs to the range (-1, 1) using f(x) = (e^x - e^(-x)) / (e^x + e^(-x)).

**Example:** Tanh(0) = 0, tanh(2) ≈ 0.96, tanh(-2) ≈ -0.96, providing a zero-centered alternative to sigmoid.

#### Test Data

A held-out dataset used only for final model evaluation after all training and hyperparameter tuning is complete, providing an unbiased estimate of performance.

**Example:** Setting aside 20% of data as test data that is never used during model development, only for the final performance report.

#### Test Error

The error rate or loss computed on the test set, measuring model performance on unseen data and serving as an estimate of real-world performance.

**Example:** A model with 5% test error (95% test accuracy) is expected to correctly classify 95% of new, previously unseen examples.

#### Time Complexity

The computational time required by an algorithm as a function of input size, typically expressed using Big-O notation.

**Example:** K-nearest neighbors has O(nd) time complexity per prediction for n training examples with d features, while decision tree prediction is O(log n) for a balanced tree.

#### Training Data

The dataset used to fit a machine learning model by adjusting its parameters to minimize the loss function.

**Example:** Using 80% of available labeled images to train a classifier by iteratively updating weights to reduce classification errors.

#### Training Error

The error rate or loss computed on the training set, measuring how well the model fits the data it was trained on.

**Example:** A model with 2% training error correctly classifies 98% of training examples, but this may indicate overfitting if test error is much higher.

#### Transfer Learning

A technique that leverages knowledge learned from one task or domain to improve learning in a different but related task or domain, typically by starting with pre-trained model weights.

**Example:** Using a ResNet-50 pre-trained on ImageNet to classify medical images by fine-tuning the final layers on a smaller medical image dataset.

#### Translation Invariance

A property of CNNs where the network's ability to recognize patterns is approximately independent of their spatial location in the input.

**Example:** A CNN trained to detect cats can recognize a cat whether it appears in the top-left, center, or bottom-right of an image.

#### Tree Depth

The maximum number of edges from the root to any leaf node in a decision tree, controlling model complexity.

**Example:** A tree with depth 5 makes up to 5 sequential decisions to reach a prediction, while depth 1 (decision stump) makes a single decision.

#### Tree Node

An internal point in a decision tree that splits data based on a feature value test, with child nodes for each outcome of the test.

**Example:** A node might test "petal length < 2.5 cm" and direct examples with shorter petals to the left child and longer petals to the right child.

#### True Negative

An instance where the model correctly predicts the negative class when the true class is indeed negative.

**Example:** A spam filter correctly identifies a legitimate email as "not spam" is a true negative.

#### True Positive

An instance where the model correctly predicts the positive class when the true class is indeed positive.

**Example:** A medical test correctly identifying a patient with disease as "positive" is a true positive.

#### Underfitting

A modeling error where a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.

**Example:** Using linear regression (underfitting) on highly non-linear data yields 70% training accuracy and 68% test accuracy, both poor.

#### Universal Approximation

A theorem stating that a feedforward neural network with at least one hidden layer and sufficient neurons can approximate any continuous function to arbitrary precision.

**Example:** A single hidden layer MLP with enough neurons can theoretically learn any continuous mapping from inputs to outputs, though depth often makes learning more practical.

#### Unsupervised Learning

A machine learning paradigm where models learn patterns and structure from unlabeled data without explicit target outputs.

**Example:** K-means clustering discovers natural groupings in customer data without being told how many groups exist or what defines each group.

#### Valid Padding

A padding strategy where no padding is added, so the output size is smaller than the input by the filter size minus one.

**Example:** A 5×5 input with a 3×3 filter and valid padding produces a 3×3 output (5 - 3 + 1 = 3).

#### Validation Data

A dataset held out from training used to tune hyperparameters and make model selection decisions during development.

**Example:** Using 20% of training data as validation data to select the best learning rate, then retraining on all training data with the chosen learning rate before final test evaluation.

#### Validation Error

The error rate or loss computed on the validation set, used for hyperparameter tuning and model selection without touching the test set.

**Example:** Tracking validation error during training to implement early stopping when it stops decreasing, preventing overfitting.

#### Vanishing Gradient

A numerical instability during neural network training where gradients become extremely small in early layers, preventing effective learning.

**Example:** In a 100-layer network using sigmoid activation, gradients might shrink from 0.01 at layer 100 to 10^(-30) at layer 1, making learning impossible without techniques like ReLU or batch normalization.

#### VGG

A CNN architecture family characterized by using many small (3×3) convolutional filters in deep networks (VGG-16, VGG-19), demonstrating that depth improves performance.

**Example:** VGG-16 uses 16 weight layers with consistent 3×3 filters throughout, achieving strong ImageNet performance with a simple, uniform architecture.

#### Voronoi Diagram

A partitioning of space into regions where each region contains all points closest to a particular data point, visualizing k-NN decision boundaries.

**Example:** For k=1, the k-NN decision boundary forms a Voronoi diagram with each region corresponding to one training example's class.

#### Weight Initialization

The strategy for setting initial values of neural network weights before training begins, critically affecting convergence speed and final performance.

**Example:** Xavier initialization sets initial weights to random values from N(0, √(1/n_in)), preventing activations from vanishing or exploding in early training.

#### Weight Sharing

A property of convolutional layers where the same filter weights are applied at every spatial location, dramatically reducing parameters compared to fully connected layers.

**Example:** A 3×3 filter with weight sharing uses only 9 parameters regardless of input size, while a fully connected layer connecting 100 input to 100 output neurons needs 10,000 parameters.

#### Weights

Learnable parameters in a neural network that multiply input values, representing the strength of connections between neurons.

**Example:** In a neuron computing output = w₁x₁ + w₂x₂ + b, the weights w₁ and w₂ determine how much each input contributes to the output.

#### Within-Cluster Variance

A measure of how spread out points are within each cluster, typically measured as the sum of squared distances from points to their cluster centroid.

**Example:** Lower within-cluster variance indicates tighter, more compact clusters with points close to their centroids.

#### Xavier Initialization

A weight initialization strategy for neural networks using symmetric activations (like tanh) that draws weights from a distribution with variance 1/n_in.

**Example:** Xavier initialization for a layer with 100 input neurons samples weights from N(0, √(1/100)) to maintain activation variance across layers.

#### Z-Score Normalization

A standardization technique that transforms features to have mean 0 and standard deviation 1 by subtracting the mean and dividing by the standard deviation.

**Example:** Feature values [10, 20, 30] with mean 20 and std 8.165 become [-1.22, 0, 1.22] after z-score normalization.
