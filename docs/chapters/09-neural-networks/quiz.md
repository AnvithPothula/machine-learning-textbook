# Quiz: Neural Networks Fundamentals

Test your understanding of neural networks fundamentals with these questions.

---

#### 1. What is the purpose of an activation function in a neural network?

<div class="upper-alpha" markdown>
1. To normalize the input data before training
2. To introduce nonlinearity so the network can learn complex patterns
3. To prevent overfitting by adding regularization
4. To compute the loss function during backpropagation
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Activation functions introduce nonlinearity into neural networks. Without them, stacking multiple layers would be mathematically equivalent to a single linear transformation—no matter how deep the network, it could only learn linear relationships. Activation functions like ReLU, sigmoid, and tanh allow networks to learn complex, nonlinear patterns by introducing non-linear transformations between layers. This is fundamental to neural networks' ability to approximate arbitrary functions and solve complex problems.

    **Concept Tested:** Activation Function, Neural Network

---

#### 2. Which activation function is most commonly used for hidden layers in modern deep neural networks?

<div class="upper-alpha" markdown>
1. Sigmoid
2. Tanh
3. ReLU
4. Linear
</div>

??? question "Show Answer"
    The correct answer is **C**.

    ReLU (Rectified Linear Unit) has become the default activation function for hidden layers because it alleviates the vanishing gradient problem, is computationally cheap (simple thresholding: $\max(0, z)$), accelerates convergence (up to 6x faster than sigmoid/tanh in some studies), and promotes sparse activations. While sigmoid and tanh were historically popular, they suffer from vanishing gradients for large input magnitudes. Linear activation (option D) provides no nonlinearity and would defeat the purpose of deep networks.

    **Concept Tested:** ReLU, Activation Function, Hidden Layer

---

#### 3. What problem does Leaky ReLU address that standard ReLU suffers from?

<div class="upper-alpha" markdown>
1. Vanishing gradients for positive inputs
2. Dying neurons that never activate due to large negative weights
3. Excessive computational cost
4. Inability to output negative values when needed for regression
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Standard ReLU outputs zero for all negative inputs, with gradient zero in that region. If a neuron's weights become large and negative, it outputs zero for all inputs and receives zero gradient during backpropagation—it "dies" and can never recover. Leaky ReLU addresses this by allowing a small negative slope (typically 0.01) for negative inputs: $\text{LeakyReLU}(z) = \max(\alpha z, z)$. This ensures neurons always have some gradient and can continue learning even if they enter the negative region.

    **Concept Tested:** Leaky ReLU, ReLU, Activation Function

---

#### 4. In a neural network with architecture 10-64-32-3 (input-hidden-hidden-output), how many weight matrices are there?

<div class="upper-alpha" markdown>
1. 2
2. 3
3. 4
4. 109
</div>

??? question "Show Answer"
    The correct answer is **B**.

    There is one weight matrix between each pair of consecutive layers. For a 10-64-32-3 network: (1) weights from input layer (10 neurons) to first hidden layer (64 neurons): $10 \times 64$ matrix, (2) weights from first hidden (64) to second hidden (32): $64 \times 32$ matrix, and (3) weights from second hidden (32) to output (3): $32 \times 3$ matrix. Total: 3 weight matrices. The total number of weights would be $10 \times 64 + 64 \times 32 + 32 \times 3 = 2,784$, but the question asks for the number of matrices, not individual weights.

    **Concept Tested:** Network Architecture, Weights, Fully Connected Layer

---

#### 5. During forward propagation, what does each neuron in a hidden layer compute?

<div class="upper-alpha" markdown>
1. The gradient of the loss with respect to its weights
2. A weighted sum of inputs plus bias, followed by an activation function
3. The distance from the input to learned cluster centroids
4. The probability that each class is correct
</div>

??? question "Show Answer"
    The correct answer is **B**.

    During forward propagation, each neuron computes two operations: (1) a weighted sum of its inputs plus a bias term: $z = \sum_{i} w_i x_i + b$, and (2) application of an activation function: $a = f(z)$. The output $a$ becomes the input to neurons in the next layer. This process repeats through all layers until producing final outputs. Computing gradients (option A) happens during backpropagation, not forward propagation. The other options describe different types of algorithms entirely.

    **Concept Tested:** Forward Propagation, Artificial Neuron, Hidden Layer

---

#### 6. What is the primary purpose of backpropagation in neural network training?

<div class="upper-alpha" markdown>
1. To make predictions on new data points
2. To efficiently compute gradients of the loss with respect to all parameters
3. To prevent overfitting by stopping training early
4. To initialize weights to appropriate starting values
</div>

??? question "Show Answer"
    The correct answer is **B**.

    Backpropagation (backward propagation of errors) uses the chain rule from calculus to efficiently compute gradients of the loss function with respect to every weight and bias in the network. These gradients indicate how to adjust each parameter to reduce loss. Backpropagation works backward from the output layer to the input layer, propagating error information. Before backpropagation was developed, training neural networks was computationally prohibitive. The algorithm makes deep learning practical by enabling efficient gradient computation.

    **Concept Tested:** Backpropagation, Gradient Descent, Neural Network

---

#### 7. What is the main advantage of mini-batch gradient descent over standard (batch) gradient descent?

<div class="upper-alpha" markdown>
1. It always finds the global minimum instead of local minima
2. It completely eliminates the need for a validation set
3. It balances computational efficiency with gradient stability
4. It eliminates the need to choose a learning rate
</div>

??? question "Show Answer"
    The correct answer is **C**.

    Mini-batch gradient descent updates weights using small batches of samples (e.g., 32, 64, 128) rather than the entire dataset (batch gradient descent) or single samples (stochastic gradient descent). This provides several advantages: more stable gradient estimates than SGD, faster updates than full batch, efficient matrix operations that leverage GPU parallelization, and reduced gradient variance. It doesn't guarantee global minima (option A), still requires validation (option B), and definitely still requires learning rate tuning (option D), but it offers an excellent practical balance.

    **Concept Tested:** Mini-Batch Gradient Descent, Gradient Descent, Batch Size

---

#### 8. A neural network's training loss continues decreasing but validation loss starts increasing after epoch 50. What is this phenomenon called, and what should be done?

<div class="upper-alpha" markdown>
1. Vanishing gradients; use ReLU activation
2. Underfitting; add more layers or neurons
3. Overfitting; apply regularization or early stopping
4. Exploding gradients; apply gradient clipping
</div>

??? question "Show Answer"
    The correct answer is **C**.

    When training loss decreases but validation loss increases, the model is overfitting—memorizing training data rather than learning generalizable patterns. The divergence indicates the model performs well on training data but poorly on unseen validation data. Solutions include regularization techniques (dropout, L1/L2 penalties), early stopping (halt training when validation loss stops improving), reducing model complexity, or augmenting training data. Epoch 50 would be a good point to stop training and restore the weights from earlier when validation loss was lowest.

    **Concept Tested:** Overfitting, Early Stopping, Regularization

---

#### 9. Why is He initialization preferred over Xavier initialization when using ReLU activations?

<div class="upper-alpha" markdown>
1. He initialization produces larger initial weights
2. He initialization accounts for ReLU's property of zeroing negative values
3. He initialization only works with ReLU, while Xavier works with all activations
4. He initialization is faster to compute
</div>

??? question "Show Answer"
    The correct answer is **B**.

    He initialization uses variance $\frac{2}{n_{in}}$ while Xavier uses $\frac{2}{n_{in} + n_{out}}$. The key difference is that He initialization accounts for ReLU setting all negative values to zero, effectively "killing" half the neurons' outputs on average. To maintain appropriate signal propagation despite this, He initialization uses larger variance (factor of 2 instead of Xavier's averaging). Xavier was designed for symmetric activations like tanh and sigmoid. Using Xavier with ReLU can lead to diminishing activations in deep networks, while He initialization maintains healthy signal propagation.

    **Concept Tested:** He Initialization, Xavier Initialization, Weight Initialization, ReLU

---

#### 10. A deep neural network (15 layers) using sigmoid activation functions trains very slowly, with early layers barely updating. What is the most likely cause?

<div class="upper-alpha" markdown>
1. Exploding gradients
2. Vanishing gradients
3. Learning rate is too high
4. Batch size is too small
</div>

??? question "Show Answer"
    The correct answer is **B**.

    The vanishing gradient problem occurs when gradients become extremely small as they propagate backward through many layers. Sigmoid functions saturate (gradient ≈ 0) for large positive or negative inputs. In a 15-layer network, gradients are multiplied through all these saturating functions during backpropagation, resulting in exponentially shrinking gradients. Early layers receive nearly zero gradient and barely update. Solutions include using ReLU activations (which don't saturate for positive values), skip connections, batch normalization, or proper weight initialization. This is why sigmoid has been largely replaced by ReLU in deep networks.

    **Concept Tested:** Vanishing Gradient, Sigmoid, Backpropagation, Deep Learning
