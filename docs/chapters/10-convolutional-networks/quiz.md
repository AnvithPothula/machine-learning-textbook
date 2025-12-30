# Quiz: Convolutional Neural Networks

Test your understanding of convolutional neural networks with these questions.

---

#### 1. What is the primary advantage of using convolutional layers over fully connected layers for image processing?

<div class="upper-alpha" markdown>
1. Convolutional layers require more parameters, leading to better accuracy
2. Convolutional layers preserve spatial structure through local connectivity and weight sharing
3. Convolutional layers can only process grayscale images
4. Convolutional layers eliminate the need for activation functions
</div>

??? question "Show Answer"
    The correct answer is **B**. Convolutional layers preserve spatial structure by connecting neurons only to small local regions (local connectivity) and using the same weights across the entire image (weight sharing). This dramatically reduces parameters compared to fully connected layers while maintaining spatial relationships between pixels. For example, a fully connected layer on a 224×224 image with 1,000 neurons requires over 150 million weights, while a convolutional layer with 64 filters of size 3×3 requires only 576 weights.

    **Concept Tested:** Convolutional Neural Network, Local Connectivity, Weight Sharing

---

#### 2. Given a 7×7 input image and a 3×3 filter with stride 2 and valid padding, what is the output size?

<div class="upper-alpha" markdown>
1. 5×5
2. 3×3
3. 7×7
4. 2×2
</div>

??? question "Show Answer"
    The correct answer is **B**. Using the output size formula: $\lfloor \frac{n - k}{s} + 1 \rfloor$ where n=7 (input size), k=3 (filter size), and s=2 (stride), we get: $\lfloor \frac{7 - 3}{2} + 1 \rfloor = \lfloor \frac{4}{2} + 1 \rfloor = \lfloor 2 + 1 \rfloor = 3$. Therefore, the output is 3×3.

    **Concept Tested:** Convolution Operation, Stride, Valid Padding

---

#### 3. What type of padding should you use to maintain the same spatial dimensions through multiple convolutional layers when using stride 1?

<div class="upper-alpha" markdown>
1. Valid padding
2. Same padding
3. Zero padding
4. Reflective padding
</div>

??? question "Show Answer"
    The correct answer is **B**. Same padding adds enough zeros around the border so that the output has the same spatial dimensions as the input when using stride 1. For a k×k filter, same padding requires p = (k-1)/2 pixels of padding on all sides. For example, a 3×3 filter needs 1 pixel of padding, and a 5×5 filter needs 2 pixels. Valid padding provides no padding and shrinks the output, while "zero padding" and "reflective padding" are not standard CNN terminology for this purpose.

    **Concept Tested:** Same Padding, Padding

---

#### 4. In a CNN, what does the receptive field of a neuron in layer 3 represent?

<div class="upper-alpha" markdown>
1. The number of filters in that layer
2. The size of the feature map at that layer
3. The region of the input image that affects that neuron's activation
4. The stride used in that layer
</div>

??? question "Show Answer"
    The correct answer is **C**. The receptive field is the region of the input image that influences a particular neuron's activation. As you go deeper in the network, receptive fields grow larger. For example, a layer 1 neuron with a 3×3 filter has a 3×3 receptive field, but a layer 2 neuron receiving from a 3×3 region of layer 1 would have a 5×5 receptive field in the input, and a layer 3 neuron would have a 7×7 receptive field. This allows deep CNNs to capture increasingly global context while maintaining computational efficiency.

    **Concept Tested:** Receptive Field, Spatial Hierarchies

---

#### 5. What is the primary purpose of max pooling layers in CNNs?

<div class="upper-alpha" markdown>
1. To increase the spatial dimensions of feature maps
2. To provide translation invariance and reduce computational complexity
3. To add more learnable parameters to the network
4. To replace activation functions like ReLU
</div>

??? question "Show Answer"
    The correct answer is **B**. Max pooling downsamples feature maps by taking the maximum value in local regions, typically using 2×2 windows with stride 2 (which halves spatial dimensions). This provides two critical benefits: translation invariance (small shifts in the input don't significantly change the output) and computational efficiency (reduces dimensions for downstream layers). Max pooling does not add learnable parameters—it's a fixed operation—and it complements rather than replaces activation functions.

    **Concept Tested:** Max Pooling, Translation Invariance

---

#### 6. You're designing a CNN for 32×32 RGB images. Your first convolutional layer uses 64 filters of size 5×5 with valid padding and stride 1. How many parameters does this layer have (including biases)?

<div class="upper-alpha" markdown>
1. 1,600
2. 4,800
3. 4,864
4. 65,536
</div>

??? question "Show Answer"
    The correct answer is **C**. Each filter has 5×5×3 = 75 weights (5×5 spatial dimensions × 3 input channels for RGB) plus 1 bias term = 76 parameters per filter. With 64 filters, the total is 64 × 76 = 4,864 parameters. This demonstrates the parameter efficiency of CNNs: despite processing the entire 32×32×3 = 3,072-dimensional input, we only need 4,864 parameters due to weight sharing and local connectivity.

    **Concept Tested:** Filter, Weight Sharing, Local Connectivity

---

#### 7. Which architectural innovation was introduced by ResNet to enable training of very deep networks (100+ layers)?

<div class="upper-alpha" markdown>
1. Using only 3×3 filters
2. Skip connections (residual connections)
3. Inception modules with parallel paths
4. Aggressive data augmentation
</div>

??? question "Show Answer"
    The correct answer is **B**. ResNet introduced skip connections (also called residual connections) that add the input x directly to the output: H(x) = F(x) + x. This allows gradients to flow directly through the network via the skip path, alleviating vanishing gradients and making it easier to optimize very deep networks. ResNet-152 successfully trained 152 layers and achieved 3.6% error on ImageNet in 2015, surpassing human-level performance (~5% error). VGG introduced using only 3×3 filters, Inception introduced parallel paths, and AlexNet emphasized data augmentation.

    **Concept Tested:** ResNet, CNN Architecture

---

#### 8. In the context of CNNs, what hierarchical pattern of features typically emerges across layers?

<div class="upper-alpha" markdown>
1. Complex objects → object parts → edges
2. Edges → textures/parts → complete objects
3. All layers learn similar edge detectors
4. Random features with no hierarchical structure
</div>

??? question "Show Answer"
    The correct answer is **B**. CNNs learn spatial hierarchies of features where early layers detect simple patterns like edges, colors, and textures; middle layers combine these into object parts like eyes, wheels, or corners; and deep layers recognize complete objects, faces, or scenes. This hierarchical organization mirrors biological vision systems and emerges naturally from the training process without explicit programming. Visualization studies consistently show this progression from simple to complex features across CNN depth.

    **Concept Tested:** Spatial Hierarchies, CNN Architecture

---

#### 9. Your CNN is overfitting on a small image dataset. Which data augmentation technique would be LEAST appropriate for natural images?

<div class="upper-alpha" markdown>
1. Random horizontal flips
2. Random vertical flips
3. Random crops with padding
4. Color jittering (brightness/contrast adjustment)
</div>

??? question "Show Answer"
    The correct answer is **B**. Random vertical flips are generally inappropriate for natural images because most objects have a consistent orientation (e.g., animals, vehicles, buildings). Flipping an image vertically creates unrealistic training examples that don't match real-world test data. Horizontal flips are appropriate because objects can appear on either side of an image. Random crops, scaling, and color jittering all preserve the semantic content while providing useful variation to reduce overfitting.

    **Concept Tested:** Data Augmentation

---

#### 10. Which CNN architecture introduced the concept of multi-scale feature extraction through parallel convolutional paths with different filter sizes in the same layer?

<div class="upper-alpha" markdown>
1. LeNet
2. AlexNet
3. VGG
4. Inception (GoogLeNet)
</div>

??? question "Show Answer"
    The correct answer is **D**. Inception (also known as GoogLeNet) introduced the Inception module, which applies parallel convolutional paths with different filter sizes (1×1, 3×3, 5×5) and max pooling simultaneously, then concatenates the outputs. This captures patterns at multiple scales in a single layer. The architecture also uses 1×1 "bottleneck" convolutions to reduce dimensionality before expensive operations, achieving computational efficiency. Despite having only 5 million parameters (vs. 60M for AlexNet), Inception won ImageNet 2014 with 6.7% error.

    **Concept Tested:** Inception, CNN Architecture
