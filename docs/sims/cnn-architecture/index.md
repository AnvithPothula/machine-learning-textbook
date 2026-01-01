---
title: CNN Architecture Visualizer
description: Interactive visualization showing the layer structure and data flow through different CNN architectures.
---

# CNN Architecture Visualizer

<iframe src="main.html" height="752px" width="100%" scrolling="no"></iframe>

[Run the CNN Architecture MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## Description

This MicroSim visualizes the architecture of Convolutional Neural Networks (CNNs), showing how data flows through different types of layers. Explore multiple architecture styles from simple CNNs to modern designs like VGG and ResNet.

**Key Features:**

- **Multiple Architectures**: Choose from Simple CNN, VGG-like, and ResNet Block architectures
- **Layer Visualization**: See the relative size and depth of each layer type
- **Animated Data Flow**: Watch data propagate through the network
- **Layer Details**: View dimensions, filter counts, and neuron counts
- **Color-Coded Layers**: Different colors for different layer types (conv, pool, FC, etc.)

**Architecture Components:**

- **Input Layer** (Blue): Raw image data (H × W × C)
- **Convolutional Layers** (Green): Feature extraction with learnable filters
- **Pooling Layers** (Orange): Downsampling to reduce spatial dimensions
- **Fully Connected Layers** (Purple): Classification and decision making
- **Output Layer** (Red): Final predictions
- **Add Layer** (Cyan): Skip connections in ResNet architectures

**Learning Objectives:**

- Understand the sequential structure of CNN architectures
- See how spatial dimensions change through layers
- Learn the difference between feature extraction and classification stages
- Compare different architectural patterns (simple, VGG-style, ResNet)

## Lesson Plan

**Prerequisites**: Basic understanding of neural networks and image processing

**Suggested Activities**:

1. Start with "Simple CNN" to understand the basic flow
2. Observe how dimensions change from input to output
3. Switch to "VGG-like" to see deeper architectures
4. Compare the number of parameters in different architectures
5. Explore "ResNet Block" to understand skip connections
6. Enable data flow animation to visualize forward pass

**Discussion Questions**:

- Why do spatial dimensions decrease through the network?
- How does the number of channels (depth) change?
- What is the purpose of pooling layers?
- Why do we use fully connected layers at the end?
- How do skip connections in ResNet help with training?

## You can include this MicroSim on your website

Use the following iframe code:

```html
<iframe src="https://your-site.github.io/sims/cnn-architecture/main.html" height="752px" width="100%" scrolling="no"></iframe>
```

## Technical Details

- **Library**: p5.js 1.11.10
- **Canvas**: Width-responsive, 650px drawing area + 100px controls
- **Architectures**: Simple CNN, VGG-like, ResNet Block
- **Layer Types**: Input, Convolution, Pooling, Fully Connected, Output, Add
