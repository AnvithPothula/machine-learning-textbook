# Activation Function Comparison

<iframe src="main.html" width="100%" height="850px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](main.html){: target="_blank" .md-button }

## Description

An interactive visualization comparing sigmoid, tanh, ReLU, and Leaky ReLU activation functions.

## Learning Objectives

- Compare shapes and output ranges of common activation functions
- Understand derivatives and gradient flow through different activations
- Recognize saturation regions and vanishing gradient problems
- Identify the dying neuron problem in ReLU and how Leaky ReLU addresses it

## How to Use

1. **Adjust x**: Slide to change the input value and see function outputs
2. **Show Derivatives**: Toggle to display derivative curves (dashed lines)
3. **Leaky ReLU α**: Adjust the negative slope parameter for Leaky ReLU
4. **Highlight Saturation**: Toggle to show saturation zones (yellow regions)
5. **Comparison Mode**: View all functions overlaid on a single plot

## Key Concepts

### Sigmoid
- Output range: [0, 1]
- Saturates at extremes (vanishing gradient)
- Used in binary classification output layers

### Tanh
- Output range: [-1, 1]
- Zero-centered (better than sigmoid)
- Still suffers from vanishing gradients

### ReLU (Rectified Linear Unit)
- Output range: [0, ∞)
- Fast computation, no vanishing gradient
- Can have "dying neurons" (stuck at zero)
- **Most common for hidden layers**

### Leaky ReLU
- Output range: (-∞, ∞)
- Small negative slope prevents dying neurons
- Combines ReLU benefits with gradient flow

## Interactive Features

- **2×2 Grid View**: Compare all four functions simultaneously
- **Comparison Mode**: Overlay all functions on one plot
- **Real-time Derivatives**: See gradient values at any input
- **Property Table**: Quick reference for key characteristics

## Related Concepts

- [Activation Functions](../../chapters/09-neural-networks/index.md#activation-functions)
- [Neural Networks](../../chapters/09-neural-networks/index.md)
