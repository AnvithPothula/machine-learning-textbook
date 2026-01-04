# Neural Network Architecture Visualizer

<iframe src="main.html" width="100%" height="750px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](main.html){: target="_blank" .md-button }

## Description

An interactive tool for building and visualizing different neural network architectures.

## Learning Objectives

- Explore how network architecture affects capacity and complexity
- Visualize how neurons connect across layers in fully connected networks
- Understand the role of depth (layers) and width (neurons per layer)
- Observe forward propagation through the network layers

## How to Use

1. **Adjust Layer Sizes**: Use sliders to change neurons in hidden layers
2. **Select Presets**: Choose common architectures (shallow, deep, wide)
3. **Animate**: Click to see forward propagation in action
4. **Observe**: Watch parameter count and connections update dynamically

## Key Concepts

### Network Depth
- Number of layers in the network
- Deeper networks learn hierarchical features
- Each layer can transform representations

### Network Width
- Number of neurons per layer
- Wider layers capture more features simultaneously
- Trade-off: capacity vs overfitting risk

### Parameters
- Total learnable weights and biases
- Formula: (inputs + 1) × outputs per layer
- More parameters = more capacity but higher risk of overfitting

## Architecture Types

- **Shallow**: Few layers, may underfit complex patterns
- **Deep**: Many layers, learns hierarchical representations
- **Wide**: Many neurons per layer, high capacity per level

## Interactive Features

- **Real-time Updates**: See architecture change as you adjust sliders
- **Parameter Counter**: Track total network parameters
- **Forward Propagation Animation**: Visualize activation flow
- **Connection Visualization**: Green = positive weights, red = negative weights

## Related Concepts

- [Neural Networks](../../chapters/09-neural-networks/index.md)
- [Hidden Layers](../../chapters/09-neural-networks/index.md#network-architecture)
- [Fully Connected Layers](../../chapters/09-neural-networks/index.md#network-architecture)
