---
title: Convolution Operation
description: Interactive visualization demonstrating how convolution filters slide over images to detect features like edges.
---

# Convolution Operation

<iframe src="main.html" height="772px" width="100%" scrolling="no"></iframe>

[Run the Convolution Operation MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## Description

This MicroSim demonstrates the fundamental convolution operation used in Convolutional Neural Networks (CNNs). Watch as different filters slide over an input image (5×5 grid) to produce an output feature map (3×3 grid).

**Key Features:**

- **Multiple Filter Types**: Choose from Vertical Edge, Horizontal Edge, Blur, Sharpen, and Identity filters
- **Step-by-Step Animation**: See exactly how the filter moves across the image
- **Computation Details**: View the element-wise multiplication and summation for each position
- **Interactive Controls**: Adjust animation speed and toggle step visualization

**How Convolution Works:**

1. A small filter (kernel) slides across the input image
2. At each position, element-wise multiplication is performed
3. The products are summed to produce a single output value
4. The filter moves to the next position (with a stride of 1)
5. The complete feature map shows detected features

**Learning Objectives:**

- Understand how convolution filters detect patterns in images
- See the sliding window mechanism in action
- Learn how different filters detect different features (edges, blur, etc.)
- Visualize the reduction in spatial dimensions from input to output

## Lesson Plan

**Prerequisites**: Basic understanding of matrix operations and image representations

**Suggested Activities**:

1. Start with the Vertical Edge filter and observe which parts of the image produce strong responses
2. Switch to Horizontal Edge and compare the feature map
3. Try the Blur filter to see how it smooths the image
4. Enable "Show Steps" to understand the computation at each position
5. Experiment with different animation speeds to better observe the sliding window

**Discussion Questions**:

- Why does the output (3×3) have smaller dimensions than the input (5×5)?
- How do different filters detect different types of features?
- What happens when the filter slides over a region with no edges?
- How might padding affect the output dimensions?

## You can include this MicroSim on your website

Use the following iframe code:

```html
<iframe src="https://your-site.github.io/sims/convolution-operation/main.html" height="772px" width="100%" scrolling="no"></iframe>
```

## Technical Details

- **Library**: p5.js 1.11.10
- **Canvas**: Width-responsive, 650px drawing area + 120px controls
- **Filter Size**: 3×3 (standard convolution kernel)
- **Stride**: 1 (filter moves one pixel at a time)
- **Padding**: None (valid convolution)
