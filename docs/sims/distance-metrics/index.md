---
title: Distance Metrics Visualization
description: Interactive comparison of Euclidean vs Manhattan distance metrics showing geometric differences
---

# Distance Metrics Visualization

<iframe src="main.html" width="100%" height="680px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[Open Fullscreen](main.html){: target="_blank" .md-button }

## About This MicroSim

This interactive visualization demonstrates the geometric difference between two fundamental distance metrics used in machine learning: **Euclidean distance** (straight-line distance) and **Manhattan distance** (grid-based distance).

### How to Use

1. **Drag Point B**: Click and drag the blue point (B) anywhere on either side of the visualization
2. **Observe Distances**: Watch how both distance metrics update in real-time
3. **Compare Paths**: The green line shows Euclidean distance (straight), while the orange L-shape shows Manhattan distance (grid path)
4. **Reset**: Click the "Reset Point B" button to return to the default configuration

### Key Concepts

- **Euclidean Distance**: The straight-line distance between two points, calculated as √((x₂-x₁)² + (y₂-y₁)²)
- **Manhattan Distance**: The sum of absolute differences along each dimension, calculated as |x₂-x₁| + |y₂-y₁|
- **Ratio**: Manhattan distance is always ≥ Euclidean distance. When points align diagonally, the ratio equals √2 ≈ 1.414

### Educational Value

This visualization helps students understand:

- How different distance metrics measure "nearness" differently
- Why Manhattan distance is called "taxicab" or "city block" distance (follows grid paths)
- When each metric is appropriate (Euclidean for continuous space, Manhattan for grid-like data)
- How the choice of distance metric affects KNN algorithm behavior

## Learning Objectives

**Bloom's Taxonomy Level**: Understand (L2)

After using this MicroSim, students should be able to:

1. Explain the geometric difference between Euclidean and Manhattan distance
2. Calculate both distance metrics for given points
3. Understand when each metric is more appropriate for a given problem
4. Recognize that Manhattan distance is always at least as large as Euclidean distance

## Technical Details

- **Library**: p5.js
- **Responsive**: Fixed canvas size (800x600)
- **Interactivity**: Draggable point with real-time distance updates
- **Features**: Split-screen comparison, grid visualization, formula display

## Integration

To embed this MicroSim in your course materials:

```html
<iframe src="https://your-site.github.io/docs/sims/distance-metrics/main.html"
        width="100%" height="680px" scrolling="no"></iframe>
```
