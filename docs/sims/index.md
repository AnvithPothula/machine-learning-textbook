# Interactive MicroSims

This textbook includes 18 interactive MicroSims designed to enhance your understanding of machine learning concepts through visualization and hands-on exploration.

## What are MicroSims?

MicroSims are lightweight, browser-based interactive simulations that allow you to:

- **Visualize** complex algorithms and mathematical concepts
- **Experiment** with parameters and observe real-time changes
- **Explore** different scenarios and edge cases
- **Understand** abstract concepts through concrete examples

All MicroSims run directly in your browser with no installation required.

---

## MicroSims by Chapter

### Chapter 2: K-Nearest Neighbors

- **[Distance Metrics](distance-metrics/index.md)** - Compare Euclidean vs Manhattan distance calculations
- **[K-Selection Simulator](k-selection-simulator/index.md)** - Explore how different k values affect classification boundaries

### Chapter 3: Decision Trees

- **[Entropy-Gini Comparison](entropy-gini-comparison/index.md)** - Visualize impurity metrics for splitting criteria

### Chapter 4: Logistic Regression

- **[Sigmoid Explorer](sigmoid-explorer/index.md)** - Interactive sigmoid function transformation

### Chapter 5: Regularization

- **[Ridge Regression Geometry](ridge-regression-geometry/index.md)** - L2 regularization with circular constraint visualization
- **[Lasso Regression Geometry](lasso-regression-geometry/index.md)** - L1 regularization with diamond constraint visualization

### Chapter 6: Support Vector Machines

- **[SVM Margin Maximization](svm-margin-maximization/index.md)** - Interactive SVM decision boundary and margin visualization

### Chapter 8: Data Preprocessing

- **[Feature Scaling Visualizer](feature-scaling-visualizer/index.md)** - Compare Min-Max scaling vs Z-score standardization
- **[Categorical Encoding Explorer](categorical-encoding-explorer/index.md)** - Compare Label encoding vs One-Hot encoding

### Chapter 9: Neural Networks

- **[Network Architecture Visualizer](network-architecture-visualizer/index.md)** - Explore different neural network architectures
- **[Activation Functions](activation-functions/index.md)** - Compare Sigmoid, Tanh, ReLU, and Leaky ReLU

### Chapter 10: Convolutional Neural Networks

- **[Convolution Operation](convolution-operation/index.md)** - See how convolution filters slide over images
- **[CNN Architecture](cnn-architecture/index.md)** - Visualize different CNN architectures with data flow

### Chapter 11: Transfer Learning

- **[Training Validation Curves](training-validation-curves/index.md)** - Observe training vs validation loss over epochs

### Chapter 12: Model Evaluation

- **[Confusion Matrix Explorer](confusion-matrix-explorer/index.md)** - Interactive confusion matrix with metrics calculations
- **[ROC Curve Comparison](roc-curve-comparison/index.md)** - Compare ROC curves for different classifiers
- **[K-Fold Cross Validation](kfold-cross-validation/index.md)** - Visualize k-fold cross-validation partitioning

---

## Using MicroSims

Each MicroSim includes:

- **Interactive controls** - Sliders, buttons, and dropdowns to adjust parameters
- **Real-time visualization** - See changes immediately as you adjust controls
- **Educational context** - Descriptions and learning objectives
- **Lesson plans** - Suggested activities and discussion questions

### How to Use

1. **Read the description** - Understand what the MicroSim demonstrates
2. **Experiment with controls** - Try different parameter values
3. **Observe the changes** - See how the visualization responds
4. **Reflect on patterns** - Think about what you're observing
5. **Apply your learning** - Connect concepts to chapter content

---

## Technical Details

All MicroSims are built using:

- **[p5.js](https://p5js.org/)** - For interactive visualizations and animations
- **[Chart.js](https://www.chartjs.org/)** - For data charts and plots
- **[vis-network](https://visjs.org/)** - For network diagrams

MicroSims are:

- **Width-responsive** - Adapt to your screen size
- **Accessible** - Work on desktop, tablet, and mobile
- **Open source** - View the code and learn from it
- **Embeddable** - Can be used in other educational contexts

---

## Embedding MicroSims

Educators can embed MicroSims in their own materials using iframes:

```html
<iframe src="https://your-site.github.io/sims/[microsim-name]/main.html"
        height="800px"
        width="100%"
        scrolling="no">
</iframe>
```

Replace `[microsim-name]` with the specific MicroSim directory name.

---

## Feedback

Have suggestions for improving existing MicroSims or ideas for new ones? We welcome your feedback!

**Total MicroSims**: 18 interactive visualizations
