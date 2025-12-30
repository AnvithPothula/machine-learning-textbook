# K Selection Interactive Simulator

An interactive visualization demonstrating how the choice of k affects KNN decision boundaries and prediction behavior.

## Learning Objectives

- Analyze how k value affects the bias-variance tradeoff in KNN
- Understand why k selection is critical for KNN performance
- Observe how different k values create different decision boundaries
- Recognize the relationship between k=1 and Voronoi diagrams

## How to Use

1. **Adjust k**: Use the slider to change the number of neighbors (1-25)
2. **Drag Test Point**: Click and drag the red test point to see predictions change
3. **Show Voronoi**: Toggle to visualize Voronoi cells when k=1
4. **Add Noise**: Add outlier points to see how noise affects small k values
5. **Reset**: Clear noise points and return to original data

## Key Concepts

### Small k (k=1)
- High variance, low bias
- Decision boundary follows training data closely
- Sensitive to noise and outliers
- Creates Voronoi diagram partitions

### Large k (k>20)
- Low variance, high bias
- Smooth decision boundaries
- May be too simple (underfit)
- Less affected by individual points

### Optimal k
- Balances bias and variance
- Often between 3-15 for many datasets
- Should be determined by cross-validation

## Interactive Features

- **Real-time Decision Boundaries**: See how k affects class regions
- **Neighbor Connections**: Lines show which points influence prediction
- **Vote Breakdown**: See exact vote counts for each class
- **Distance Display**: View distances to nearest neighbors
- **Warning Indicators**: Alerts for extreme k values

## Related Concepts

- [K Selection](../../chapters/02-k-nearest-neighbors/index.md#k-selection-choosing-the-right-number-of-neighbors)
- [Decision Boundaries](../../chapters/02-k-nearest-neighbors/index.md#decision-boundaries-and-voronoi-diagrams)
- [Voronoi Diagrams](../../chapters/02-k-nearest-neighbors/index.md#decision-boundaries-and-voronoi-diagrams)

<iframe src="k-selection-simulator.html" width="100%" height="800" frameborder="0"></iframe>
