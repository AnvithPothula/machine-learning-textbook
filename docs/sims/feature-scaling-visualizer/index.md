# Feature Scaling Visualizer

<iframe src="main.html" width="100%" height="750px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](main.html){: target="_blank" .md-button }

## Description

An interactive visualization comparing min-max scaling and z-score standardization across different data distributions.

## Learning Objectives

- Understand how min-max scaling transforms data to the [0, 1] range
- Compare z-score standardization that creates mean=0, std=1 distributions
- Observe how outliers affect each scaling method differently
- Recognize when to use each scaling technique

## How to Use

1. **Select Distribution**: Choose from Normal, Skewed, With Outliers, or Bimodal distributions
2. **Adjust Parameters**: Use sliders to change sample size, mean, and standard deviation
3. **Add Outliers**: Click the button to add outliers and observe their impact
4. **Compare Results**: Examine histograms, box plots, and statistics across all three panels

## Key Concepts

### Min-Max Scaling
- Transforms to [0, 1] range
- Formula: x' = (x - min) / (max - min)
- Highly sensitive to outliers
- Preserves distribution shape

### Z-Score Standardization
- Transforms to mean=0, std=1
- Formula: x' = (x - μ) / σ
- Less sensitive to outliers
- Assumes approximately Gaussian distribution

## Interactive Features

- **Distribution Types**: Compare scaling effects on different data shapes
- **Live Statistics**: View mean, std, min, max for each transformation
- **Box Plots**: Visualize quartiles and outliers
- **Histograms**: See the distribution shape before and after scaling

## Related Concepts

- [Normalization](../../chapters/08-data-preprocessing/index.md#min-max-scaling-normalization)
- [Standardization](../../chapters/08-data-preprocessing/index.md#z-score-normalization-standardization)
- [Data Preprocessing](../../chapters/08-data-preprocessing/index.md)
