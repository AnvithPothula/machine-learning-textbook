# Categorical Encoding Explorer

<iframe src="main.html" width="100%" height="870px" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>

[View Fullscreen](main.html){: target="_blank" .md-button }

## Description

An interactive visualization comparing label encoding and one-hot encoding for categorical variables.

## Learning Objectives

- Understand how label encoding assigns integer values to categories
- Learn how one-hot encoding creates binary columns for each category
- Recognize the difference between nominal and ordinal variables
- Identify when to use each encoding method

## How to Use

1. **Select Example Dataset**: Choose from Default, Iris Species, or Car Types
2. **Toggle drop_first**: See how the parameter affects one-hot encoding dimensionality
3. **Add Rows**: Click to add more sample data rows
4. **Compare Encodings**: Examine the three tables showing original data, label encoding, and one-hot encoding

## Key Concepts

### Label Encoding
- Assigns integers to categories (0, 1, 2, ...)
- Memory-efficient (single column)
- Suitable for ordinal variables and target labels
- **Warning**: Introduces artificial ordering for nominal variables

### One-Hot Encoding
- Creates binary column for each category
- No artificial ordering imposed
- Required for nominal variables in most algorithms
- Use `drop_first=True` to avoid multicollinearity

## Interactive Features

- **Multiple Example Datasets**: See encoding on different data types
- **Live Comparison**: View all three representations simultaneously
- **Dimensionality Tracking**: See how column count changes
- **Pros/Cons Analysis**: Understand trade-offs for each method

## Related Concepts

- [Label Encoding](../../chapters/08-data-preprocessing/index.md#label-encoding)
- [One-Hot Encoding](../../chapters/08-data-preprocessing/index.md#one-hot-encoding)
- [Data Preprocessing](../../chapters/08-data-preprocessing/index.md)
