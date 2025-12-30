# FAQ Coverage Gaps

**Generated:** 2025-12-29

This document identifies concepts from the learning graph that are not covered in the FAQ, prioritized by importance for student understanding.

## Summary

- **Total Concepts:** 200
- **Covered in FAQ:** 146 (73%)
- **Not Covered:** 54 (27%)

## Critical Gaps (High Priority)

These concepts have high centrality in the learning graph (many dependencies) or are fundamental to understanding covered topics. **Recommendation: Add questions for all 15 critical gaps.**

### Convolutional Neural Networks (8 concepts)

1. **VGG Architecture**
   - Centrality: High (referenced in transfer learning)
   - Category: Advanced Topics
   - Suggested Question: "What are the key characteristics of VGG architecture and why was it significant?"

2. **Inception Architecture**
   - Centrality: High (multi-scale feature learning)
   - Category: Advanced Topics
   - Suggested Question: "How does the Inception architecture use multiple filter sizes in parallel?"

3. **Depthwise Separable Convolution**
   - Centrality: Medium (efficiency technique)
   - Category: Technical Details
   - Suggested Question: "What is a depthwise separable convolution and why does it reduce parameters?"

4. **Dilated Convolution**
   - Centrality: Medium (receptive field expansion)
   - Category: Technical Details
   - Suggested Question: "What is dilated convolution and when is it useful?"

5. **Global Average Pooling**
   - Centrality: High (modern CNN standard)
   - Category: Technical Details
   - Suggested Question: "What is global average pooling and why is it preferred over fully connected layers?"

6. **Receptive Field**
   - Centrality: High (fundamental concept)
   - Category: Core Concepts
   - Suggested Question: "What is the receptive field in a CNN and how does it grow through layers?"

7. **Feature Pyramid**
   - Centrality: Medium (multi-scale detection)
   - Category: Advanced Topics
   - Suggested Question: "What is a feature pyramid network and why is it useful for object detection?"

8. **Spatial Pyramid Pooling**
   - Centrality: Low (specialized technique)
   - Category: Advanced Topics
   - Suggested Question: "What is spatial pyramid pooling and how does it handle variable input sizes?"

### Optimization (7 concepts)

9. **Learning Rate Scheduling**
   - Centrality: High (critical for training)
   - Category: Best Practices
   - Suggested Question: "What is learning rate scheduling and which schedules are most effective?"

10. **Nesterov Momentum**
    - Centrality: Medium (SGD variant)
    - Category: Technical Details
    - Suggested Question: "What is Nesterov momentum and how does it differ from standard momentum?"

11. **RMSprop**
    - Centrality: High (Adam predecessor)
    - Category: Technical Details
    - Suggested Question: "What is RMSprop and how does it adapt learning rates?"

12. **Weight Decay**
    - Centrality: High (regularization)
    - Category: Technical Details
    - Suggested Question: "What is weight decay and how is it related to L2 regularization?"

13. **Gradient Accumulation**
    - Centrality: Medium (memory efficiency)
    - Category: Best Practices
    - Suggested Question: "What is gradient accumulation and when should I use it?"

14. **Learning Rate Warmup**
    - Centrality: Medium (training stability)
    - Category: Best Practices
    - Suggested Question: "What is learning rate warmup and why does it help training?"

15. **Gradient Noise**
    - Centrality: Low (regularization technique)
    - Category: Advanced Topics
    - Suggested Question: "What is gradient noise and how can it improve generalization?"

## Medium Priority Gaps

These concepts are moderately important and would enhance FAQ completeness. **Recommendation: Add 8-10 questions from this list.**

### Neural Network Architectures (5 concepts)

16. **Residual Connections**
    - Suggested Question: "What are residual connections and why do they enable training of very deep networks?"

17. **Skip Connections**
    - Suggested Question: "How do skip connections help prevent vanishing gradients?"

18. **Highway Networks**
    - Suggested Question: "What are highway networks and how do they relate to ResNets?"

19. **Layer Normalization**
    - Suggested Question: "What is layer normalization and how does it differ from batch normalization?"

20. **Attention Mechanism**
    - Suggested Question: "What is an attention mechanism and how does it help neural networks focus on important features?"

### Support Vector Machines (5 concepts)

21. **Kernel Parameters (Gamma, C)**
    - Suggested Question: "How do I choose the C and gamma parameters for SVM with RBF kernel?"

22. **Nu-SVM**
    - Suggested Question: "What is nu-SVM and how does it differ from C-SVM?"

23. **One-Class SVM**
    - Suggested Question: "What is one-class SVM and when should I use it for anomaly detection?"

24. **SMO Algorithm**
    - Suggested Question: "What is the SMO (Sequential Minimal Optimization) algorithm for training SVMs?"

25. **Support Vector Details**
    - Suggested Question: "What exactly are support vectors and why are they important?"

### Data Preprocessing (3 concepts)

26. **Outlier Detection**
    - Suggested Question: "How do I detect and handle outliers in my dataset?"

27. **Label Encoding vs One-Hot**
    - Suggested Question: "When should I use label encoding vs one-hot encoding for categorical variables?"

28. **Data Imputation Strategies**
    - Suggested Question: "What are the best strategies for imputing missing values?"

### Regularization (1 concept)

29. **Elastic Net**
    - Suggested Question: "What is Elastic Net and when should I use it instead of L1 or L2 regularization?"

### Clustering (2 concepts)

30. **Silhouette Score**
    - Suggested Question: "What is the silhouette score and how does it help evaluate clustering quality?"

31. **Dendrogram**
    - Suggested Question: "What is a dendrogram and how is it used in hierarchical clustering?"

### Evaluation Metrics (4 concepts)

32. **Balanced Accuracy**
    - Suggested Question: "What is balanced accuracy and when should I use it instead of regular accuracy?"

33. **Matthews Correlation Coefficient**
    - Suggested Question: "What is the Matthews Correlation Coefficient and why is it good for imbalanced datasets?"

34. **Cohen's Kappa**
    - Suggested Question: "What is Cohen's Kappa and how does it measure inter-rater agreement?"

35. **Mean Average Precision**
    - Suggested Question: "What is mean average precision (mAP) and how is it used in object detection?"

### Transfer Learning (2 concepts)

36. **Domain Adaptation**
    - Suggested Question: "What is domain adaptation and how does it help transfer learning across different domains?"

37. **Model Zoo**
    - Suggested Question: "What is a model zoo and where can I find pre-trained models?"

### Foundation Concepts (3 concepts)

38. **Parametric vs Non-Parametric**
    - Suggested Question: "What is the difference between parametric and non-parametric models?"

39. **Instance-Based Learning**
    - Suggested Question: "What is instance-based learning and how does it differ from model-based learning?"

40. **Online Learning**
    - Suggested Question: "What is online learning and when is it preferred over batch learning?"

## Low Priority Gaps

These are advanced, specialized, or leaf-node concepts that can be addressed in future updates. **Recommendation: Address selectively based on user demand.**

### Specialized CNN Concepts (5 concepts)

41. **Object Detection** - Domain-specific application
42. **Semantic Segmentation** - Domain-specific application
43. **Instance Segmentation** - Domain-specific application
44. **Anchor Boxes** - Object detection specific
45. **Region Proposals** - Object detection specific

### Advanced Decision Trees (1 concept)

46. **Cost Complexity Pruning** - Specialized pruning technique

### Advanced Regularization (1 concept)

47. **Batch Renormalization** - Variant of batch norm

### Advanced Neural Networks (7 concepts)

48. **Gradient Checkpointing** - Memory optimization
49. **Mixed Precision Training** - Hardware optimization
50. **Distributed Training** - Scale-out technique
51. **Model Parallelism** - Large model training
52. **Data Parallelism** - Batch parallelization
53. **Knowledge Distillation** - Model compression
54. **Neural Architecture Search** - Automated design

## Implementation Recommendations

### Phase 1: Critical Gaps (Week 1)
Add 15 questions covering all critical gaps, focusing on:
- CNN architectures (VGG, Inception, receptive field, global average pooling)
- Optimization techniques (learning rate scheduling, RMSprop, weight decay, Nesterov momentum)

**Expected Impact:** Increase coverage from 73% to 81%

### Phase 2: Medium Priority (Weeks 2-3)
Add 10 questions from medium priority list, focusing on:
- Residual/skip connections
- SVM kernel tuning
- Advanced preprocessing
- Additional evaluation metrics

**Expected Impact:** Increase coverage from 81% to 86%

### Phase 3: Selective Low Priority (Week 4+)
Add 3-5 questions based on user feedback and most requested topics

**Expected Impact:** Increase coverage from 86% to 89%

## Coverage by Category After Implementation

| Category | Current | After Phase 1 | After Phase 2 | Target |
|----------|---------|---------------|---------------|--------|
| CNN | 55% | 75% | 80% | 80% |
| Optimization | 63% | 88% | 94% | 90% |
| Neural Networks | 70% | 73% | 86% | 85% |
| SVM | 69% | 69% | 94% | 90% |
| Overall | 73% | 81% | 86% | 85% |

## Questions with Highest Student Demand

Based on typical student inquiries in machine learning courses:

1. **Architecture Choice:** "How do I choose between different CNN architectures?"
2. **Learning Rate:** "What learning rate schedule should I use?"
3. **Kernel Tuning:** "How do I tune SVM kernel parameters?"
4. **Residual Networks:** "Why do residual connections help so much?"
5. **RMSprop vs Adam:** "When should I use RMSprop instead of Adam?"
6. **Outliers:** "How do I handle outliers in my data?"
7. **One-Class SVM:** "How do I use SVM for anomaly detection?"
8. **Global Average Pooling:** "Why use global average pooling instead of flatten?"
9. **Weight Decay:** "What's the relationship between weight decay and L2 regularization?"
10. **Receptive Field:** "How do I calculate receptive field in my CNN?"

## Conclusion

The FAQ covers 73% of concepts with strong coverage of fundamentals (90% foundation concepts) and complete coverage of some areas (100% KNN). The 27% gap consists primarily of:

1. **Advanced CNN architectures** (8 concepts) - Critical for modern computer vision
2. **Optimization techniques** (7 concepts) - Critical for effective training
3. **Advanced neural network concepts** (5 concepts) - Important for deep learning
4. **SVM tuning and variants** (5 concepts) - Important for practical SVM use

Adding the 15 critical gap questions would raise coverage to 81%, bringing the FAQ to "very good" completeness. The remaining gaps are primarily specialized techniques that can be addressed based on user demand.

**Priority Action:** Implement Phase 1 (15 questions on critical gaps) within 1 week to achieve 81% coverage.
