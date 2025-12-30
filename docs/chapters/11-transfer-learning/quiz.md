# Quiz: Transfer Learning and Pre-Trained Models

Test your understanding of transfer learning and pre-trained models with these questions.

---

#### 1. What is the fundamental principle behind transfer learning in deep learning?

<div class="upper-alpha" markdown>
1. Training multiple models simultaneously on different tasks
2. Leveraging features learned from large-scale tasks to solve new tasks with less data
3. Transferring data from one domain to another before training
4. Using the same hyperparameters across all machine learning tasks
</div>

??? question "Show Answer"
    The correct answer is **B**. Transfer learning allows us to take knowledge learned from one task (typically on a large dataset like ImageNet) and apply it to a new, related task. The key insight is that features learned by deep networks on large-scale tasks—especially general features like edges, textures, and shapes in early layers—are often transferable to other tasks. This enables achieving excellent performance with far less data and computation than training from scratch. For example, a ResNet-50 trained on 1.2 million ImageNet images can be adapted to classify ants vs. bees with only 240 training images.

    **Concept Tested:** Transfer Learning, Pre-Trained Model

---

#### 2. When using a pre-trained model from ImageNet, why is it critical to normalize input images with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]?

<div class="upper-alpha" markdown>
1. These values maximize model accuracy on all datasets
2. The pre-trained weights expect inputs in this distribution from ImageNet training
3. These values prevent overfitting
4. PyTorch requires these specific values for all image data
</div>

??? question "Show Answer"
    The correct answer is **B**. Pre-trained ImageNet models were trained with inputs normalized using ImageNet's channel-wise statistics (mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]). The learned weights are optimized for this input distribution. If you preprocess your images differently, the pre-trained weights will receive inputs in an unexpected distribution, leading to poor performance. The normalization formula x' = (x - μ)/σ ensures your inputs match the distribution the model was trained on.

    **Concept Tested:** Pre-Trained Model, Transfer Learning

---

#### 3. What is the key difference between feature extraction and fine-tuning in transfer learning?

<div class="upper-alpha" markdown>
1. Feature extraction uses a pre-trained model while fine-tuning trains from scratch
2. Feature extraction freezes the base network layers, while fine-tuning allows them to update
3. Feature extraction requires more data than fine-tuning
4. Feature extraction only works with ImageNet models
</div>

??? question "Show Answer"
    The correct answer is **B**. In feature extraction, all convolutional layers are frozen (requires_grad=False), and only a new classification head is trained. This treats the pre-trained model as a fixed feature extractor. In fine-tuning, all layers (or a subset of later layers) are trainable, allowing the network to adapt its features to the new task. Feature extraction is faster, requires less memory, and is less prone to overfitting with small datasets, while fine-tuning typically achieves higher accuracy by adapting the entire network to your specific domain.

    **Concept Tested:** Feature Extraction, Fine-Tuning

---

#### 4. You're fine-tuning a ResNet-18 on a small custom dataset. Which learning rate would be most appropriate?

<div class="upper-alpha" markdown>
1. 0.1
2. 0.01
3. 0.001
4. 1.0
</div>

??? question "Show Answer"
    The correct answer is **C**. When fine-tuning pre-trained models, use small learning rates like 0.001 or 0.0001—much smaller than training from scratch (which typically uses 0.01 or 0.1). The pre-trained weights are already in a good region of the parameter space, so aggressive updates can destroy learned features. Small learning rates allow gentle adaptation to the new task while preserving useful pre-trained representations. Learning rates like 0.1 or 1.0 would likely cause the loss to diverge or performance to degrade.

    **Concept Tested:** Fine-Tuning, Optimizer

---

#### 5. In the ants vs. bees transfer learning example with 240 total training images, what validation accuracy was achieved using fine-tuning?

<div class="upper-alpha" markdown>
1. ~65%
2. ~75%
3. ~85%
4. ~95%
</div>

??? question "Show Answer"
    The correct answer is **D**. The example achieved approximately 94.77% validation accuracy with only 240 training images (120 per class) after just 5 epochs of fine-tuning. This demonstrates transfer learning's remarkable effectiveness with limited data—training from scratch on such a small dataset would likely achieve only 60-70% accuracy. The pre-trained ImageNet features provide such strong initialization that minimal task-specific training is needed.

    **Concept Tested:** Transfer Learning, Fine-Tuning

---

#### 6. What does the momentum parameter (typically set to 0.9) control in SGD with momentum?

<div class="upper-alpha" markdown>
1. The percentage of training data used in each batch
2. How much previous gradient history influences the current update
3. The probability of dropout during training
4. The rate at which learning rate decays
</div>

??? question "Show Answer"
    The correct answer is **B**. Momentum (β, typically 0.9) controls how much the accumulated velocity from previous gradients influences the current parameter update. The update rule is: v_{t+1} = β*v_t + ∇L(θ_t) and θ_{t+1} = θ_t - η*v_{t+1}. A momentum of 0.9 means 90% of the previous velocity is retained. This accelerates convergence in consistent gradient directions, dampens oscillations in fluctuating directions, and helps escape shallow local minima. Higher momentum (closer to 1.0) gives more weight to history; lower momentum (closer to 0) approaches standard SGD.

    **Concept Tested:** Momentum, Optimizer

---

#### 7. Your medical imaging task involves X-ray images that look very different from natural photos in ImageNet. What strategy would best address this domain shift?

<div class="upper-alpha" markdown>
1. Don't use transfer learning; train from scratch instead
2. Use transfer learning with domain-specific data augmentation and fine-tuning
3. Only use the ImageNet dataset without any custom data
4. Increase the learning rate to overcome domain differences
</div>

??? question "Show Answer"
    The correct answer is **B**. Domain adaptation techniques help bridge the gap when source and target domains differ significantly. For medical imaging, you should: (1) use transfer learning as initialization (low-level features like edges are still useful), (2) apply domain-specific augmentation (e.g., adding noise, simulating different imaging conditions), and (3) fine-tune with appropriate learning rates. Training from scratch wastes the opportunity to leverage useful low-level features and would require far more data. Simply increasing learning rate would destroy pre-trained features. Transfer learning with adaptation strategies typically outperforms both training from scratch and ignoring domain shift.

    **Concept Tested:** Domain Adaptation, Transfer Learning

---

#### 8. What problem does validation error help identify during transfer learning?

<div class="upper-alpha" markdown>
1. Whether the model architecture is appropriate
2. Whether the model is overfitting or underfitting
3. The optimal number of training epochs
4. All of the above
</div>

??? question "Show Answer"
    The correct answer is **D**. Validation error is crucial for multiple aspects of model development: (1) If both training and validation errors are high, the model is underfitting (try a larger model or more training). (2) If training error is low but validation error is high, the model is overfitting (use more data, augmentation, or early stopping). (3) Monitoring when validation error stops improving indicates the optimal number of epochs for early stopping. (4) Comparing validation error across different architectures helps with model selection. The validation set acts as a proxy for real-world performance without touching the test set.

    **Concept Tested:** Validation Error, Generalization

---

#### 9. In online learning scenarios with transfer learning, what is the main risk when fine-tuning on new data over time?

<div class="upper-alpha" markdown>
1. The model becomes too large to deploy
2. Catastrophic forgetting of previously learned knowledge
3. Increased training time with each update
4. Reduced accuracy on all data
</div>

??? question "Show Answer"
    The correct answer is **B**. Catastrophic forgetting occurs when fine-tuning on new data causes the model to forget previous knowledge, performing poorly on old examples. For example, if you initially trained on data from 2020 and then fine-tune only on 2021 data, performance on 2020 data may degrade significantly. Mitigation strategies include: (1) mixing old and new data during updates (e.g., 80% new, 20% old), (2) using smaller learning rates for online updates, and (3) applying regularization techniques like Elastic Weight Consolidation (EWC) that preserve important weights from previous training.

    **Concept Tested:** Online Learning, Fine-Tuning

---

#### 10. When performing feature extraction with a frozen ResNet-18, which parameter configuration would you use in PyTorch?

<div class="upper-alpha" markdown>
1. optimizer = optim.SGD(model.parameters(), lr=0.001)
2. optimizer = optim.SGD(model.fc.parameters(), lr=0.001)
3. optimizer = optim.Adam(model.features.parameters(), lr=0.001)
4. optimizer = optim.SGD(model.conv_layers.parameters(), lr=0.001)
</div>

??? question "Show Answer"
    The correct answer is **B**. For feature extraction, you freeze all base network layers (set requires_grad=False) and only optimize the new classification head. In ResNet, the final fully connected layer is called 'fc'. Therefore, you pass only model.fc.parameters() to the optimizer, ensuring only those weights are updated. Option A would try to update all parameters (but frozen parameters would be skipped). Options C and D use incorrect attribute names—ResNet doesn't have 'features' or 'conv_layers' attributes. This optimization strategy is much faster and more memory-efficient than fine-tuning the entire network.

    **Concept Tested:** Feature Extraction, Optimizer
