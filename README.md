# Foot Contact Analysis  

## Project Overview  
Foot contact estimation is a fundamental problem in biomechanics with applications in sports science, rehabilitation, robotics, and motion analysis. This project leverages a 20GB movement dataset to classify foot contact states using advanced deep learning methodologies. By iteratively developing models and incorporating cutting-edge techniques, the project achieves state-of-the-art accuracy of 94%, utilizing Vision Transformers (ViT).  

---

## Key Features  
1. **Extensive Dataset Utilization**:  
   - Employed a 20GB dataset comprising motion sequences annotated with foot contact states such as heel contact, mid-stance, and toe-off.  
   - Implemented preprocessing techniques to standardize data, improve quality, and create training-ready samples.  

2. **Model Development and Optimization**:  
   - Designed a custom **Multi-Layer Perceptron (MLP)** pipeline, achieving 71% accuracy.  
   - Applied **Transfer Learning** with pre-trained CNNs like VGG16 and ResNet50, boosting accuracy to 90%.  

3. **Vision Transformer (ViT) Implementation**:  
   - Leveraged attention mechanisms for spatial and temporal feature extraction.  
   - Customized patch extraction tailored for motion data, culminating in a peak accuracy of 94%.  

4. **Innovative Problem Solving**:  
   - Tackled challenges of data size, noise, and model generalization.  
   - Optimized training pipelines for scalability and efficiency.  

---

## Methodology  

### 1. **Dataset Preparation**  
- **Source**: A high-quality dataset comprising motion-capture sequences annotated with foot contact labels.  
- **Preprocessing**:  
  - Resampled data for consistency in frame rates.  
  - Applied normalization to align input values within a standard range.  
  - Augmented data using transformations like temporal jittering, adding robustness to the models.  
- **Split Ratio**: Divided into training (80%), validation (20%), and separate human dataset for testing to ensure balanced evaluation.  

---

### 2. **Model Development**  

#### **Baseline Model: Multi-Layer Perceptron (MLP)**  
- **Pipeline Design**:  
  - Flattened motion data into feature vectors.  
  - Constructed a model with multiple dense layers, ReLU activation, and dropout regularization.  
  - Output layer with softmax activation for classification.  
- **Outcome**: Achieved a baseline accuracy of 71%, setting a benchmark for advanced models.  

#### **Transfer Learning with Convolutional Neural Networks (CNNs)**  
- **Why CNNs?** Spatial features like pressure regions are critical for foot contact estimation, making CNNs a natural choice.  
- **Implementation**:  
  - Integrated VGG16 and ResNet50 as feature extractors.  
  - Fine-tuned layers closer to the output for task-specific learning.  
  - Added custom fully connected layers for enhanced classification performance.  
- **Result**: Achieved an accuracy of 90%, significantly improving over the baseline.  

#### **Vision Transformers (ViT)**  
- **Key Features**:  
  - Processed data as patches, treating them as input sequences.  
  - Employed self-attention mechanisms to capture interdependencies between patches.  
  - Customized patch size and encoding techniques for motion-specific data.  
- **Impact**: Delivered a state-of-the-art accuracy of 94%, outperforming CNN-based methods.  

---

## Results  
| **Model**                  | **Accuracy** |  
|-----------------------------|--------------|  
| GPT-4 API Baseline          | 65%          |  
| Custom MLP                  | 71%          |  
| Transfer Learning (CNNs)    | 90%          |  
| Vision Transformer (ViT)    | 94%          |  

---

## Challenges and Solutions  

1. **Large Dataset Size**:  
   - Challenge: Managing 20GB of motion data was computationally expensive.  
   - Solution: Implemented batch processing and efficient data augmentation pipelines.  

2. **Overfitting Risks**:  
   - Challenge: Training deep networks on limited annotations could lead to overfitting.  
   - Solution: Used dropout, batch normalization, and early stopping techniques.  

3. **Model Adaptation**:  
   - Challenge: Adapting generic models (e.g., VGG16) for foot contact analysis.  
   - Solution: Fine-tuned layers and customized architectures for domain-specific learning.  

4. **Resource Constraints**:  
   - Challenge: Limited computational resources for training large models.  
   - Solution: Utilized transfer learning and Vision Transformer optimizations to maximize efficiency.  

---

## Future Directions  

1. **Real-Time Deployment**: Develop pipelines for live foot contact estimation in sports or rehabilitation scenarios.  
2. **Cross-Dataset Validation**: Test models on other motion datasets to ensure robustness and generalizability.  
3. **Lightweight Models**: Optimize models for deployment on edge devices, like wearables.  
4. **Hybrid Architectures**: Combine CNNs and transformers to explore synergistic effects in feature extraction.  

---


## Acknowledgments  
- **Pre-trained Models**: VGG16 and ResNet50 from TensorFlow/Keras libraries.  
- **Vision Transformers**: Hugging Face’s ViT library.  
- **Dataset Providers**: Motion dataset contributors for their valuable resource.  
