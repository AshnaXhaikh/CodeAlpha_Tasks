# Handwritten Digit Classification using CNN

**Organization:** CodeAlpha  
**Internship Project:** Image Classification  

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from grayscale images. The goal is to accurately predict the digit (0–9) for each input image using deep learning techniques.

---
View My Live App: [CNN_Digit_Classifier](https://huggingface.co/spaces/ashnaxhaikh/CNN_DigitClassifier)
---

## Dataset
- **Type:** Grayscale images of size 28x28  
- **Target:** Digit class labels (0–9)  
- **Splits:** Training and testing datasets  

---

## Methodology

1. **Data Preprocessing**  
   - Normalize pixel values to [0,1] range.  
   - Cast labels to integer type (`int64`).  
   - Shuffle and batch the dataset for training.  

2. **Model Architecture**  
   - **Input:** 28x28 grayscale images  
   - **Conv2D layers:** Extract spatial features  
     - Conv2D(32) → MaxPooling(2x2)  
     - Conv2D(64) → MaxPooling(2x2)  
     - Conv2D(64)  
   - **Dense layers:** Flatten → Dense(64) → Dense(10, softmax)  

3. **Training**  
   - Optimizer: Adam  
   - Loss: Sparse Categorical Crossentropy  
   - Metrics: Accuracy  
   - Epochs: 5  

4. **Evaluation & Visualization**  
   - Predict on test data batch  
   - Display first 9 images with predicted and true labels  

---

## Results
- Model achieves good classification accuracy on test data.  
- Visualization shows correct predictions for sample digits.  

---

## Conclusion
This CNN-based approach provides an effective method for handwritten digit recognition. Further improvements can be achieved with data augmentation, more epochs, or deeper architectures.
