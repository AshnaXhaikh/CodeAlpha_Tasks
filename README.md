# Internship Projects – Machine Learning 

**Organization:** CodeAlpha  
**Scope:** Multiple projects covering credit scoring, audio emotion recognition, image classification, and disease prediction.

---

## 1. Credit Scoring Prediction

**Objective:** Predict whether a credit card user will **default on payment**.  
**Goal:** Maximize recall for defaulters (class 1) and deploy the model as a **Streamlit web app**.

### Dataset
- Customer financial information: payment history, credit limit, bill amounts, demographic info.

### Methodology
1. Data Preprocessing: Handle missing values, scale features, split into train/test sets.  
2. EDA: Countplots, scatter plots, histograms, heatmaps to explore patterns.  
3. Model Training: Logistic Regression, Decision Tree, Random Forest (best).  
4. Threshold Tuning: Adjusted classification threshold (0.31) to improve recall for defaulters.  
5. Evaluation: Confusion matrix, precision, recall, F1-score, ROC-AUC.  
6. Deployment: Interactive Streamlit app for real-time predictions.

[View Live App](https://huggingface.co/spaces/ashnaxhaikh/Credit_Scoring)

---

## 2. Audio Emotion Recognition Using CNN (TESS Dataset)

**Objective:** Classify emotions (happy, sad, angry, etc.) from audio recordings.  

### Dataset
- **TESS (Toronto Emotional Speech Set)**: Actors speaking sentences with various emotions.  
- Features: MFCC (Mel-Frequency Cepstral Coefficients) extracted from audio files.

### Methodology
1. Extract and pad MFCC features to uniform size.  
2. Train CNN to classify emotions.  
3. Save model weights (`emotion_cnn.h5`) and label encoder (`label_encoder.pkl`).  
4. Deploy via Streamlit or Hugging Face Spaces for real-time predictions.

[View Live App](https://huggingface.co/spaces/ashnaxhaikh/Speech-Emotion-Recognition)

---

## 3. Handwritten Digit Classification using CNN

**Objective:** Classify handwritten digits (0–9) from 28x28 grayscale images.  

### Dataset
- Grayscale images of digits, split into training and testing sets.

### Methodology
1. Normalize pixel values and cast labels to integer type.  
2. CNN Architecture:  
   - Conv2D(32) → MaxPooling2D(2x2)  
   - Conv2D(64) → MaxPooling2D(2x2)  
   - Conv2D(64) → Flatten → Dense(64) → Dense(10, softmax)  
3. Compile & Train: Adam optimizer, sparse categorical crossentropy, 5 epochs.  
4. Evaluation: Accuracy metrics and visualization of predictions for sample images.

[View Live App](https://huggingface.co/spaces/ashnaxhaikh/CNN_DigitClassifier)

---

## 4. Disease Prediction from Medical Data

**Objective:** Predict likelihood of diseases using structured medical datasets.  
**Datasets & Projects:**
1. **Heart Disease Prediction** – UCI Heart Disease Dataset (Cleveland subset)  
   - Features: Age, sex, chest pain type, resting BP, cholesterol, etc.  
   - Key Insights: Chest pain type (`cp`) and max heart rate (`thalach`) most predictive.  

2. **Breast Cancer Prediction** – Breast Cancer Wisconsin (Diagnostic) Dataset  
   - Features: 30 tumor cell nuclei measurements.  
   - Key Insights: SVM achieved highest precision; mean radius, texture, perimeter important.  

3. **Diabetes Prediction** – Pima Indians Diabetes Dataset  
   - Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age.  
   - Key Insights: XGBoost performed best; Glucose, BMI, Age most predictive.  

### Methodology (Common)
- Data Preprocessing: Handle missing/invalid values, train-test split, scaling.  
- EDA: Feature distributions, correlations, boxplots.  
- Model Training: Logistic Regression, SVM, Random Forest, XGBoost.  
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- Feature Importance & ROC Curves for interpretability.

### Conclusion
Tree-based models generally performed best for medical datasets. Feature selection, scaling, and hyperparameter tuning were key for improving model performance.

---

## Overall Internship Outcomes

- Hands-on experience with **structured and unstructured data**.  
- Developed **classification models** for tabular, image, and audio data.  
- Deployment experience via **Streamlit & Hugging Face Spaces**.  
- Focused on **model evaluation, threshold tuning, and feature importance**.  
