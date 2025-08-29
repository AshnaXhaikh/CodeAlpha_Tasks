# Disease Prediction from Medical Data

**Organization:** CodeAlpha  
**Task:** Task 4 – Disease Prediction from Medical Data  

## Overview
This repository contains three machine learning projects focused on predicting diseases from structured medical datasets. Each project implements data preprocessing, exploratory data analysis (EDA), model training, evaluation, and feature importance analysis. The goal is to identify the most effective models for accurate prediction of health conditions.

---

## Projects

### 1. Heart Disease Prediction
- **Dataset:** UCI Heart Disease Dataset (Cleveland subset)  
- **Features:** Age, sex, chest pain type (cp), resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels (ca), thalassemia type (thal)  
- **Target:** `target` (0 = no disease, 1 = disease)  
- **Models Used:** Logistic Regression, SVM, Random Forest, XGBoost  
- **Key Insights:** Chest pain type (`cp`) and maximum heart rate (`thalach`) are the most predictive features. Tree-based models performed best.

---

### 2. Breast Cancer Prediction
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Features:** 30 measurements of tumor cell nuclei (e.g., mean radius, texture, perimeter)  
- **Target:** `diagnosis` (Malignant = 1, Benign = 0)  
- **Models Used:** Logistic Regression, SVM, Random Forest, XGBoost  
- **Key Insights:** SVM achieved the highest precision and F1-score. Key predictive features include mean radius, texture, and perimeter.

---

### 3. Diabetes Prediction
- **Dataset:** Pima Indians Diabetes Dataset (UCI ML Repository)  
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
- **Target:** `Outcome` (0 = non-diabetic, 1 = diabetic)  
- **Models Used:** Logistic Regression, SVM, Random Forest, XGBoost  
- **Key Insights:** XGBoost performed best overall. Most important features: Glucose, BMI, Age, Insulin, Pregnancies.

---

## Methodology (Common Across Projects)
1. **Data Preprocessing**  
   - Handle missing or invalid values  
   - Split features (`X`) and target (`y`)  
   - Train-test split and scaling (if required)  

2. **Exploratory Data Analysis (EDA)**  
   - Feature distributions, correlation heatmaps, boxplots  
   - Identify outliers and relationships with the target  

3. **Model Training & Evaluation**  
   - Train multiple classifiers: Logistic Regression, SVM, Random Forest, XGBoost  
   - Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC  

4. **Feature Importance & ROC Curves**  
   - Identify top predictive features  
   - Visualize model performance via ROC curves  

---

## Conclusion
These projects demonstrate the application of machine learning for medical diagnosis. Tree-based models (Random Forest, XGBoost) generally provide better performance on tabular medical datasets. Proper feature selection, scaling, and hyperparameter tuning are essential for improving model precision and recall.

---
