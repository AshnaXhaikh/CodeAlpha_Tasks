# Breast Cancer Prediction Project

## Project Overview
This project aims to build and evaluate machine learning models to predict whether a breast mass is malignant or benign based on a set of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository. The primary goal is to identify the most effective model for accurately classifying a diagnosis.  

---

## Dataset
- **Source:** UCI Machine Learning Repository  
- **Features:** 30 features representing measurements of tumor cell nuclei (e.g., mean radius, texture, perimeter).  
- **Target:** `diagnosis` column, a binary class indicating Malignant (M) or Benign (B).  

---

## Methodology
The project follows a standard machine learning pipeline:

1. **Data Loading and Initial Cleanup**  
   - Load dataset and drop unnecessary columns (`id`, `Unnamed: 32`).  
   - Convert `diagnosis` to binary numerical format (1 for Malignant, 0 for Benign).  

2. **Exploratory Data Analysis (EDA)**  
   - Generate correlation heatmap to understand feature relationships.  
   - Identify which features are most predictive of the diagnosis.  

3. **Data Preprocessing**  
   - Separate features (`X`) and target (`y`).  
   - Split data into training (80%) and testing (20%) sets.  
   - Scale numerical features using `StandardScaler`.  

4. **Model Training and Evaluation**  
   - Train four classification models on preprocessed data.  
   - Evaluate performance using key metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.  

---

## Models and Results

| Model                   | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|-------------------------|---------|-----------|--------|----------|---------|
| Logistic Regression      | 0.9737  | 0.9762    | 0.9535 | 0.9647   | 0.9697  |
| Support Vector Machine (SVM) | 0.9825  | 1.0000    | 0.9535 | 0.9762   | 0.9767  |
| Random Forest            | 0.9649  | 0.9756    | 0.9302 | 0.9524   | 0.9581  |
| XGBoost                  | 0.9561  | 0.9524    | 0.9302 | 0.9412   | 0.9510  |

---

## Conclusion
Based on the results, the **Support Vector Machine (SVM)** model is the most effective for this task.  

- Achieved the highest Accuracy, F1-Score, and ROC-AUC.  
- Perfect precision of 1.00 indicates that every case predicted as malignant was actually malignant.  

This makes the SVM model a highly reliable tool for predicting breast cancer diagnosis on this dataset.
