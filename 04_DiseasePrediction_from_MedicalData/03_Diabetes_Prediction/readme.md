# Diabetes Prediction Project

**Organization:** CodeAlpha  
**Task:** Disease Prediction from Medical Data  

## Objective
Predict the likelihood of diabetes in patients using structured medical data, including features such as glucose level, BMI, age, blood pressure, and other health indicators. The goal is to identify the most effective machine learning model for accurate prediction.

---

## Dataset
- **Source:** UCI Machine Learning Repository – Pima Indians Diabetes Dataset  
- **Features:**  
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`  
- **Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)  

---

## Methodology
The project follows a standard machine learning pipeline:

1. **Data Preprocessing**  
   - Replace invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with median values.  
   - Split data into features (`X`) and target (`y`).  
   - Train-test split (80/20) and scale numerical features.  

2. **Exploratory Data Analysis (EDA)**  
   - Feature distributions, correlation heatmap, and boxplots to identify outliers and relationships with the target.  

3. **Model Training & Evaluation**  
   - Models trained: Logistic Regression, SVM, Random Forest, XGBoost.  
   - Metrics used: Accuracy, Precision, Recall, F1-score, ROC-AUC.  

4. **Feature Importance Analysis**  
   - Identify top predictive features using Random Forest and XGBoost.  

5. **ROC Curve Visualization**  
   - Compare models based on true positive vs. false positive trade-offs.  

---

## Results

| Model               | Accuracy | Precision | Recall  | F1-score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.708    | 0.600     | 0.500  | 0.545    | 0.660   |
| SVM                 | 0.740    | 0.652     | 0.556  | 0.600    | 0.698   |
| Random Forest       | 0.740    | 0.652     | 0.556  | 0.600    | 0.698   |
| XGBoost             | 0.766    | 0.680     | 0.630  | 0.654    | 0.735   |

**Key Insights:**  
- **XGBoost** performed best overall with the highest Accuracy, F1-score, and ROC-AUC.  
- Top predictive features: `Glucose`, `BMI`, `Age`, `Insulin`, `Pregnancies`.  
- Tree-based models handle nonlinearities better than linear models in this dataset.  

---

## Conclusion
XGBoost is the most reliable model for predicting diabetes in this dataset. Proper feature engineering, hyperparameter tuning, and cross-validation can further improve precision and recall.  

---
