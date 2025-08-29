# Heart Disease Prediction

## Objective

Predict the possibility of heart disease in patients using structured medical data (features like symptoms, age, blood test results, and physiological measurements) by applying machine learning classification techniques.

---

## Dataset

* **Source:** UCI Machine Learning Repository – Heart Disease Dataset (Cleveland subset)
* **Features:**

  * `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol),
    `fbs` (fasting blood sugar), `restecg` (resting ECG results), `thalach` (max heart rate),
    `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (ST segment slope),
    `ca` (number of major vessels), `thal` (thalassemia type), `target` (presence of heart disease).

---

## Methodology

1. **Data Preprocessing:**

   * Handle missing values (`ca` and `thal`) by imputation or dropping rows
   * Convert target to binary (0 = no disease, 1 = disease)
   * Scale numerical features for models like Logistic Regression and SVM

2. **Models Applied:**

   * Logistic Regression
   * Support Vector Machine (SVM)
   * Random Forest Classifier
   * XGBoost Classifier

3. **Evaluation Metrics:**

   * Accuracy
   * Precision, Recall, F1-score
   * ROC-AUC
   * Confusion Matrix

4. **Feature Importance Analysis:**

   * Identified top predictors: `cp` (chest pain type), `thalach` (maximum heart rate), `ca` (vessels colored), `oldpeak`, `thal`.

---

## Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.81     | 0.76      | 0.91   | 0.83     | 0.81    |
| SVM                 | 0.93     | 0.92      | 0.94   | 0.93     | 0.93    |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| XGBoost             | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |

> **Note:** Perfect scores for Random Forest and XGBoost may indicate overfitting; cross-validation is recommended to ensure generalization.

---

## Insights

* **Most important features:** `cp`, `thalach`, `ca`, `oldpeak`, `thal`
* **Moderate predictors:** `age`, `chol`, `exang`, `trestbps`
* **Least important:** `slope`, `sex`, `restecg`, `fbs`
* Symptoms and physiological indicators dominate predictions, while demographic and metabolic features have less impact.

---

## Conclusion

The models successfully predict heart disease from patient data. Tree-based methods (Random Forest, XGBoost) achieved perfect accuracy on this dataset, highlighting the predictive power of symptom and physiological features. Proper cross-validation is recommended to validate model robustness.

---
