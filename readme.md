# 🧾 Credit Scoring Prediction Project

This project is focused on predicting whether a credit card user will **default on payment** or not. We started with basic preprocessing and model development, and progressed toward optimizing the model for better recall of defaulters. Finally, the best-performing model was deployed as a **Streamlit web app**.

---

**Problem Statement**

Credit card default poses significant financial risks to lenders. Identifying customers likely to default is crucial for maintaining financial stability and minimizing losses. The goal of this project is to build a robust machine learning model that predicts whether a customer will default on their credit card payment in the next month. The dataset includes various customer features such as payment history, credit limit, and bill amounts. We aim to maximize recall for identifying defaulters (class 1), as missing them is more costly than misclassifying non-defaulters. The model is deployed as an interactive Streamlit web app for real-time prediction.

---


## 📂 Project Structure

```
credit-scoring/
│
├── scaler.joblib           # saved scaler
├── random_forest_model.joblib             # Saved model and
├── threshold                # saved threshold
├── app.py                   # Streamlit app 
├── CreditScoring.ipynb   # Main Jupyter notebook
├── requirements.txt       # Required packages
└── README.md              # Project summary
```

---

## 📌 Objective

Predict if a user will **default** on their credit card payment using machine learning models. A special focus was placed on maximizing **recall** for the defaulter class (`class 1`) to avoid false negatives.

---

## 🧼 Data Preprocessing

* Loaded dataset and inspected for class imbalance.
* Handled missing values and scaled the numeric features using `StandardScaler`.
* Split data into training and test sets.

---

## 📊 Exploratory Data Analysis (EDA)

We explored the data using the following visualizations:

1. **Countplot**
2. **Scatter Plot** 
3. **Histogram**
4. **Heatmap**
These visualizations helped us understand patterns in user behavior.

---

## 🤖 Model Building

We trained three classification models:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier** (Best performing model)

Random Forest achieved the best results overall.

---

## 🎯 Threshold Tuning

To improve recall for defaulters:

* We evaluated different classification thresholds on the Random Forest model.
* Plotted **Precision-Recall vs Threshold**.
* Selected `threshold = 0.31` to increase **recall** of class `1` (defaulters).
* Used classification report to evaluate the trade-off.

---

## 🧪 Model Evaluation

The model was evaluated using:

* Confusion Matrix
* Classification Report
* Precision, Recall, F1-Score
* ROC AUC Score

---

## 💾 Model Saving

Used `joblib` to save the final **Random Forest Classifier** and **StandardScaler**:

```python
joblib.dump(final_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(threshold, "threshold.joblib")
```

---

## 🚀 Streamlit App Deployment

Created an interactive web application using **Streamlit** that allows users to input the following details:

* `LIMIT_BAL`
* `AGE`
* `SEX`, `EDUCATION`, `MARRIAGE`
* Previous bill statements and payments (`BILL_AMT1`, `PAY_AMT1`, etc.)

On submission, the model predicts whether the user is likely to default or not, based on the optimized threshold.

---

## 📌 How to Run the App

1. Clone the repository:

```bash
git clone https://github.com/yourusername/credit-scoring.git
cd credit-scoring
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/app.py
```
