# 📉 Customer Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses such as telecom, banking, and SaaS companies.
This project aims to **predict which customers are likely to stop using a service** and identify the most important factors influencing churn.

Multiple machine-learning classification models were trained and compared, and business insights were derived from exploratory data analysis (EDA).

---

## 🎯 Objectives

* Analyze customer data to find churn patterns
* Perform Exploratory Data Analysis (EDA)
* Build classification models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Evaluate models using:

  * Accuracy
  * Recall
  * ROC-AUC
* Recommend business strategies to reduce churn

---

## 📊 Dataset

The dataset used is the **Telco Customer Churn** dataset.

It contains customer information such as:

* Demographics
* Services subscribed
* Contract type
* Tenure
* Monthly charges
* Payment methods
* Churn status (target variable)

---

## 🧰 Tools & Technologies

* Python
* Google Colab
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

## 🔍 Exploratory Data Analysis (EDA)

During EDA, the following analyses were performed:

* Distribution of churned vs retained customers
* Tenure vs churn
* Monthly charges vs churn
* Contract type vs churn
* Correlation heatmap
* Feature importance analysis

### 📌 Key Insights:

* Customers with month-to-month contracts churn more frequently
* New customers (low tenure) are more likely to churn
* Higher monthly charges increase churn probability
* Certain internet service types show higher churn rates

---

## 🤖 Machine Learning Models

Three models were trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

The **best model** was selected based on **ROC-AUC score and Recall**, as these are critical for identifying churn-prone customers.

---

## 📈 Model Evaluation

Metrics used:

* Accuracy
* Recall
* ROC-AUC
* Confusion Matrix
* ROC Curve

Example comparison:

| Model               | Accuracy | Recall | ROC-AUC |
| ------------------- | -------- | ------ | ------- |
| Logistic Regression | 0.80     | 0.72   | 0.84    |
| Random Forest       | 0.85     | 0.79   | 0.90    |
| XGBoost             | 0.88     | 0.83   | 0.93    |

---

## 💡 Business Recommendations

Based on the analysis:

* Offer loyalty discounts to month-to-month customers
* Target new customers with onboarding and engagement programs
* Provide special retention offers to customers with high monthly charges
* Promote long-term contracts

---

## 📁 Repository Structure

```
customer-churn-prediction/
│
├── churn_analysis.ipynb
├── README.md
├── requirements.txt
├── models/
│   └── best_churn_model.pkl
├── images/
│   └── charts.png
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Notebook

Open:

```
churn_analysis.ipynb
```

in Jupyter Notebook or Google Colab.

---

## 🔮 Future Improvements

* Hyperparameter tuning
* SHAP model explainability
* Deployment using Streamlit
* Real-time churn prediction API
* Cross-validation
* Handling class imbalance

---

## 👤 Author

**Keerthi B R (Kiikii)**

Data Analytics / Data Science Intern Project

---

