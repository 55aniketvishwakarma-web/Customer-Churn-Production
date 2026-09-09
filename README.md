# 👥 Customer Churn Prediction Using Machine Learning

## Objective
Predict whether a customer is likely to leave a service.

## Algorithm
**Logistic Regression** — a supervised classification algorithm suitable for binary targets such as Churn = Yes/No.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Features
- Age
- Tenure (months)
- Monthly Charges
- Contract
- Internet Service
- Tech Support
- Payment Method

## Workflow
Data → EDA → Feature Selection → One-Hot Encoding → Scaling → Train/Test Split → Logistic Regression → Evaluation → Prediction

## Evaluation
Accuracy, Precision, Recall, F1 Score and Confusion Matrix.

## Dataset
The included dataset is **synthetic educational data** created for this portfolio project. It is not real customer data.

## How to Run
```bash
pip install -r requirements.txt
python customer_churn_prediction.py
```

## Interview Summary
I developed a customer churn prediction project using Python and Logistic Regression. I explored the data, encoded categorical features using One-Hot Encoding, standardized numerical features, split the data into training and testing sets, trained the classifier, and evaluated it using accuracy, precision, recall, F1 score and a confusion matrix. Finally, I predicted churn for a new customer.
