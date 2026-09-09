# Customer Churn Prediction using Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("dataset/customer_churn.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Exploratory Data Analysis
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.show()

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.show()

# Features and target
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

categorical_features = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]
numeric_features = ["Age", "Tenure_Months", "MonthlyCharges"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# New customer prediction
new_customer = pd.DataFrame({
    "Age": [25],
    "Tenure_Months": [6],
    "MonthlyCharges": [110],
    "Contract": ["Month-to-month"],
    "InternetService": ["Fiber optic"],
    "TechSupport": ["No"],
    "PaymentMethod": ["Electronic check"]
})

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

print("\n--- NEW CUSTOMER PREDICTION ---")
print("Prediction:", "Will Churn" if prediction == 1 else "Will Not Churn")
print(f"Churn Probability: {probability:.2%}")
