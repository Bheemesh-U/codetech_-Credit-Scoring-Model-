# Credit Scoring Model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Create or Load Dataset
# For demonstration, creating a sample dataset
data = {
    'income': [50000, 60000, 30000, 40000, 25000, 80000, 20000, 100000, 30000, 70000],
    'debts': [10000, 20000, 15000, 5000, 8000, 10000, 12000, 25000, 9000, 18000],
    'payment_history': [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],  # 1=Good, 0=Poor
    'credit_utilization': [20, 30, 80, 40, 50, 25, 90, 70, 60, 45],
    'credit_age': [5, 6, 1, 4, 3, 10, 2, 7, 3, 6],  # in years
    'employment_status': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 1=Employed, 0=Unemployed
    'target': [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]  # 1=Good Credit, 0=Bad Credit
}

df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis (optional)
# sns.pairplot(df, hue='target')
# plt.show()

# Step 3: Feature Scaling
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 5: Model Training

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Step 6: Evaluation Function
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_prob):.2f})')

# Step 7: Evaluate All Models
plt.figure(figsize=(8, 6))
evaluate_model("Logistic Regression", y_test, log_pred, log_model.predict_proba(X_test)[:, 1])
evaluate_model("Decision Tree", y_test, tree_pred, tree_model.predict_proba(X_test)[:, 1])
evaluate_model("Random Forest", y_test, rf_pred, rf_model.predict_proba(X_test)[:, 1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid()
plt.show()
