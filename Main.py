import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
df = pd.read_csv('credit_card_fraud_detection.csv')

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print columns to verify
print("Columns in the dataset:", df.columns)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature scaling
scaler = StandardScaler()
df[['amount', 'time']] = scaler.fit_transform(df[['amount', 'time']])

# Splitting the data
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Print class distribution
print("Class distribution:\n", y.value_counts())

# Check number of samples in the minority class
minority_class_size = y.value_counts().min()
print(f"Minority class size: {minority_class_size}")

# Handle imbalanced data using SMOTE
smote_neighbors = min(5, minority_class_size - 1)  # Ensure neighbors <= minority class size
smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
X_res, y_res = smote.fit_resample(X, y)

# Splitting into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Train a Random Forest model
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rfc.fit(X_train, y_train)

# Predictions and evaluation for Random Forest
y_pred = rfc.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Anomaly detection using LOF
# Adjusting n_neighbors for small datasets
n_neighbors_lof = min(2, len(X_train) - 1)  # Ensure n_neighbors <= len(X_train) - 1
lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, contamination=0.1)
y_pred_lof = lof.fit_predict(X_test)

# Convert LOF output (-1 for outliers, 1 for inliers) to match binary labels
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)

# LOF Evaluation
print("LOF Accuracy:", accuracy_score(y_test, y_pred_lof))
print("LOF Classification Report:")
print(classification_report(y_test, y_pred_lof, zero_division=0))
print("LOF Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lof, labels=[0, 1]))

# Anomaly detection using Isolation Forest
# Adjusting contamination for small datasets
isof = IsolationForest(contamination=0.1, random_state=42)
y_pred_isof = isof.fit_predict(X_test)

# Convert Isolation Forest output to match binary labels
y_pred_isof = np.where(y_pred_isof == -1, 1, 0)

# Isolation Forest Evaluation
print("Isolation Forest Accuracy:", accuracy_score(y_test, y_pred_isof))
print("Isolation Forest Classification Report:")
print(classification_report(y_test, y_pred_isof, zero_division=0))
print("Isolation Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_isof, labels=[0, 1]))
