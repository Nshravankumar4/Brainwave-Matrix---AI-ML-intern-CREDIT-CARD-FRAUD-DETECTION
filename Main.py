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

# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature scaling
scaler = StandardScaler()
df[['amount', 'time']] = scaler.fit_transform(df[['amount', 'time']])

# Splitting the data into features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Print class distribution
print("Class distribution:\n", y.value_counts())

# Handle imbalanced data using SMOTE
smote_neighbors = min(5, y.value_counts().min() - 1)  # Adjust neighbors based on minority class size
smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
X_res, y_res = smote.fit_resample(X, y)

# Splitting into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Train a Random Forest model (Supervised Learning)
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rfc.fit(X_train, y_train)

# Predictions and evaluation for Random Forest
y_pred = rfc.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Anomaly detection using LOF
lof = LocalOutlierFactor(n_neighbors=2, contamination=0.2)
y_pred_lof = lof.fit_predict(X_test)

# Convert LOF output (-1 for outliers, 1 for inliers) to match binary labels
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)

# LOF Evaluation
print("LOF Accuracy:", accuracy_score(y_test, y_pred_lof))
print("LOF Classification Report:")
print(classification_report(y_test, y_pred_lof))
print("LOF Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lof))

# Anomaly detection using Isolation Forest
isof = IsolationForest(contamination=0.2, random_state=42)
y_pred_isof = isof.fit_predict(X_test)

# Convert Isolation Forest output to match binary labels
y_pred_isof = np.where(y_pred_isof == -1, 1, 0)

# Isolation Forest Evaluation
print("Isolation Forest Accuracy:", accuracy_score(y_test, y_pred_isof))
print("Isolation Forest Classification Report:")
print(classification_report(y_test, y_pred_isof))
print("Isolation Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_isof))
