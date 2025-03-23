# evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from utils.preprocess import preprocess_input

# === 1. Load test data ===
df = pd.read_csv("data/test_data.csv")

# === 2. Separate features and target ===
y_true = df["CrimeCategory"]
X = df.drop(columns=["CrimeCategory", "Description"])  # Drop target + raw text

# === 3. Load model and target encoder ===
model = joblib.load("model/xgboost_crime_model.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

# === 4. Encode true labels
y_true_encoded = target_encoder.transform(y_true)

# === 5. Preprocess features
X_processed = preprocess_input(X)

# === 6. Predict
y_pred = model.predict(X_processed)

# === 7. Get only the labels present in this test set
present_labels = unique_labels(y_true_encoded, y_pred)
present_label_names = target_encoder.inverse_transform(present_labels)

# === 8. Print Evaluation
print("✅ Classification Report:\n")
print(classification_report(y_true_encoded, y_pred, labels=present_labels, target_names=present_label_names))

print("\n🌀 Confusion Matrix:\n")
print(confusion_matrix(y_true_encoded, y_pred, labels=present_labels))
