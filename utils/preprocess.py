# utils/preprocess.py

import joblib
import pandas as pd

# Load saved encoders and feature columns
label_encoders = joblib.load("model/label_encoders.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

categorical_cols = list(label_encoders.keys())

def preprocess_input(df):
    # Handle rare PremiseType
    if "PremiseType" in df.columns:
        known_classes = label_encoders["PremiseType"].classes_
        df["PremiseType"] = df["PremiseType"].apply(lambda x: x if x in known_classes else "OTHER")

    # Handle rare Weapon
    if "Weapon" in df.columns:
        known_classes = label_encoders["Weapon"].classes_
        df["Weapon"] = df["Weapon"].apply(lambda x: x if x in known_classes else "OTHER")

    # Simplify Race values
    if "Race" in df.columns:
        df["Race"] = df["Race"].replace({
            "BLACK_OR_AFRICAN_AMERICAN": "Black",
            "WHITE": "White",
            "UNKNOWN": "Unknown",
            "Unknown": "Unknown"
        })
        df["Race"] = df["Race"].where(df["Race"].isin(["Black", "White", "Unknown"]), "Other")

    # Encode all categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            known_classes = le.classes_
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
            df[col] = le.transform(df[col])

    # Reorder columns
    df = df[feature_columns]

    return df
