# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
from utils.preprocess import preprocess_input  # 👈 new import

# Initialize Flask app
app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/xgboost_crime_model.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "API is up and running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert incoming JSON to DataFrame
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Preprocess input using encoding logic
        input_df = preprocess_input(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

