# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
from utils.preprocess import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/xgboost_crime_model.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# === HOME ===
@app.route('/')
def home():
    return """
    <h2>🧠 Crime Prediction API</h2>
    <p>Welcome! This API uses machine learning to predict crime categories.</p>
    <ul>
        <li>✅ <code>GET /ping</code> – Check API status</li>
        <li>🎯 <code>POST /predict</code> – Get a crime prediction from your input</li>
    </ul>
    """

# === HEALTH CHECK ===
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "API is up and running!"})

# === PREDICT ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Preprocess input
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
