from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from xgboost import XGBClassifier
import os

app = Flask(__name__)
CORS(app) 

# Paths
MODEL_PATH = os.path.join("models", "xgb_model_simplified.json")
SCALER_PATH = os.path.join("models", "xgb_scaler_simplified.pkl")

# Load model + scaler
model = XGBClassifier()
model.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

# Define required feature names (order must match training dataset)
REQUIRED_FEATURES = [
    "fever",
    "muscle_pain",
    "jaundice",
    "red_eyes",
    "creatinine",
    "platelets",
    "pcr"
]

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required features: {missing}"}), 400

    try:
        # Extract features in the correct order
        features = [float(data[f]) for f in REQUIRED_FEATURES]

        # Defensive: no lists/arrays allowed
        if any(isinstance(v, (list, np.ndarray)) for v in features):
            return jsonify({"error": "Each input value must be a single number, not an array/list."}), 400

    except (ValueError, TypeError):
        return jsonify({"error": "All feature values must be numeric scalars."}), 400

    # Convert to numpy and scale
    input_data = np.array(features, dtype=float).reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]

    return jsonify({
        "prediction": int(prediction),
        "probability_0": float(probability[0]),
        "probability_1": float(probability[1])
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
