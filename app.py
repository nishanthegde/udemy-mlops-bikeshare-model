from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- Load model or fallback to dummy for testing ---
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    model = load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ Model not found at {MODEL_PATH}. Using dummy model for testing.")
    class DummyModel:
        feature_names_in_ = []
        def predict(self, X):
            return np.array([np.log(100.0)])  # log(100) ~ baseline

    model = DummyModel()

# --- API routes ---

@app.route("/")

def healthcheck():
    return jsonify({"status": "ok", "message": "Bikeshare model API running"})


@app.route("/predict", methods=["POST"])

def predict():
    try:
        # Allow parsing even when Content-Type isn't 'application/json'
        input_json = request.get_json(silent=True)
        if not input_json:
            return jsonify({"error": "Missing or invalid JSON payload"}), 400

        # Wrap JSON into DataFrame
        input_df = pd.DataFrame([input_json])

        # Enforce feature order & validate input
        if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) > 0:
            expected = list(model.feature_names_in_)
            missing = [f for f in expected if f not in input_df.columns]
            extra = [c for c in input_df.columns if c not in expected]

            if missing:
                return jsonify({"error": f"Missing features: {missing}"}), 400

            # Drop extra, reorder correctly
            input_df = input_df.reindex(columns=expected)

        # Run prediction and reverse log-transform
        pred = model.predict(input_df)
        pred = float(np.exp(pred[0]))

        return jsonify({"prediction": pred})

    except Exception as e:
        print(f"❌ Internal Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5052, debug=True)
