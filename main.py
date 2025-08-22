from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import mne
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("schizo_model.h5")

# Preprocessing function (matching training notebook)

def preprocess_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data, _ = raw[:, :]  # shape: (channels, samples)

    # Keep only first 19 channels (to match training)
    data = data[:19, :2000]  # (19, 2000)

    # Normalize per channel
    data = (data - np.mean(data, axis=1, keepdims=True)) / (
        np.std(data, axis=1, keepdims=True) + 1e-6
    )

    # Transpose -> (time_steps, channels)
    data = data.T   # (2000, 19)

    # Add batch dimension -> (1, 2000, 19)
    data = np.expand_dims(data, axis=0).astype(np.float32)

    return data

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            result = "No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "No file selected"
            else:
                filepath = os.path.join("uploads", file.filename)
                os.makedirs("uploads", exist_ok=True)
                file.save(filepath)

                try:
                    data = preprocess_edf(filepath)
                    prediction = model.predict(data)
                    label = np.argmax(prediction, axis=1)[0]

                    if label == 0:
                        result = "Schizophrenia"
                    else:
                        result = "Healthy Control"
                except Exception as e:
                    result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Preprocess EDF file
    X = preprocess_edf(file_path)

    # Predict
    preds = model.predict(X)
    result = "Schizophrenic" if preds[0][0] > 0.5 else "Healthy"

    return jsonify({"prediction": result, "raw_output": preds.tolist()})


if __name__ == "__main__":
    app.run(debug=True)

