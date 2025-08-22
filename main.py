from flask import Flask, request, render_template
import os
import numpy as np
import mne
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'edf'}

MODEL_PATH = "schizo_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess EEG for model
def preprocess_for_model(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.filter(0.5, 40., verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.resample(200, npad="auto", verbose=False)
    data = raw.get_data()

    n_channels, n_samples = data.shape
    window = 2000  # 10s at 200Hz
    if n_samples < window:
        pad = np.zeros((n_channels, window - n_samples))
        data = np.concatenate([data, pad], axis=1)
    else:
        data = data[:, :window]

    X = data[np.newaxis, ..., np.newaxis]
    return X

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return 'Invalid file type'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    X = preprocess_for_model(filepath)
    pred = model.predict(X)

    if pred.shape[-1] == 1:  
        prob = float(pred[0][0])
        label = 'Schizophrenia' if prob > 0.5 else 'Healthy'
        confidence = prob if label == 'Schizophrenia' else 1 - prob
    else:  # softmax
        label_idx = np.argmax(pred)
        label = 'Schizophrenia' if label_idx == 1 else 'Healthy'
        confidence = float(np.max(pred))

    return render_template('result.html', label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
