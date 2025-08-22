# from flask import Flask, request, render_template, jsonify
# from flask_sqlalchemy import SQLAlchemy
# import tensorflow as tf
# import numpy as np
# import mne
# import os
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)


# # Load trained model
# model = tf.keras.models.load_model("schizo_model.h5")

# app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://neondb_owner:npg_aTe0ZK9DArWJ@ep-silent-king-adjanc6b-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
# if __name__ == "__main__":
#     with app.app_context():
#         db.create_all()
# class LoginCredentials(db.Model):
#     __tablename__ = "login_credentials"
#     db.metadata,
#     autoload_with=db.engine
# # Preprocessing function (matching training notebook)
# def preprocess_edf(file_path):
#     raw = mne.io.read_raw_edf(file_path, preload=True)
#     data, _ = raw[:, :]  # shape: (channels, samples)

#     # Keep only first 19 channels (to match training)
#     data = data[:19, :2000]  # (19, 2000)

#     # Normalize per channel
#     data = (data - np.mean(data, axis=1, keepdims=True)) / (
#         np.std(data, axis=1, keepdims=True) + 1e-6
#     )

#     # Transpose -> (time_steps, channels)
#     data = data.T   # (2000, 19)

#     # Add batch dimension -> (1, 2000, 19)
#     data = np.expand_dims(data, axis=0).astype(np.float32)
#     return data

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file uploaded"
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected"
#             else:
#                 filepath = os.path.join("uploads", file.filename)
#                 os.makedirs("uploads", exist_ok=True)
#                 file.save(filepath)

#                 try:
#                     data = preprocess_edf(filepath)
#                     prediction = model.predict(data)
#                     label = np.argmax(prediction, axis=1)[0]

#                     if label == 0:
#                         result = "Schizophrenia"
#                     else:
#                         result = "Healthy Control"
#                 except Exception as e:
#                     result = f"Error: {str(e)}"

#     return render_template("loginpage.html", result=result)

# from werkzeug.security import generate_password_hash, check_password_hash

# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.get_json()
#     name = data.get("name")
#     email = data.get("email")
#     password = data.get("password")

#     # Check if email already exists
#     existing_user = LoginCredentials.query.filter_by(email=email).first()
#     if existing_user:
#         return jsonify({"status": "error", "message": "Email already registered"}), 400

#     # Save new user
#     hashed_pw = generate_password_hash(password)
#     new_user = LoginCredentials(name=name, email=email, password=hashed_pw)
#     db.session.add(new_user)
#     db.session.commit()

#     return jsonify({"status": "success", "message": "User registered!"})


# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data.get("email")
#     password = data.get("password")

#     user = LoginCredentials.query.filter_by(email=email).first()

#     if user and check_password_hash(user.password, password):
#         return jsonify({"status": "success", "message": "Login successful!"})
#     else:
#         return jsonify({"status": "error", "message": "Invalid email or password"}), 401


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]

#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     file_path = os.path.join("uploads", file.filename)
#     os.makedirs("uploads", exist_ok=True)
#     file.save(file_path)

#     # Preprocess EDF file
#     X = preprocess_edf(file_path)

#     # Predict
#     preds = model.predict(X)
#     result = "Schizophrenic" if preds[0][0] > 0.5 else "Healthy"

#     return jsonify({"prediction": result, "raw_output": preds.tolist()})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import mne
import os
import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("schizo_model.h5")

# Database connection function
def get_db_connection():
    return psycopg2.connect(
        dbname="neondb",
        user="neondb_owner",
        password="npg_aTe0ZK9DArWJ",
        host="ep-silent-king-adjanc6b-pooler.c-2.us-east-1.aws.neon.tech",
        port="5432",
        sslmode="require"
    )

@app.route("/")
def index():
    return render_template("loginpage.html")

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


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    hashed_pw = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("INSERT INTO login_credentials (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "User created successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM login_credentials WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/upload", methods=["GET", "POST"])
def upload():
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

if __name__ == "__main__":
    app.run(debug=True)
