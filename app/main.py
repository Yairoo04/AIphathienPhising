from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyzbar.pyzbar import decode
from Phishing_URL_Models.feature_extraction import extract_features
from Phishing_Image_Models.data_loader import preprocess_image  

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load các mô hình
with open("models/random_forest.pkl", "rb") as f:
    rf_url_model = pickle.load(f)

with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

cnn_model = tf.keras.models.load_model("models/cnn_phishing_image.keras")
rf_image_model = joblib.load("models/rf_image_model.pkl")

# Kiểm tra file hợp lệ
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400


        qr_codes = decode(image)
        if qr_codes:
            qr_results = []
            for qr in qr_codes:
                url = qr.data.decode("utf-8")
                print(f"URL từ QR: {url}")
                rf_features = pd.DataFrame([extract_features(url)])
                rf_prediction_proba = rf_url_model.predict_proba(rf_features)[:, 1][0]

                svm_input = vectorizer.transform([url])
                svm_prediction_proba = svm_model.decision_function(svm_input)
                svm_confidence = 1 / (1 + np.exp(-svm_prediction_proba[0]))

                ensemble_score = (0.5 * rf_prediction_proba) + (0.5 * svm_confidence)
                result = "Phishing" if ensemble_score > 0.5 else "Legitimate"

                qr_results.append({
                    "qr_url": url,
                    "rf_confidence": round(rf_prediction_proba, 4),
                    "svm_confidence": round(svm_confidence, 4),
                    "ensemble_confidence": round(ensemble_score, 4),
                    "result": result
                })

            return jsonify({"qr_results": qr_results})

        cnn_input = cv2.resize(image, (128, 128)).astype("float32") / 255.0
        cnn_input = np.expand_dims(cnn_input, axis=0)

        try:
            cnn_prediction = float(cnn_model.predict(cnn_input)[0][0])
        except Exception as e:
            print(f"Lỗi khi dự đoán với CNN: {e}")
            cnn_prediction = None

        try:
            rf_features = preprocess_image(file_path).flatten().reshape(1, -1)
            rf_prediction_proba = float(rf_image_model.predict_proba(rf_features)[:, 1][0])
        except Exception as e:
            print(f"Lỗi khi dự đoán với Random Forest: {e}")
            rf_prediction_proba = None

        # Kiểm tra và tính toán ensemble
        if cnn_prediction is not None and rf_prediction_proba is not None:
            ensemble_score = (0.5 * cnn_prediction) + (0.5 * rf_prediction_proba)
            result = "Phishing" if ensemble_score > 0.5 else "Legitimate"
        else:
            ensemble_score = None
            result = "Error"

        return jsonify({
            "rf_confidence": round(rf_prediction_proba, 4) if rf_prediction_proba is not None else "Error",
            "cnn_confidence": round(cnn_prediction, 4) if cnn_prediction is not None else "Error",
            "ensemble_confidence": round(ensemble_score, 4) if ensemble_score is not None else "Error",
            "result": result
        })

    # Xử lý URL hoặc text
    data = request.get_json()
    if not data:
        return jsonify({"error": "No valid input data"}), 400
    
    if "url" in data:
        url = data["url"].strip()
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        rf_features = pd.DataFrame([extract_features(url)])  
        rf_prediction_proba = rf_url_model.predict_proba(rf_features)[:, 1][0]

        svm_input = vectorizer.transform([url])
        svm_prediction_proba = svm_model.decision_function(svm_input)
        svm_confidence = 1 / (1 + np.exp(-svm_prediction_proba[0])) 

        ensemble_score = (0.5 * rf_prediction_proba) + (0.5 * svm_confidence)
        result = "Phishing" if ensemble_score > 0.5 else "Legitimate"

        return jsonify({
            "rf_confidence": round(rf_prediction_proba, 4),
            "svm_confidence": round(svm_confidence, 4),
            "ensemble_confidence": round(ensemble_score, 4),
            "url": url,
            "result": result
        })
    
    if "text" in data:
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400

        text_features = vectorizer.transform([text])  
        text_prediction_proba = rf_url_model.predict_proba(text_features)[:, 1][0]
        result = "Phishing" if text_prediction_proba > 0.5 else "Legitimate"

        return jsonify({
            "text_confidence": round(text_prediction_proba, 4),
            "result": result
        })

    return jsonify({"error": "Invalid request format"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
