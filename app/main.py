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
from Phishing_URL_Models.feature_extraction import extract_features  # Import hàm extract_features từ feature_extraction.py
from Phishing_Image_Models.data_loader import preprocess_image  # Hàm tiền xử lý ảnh cho RF

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

# Load mô hình CNN (EfficientNetB0)
cnn_model = tf.keras.models.load_model("models/cnn_phishing_image.keras")

# Load mô hình Random Forest cho ảnh
rf_image_model = joblib.load("models/rf_image_model.pkl")

# Kiểm tra định dạng file hợp lệ
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

        # Lưu file ảnh tạm thời
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Đọc ảnh và kiểm tra lỗi
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Xử lý ảnh cho CNN (Resize về 128x128)
        cnn_input = cv2.resize(image, (128, 128))
        cnn_input = cnn_input.astype("float32") / 255.0 
        cnn_input = np.expand_dims(cnn_input, axis=0) 

        # Dự đoán với CNN
        cnn_prediction = float(cnn_model.predict(cnn_input)[0][0]) 
        # Xử lý ảnh cho Random Forest
        rf_features = preprocess_image(file_path).flatten().reshape(1, -1)  
        rf_prediction_proba = float(rf_image_model.predict_proba(rf_features)[:, 1][0])  

        # Ensemble (tổng hợp kết quả)
        cnn_weight = 0.5
        rf_weight = 0.5
        ensemble_score = (cnn_weight * cnn_prediction) + (rf_weight * rf_prediction_proba)
        result = "Phishing" if ensemble_score > 0.5 else "Legitimate"

        return jsonify({
            "rf_confidence": round(rf_prediction_proba, 4),
            "cnn_confidence": round(cnn_prediction, 4),
            "ensemble_confidence": round(float(ensemble_score), 4),
            "result": result
        })
    else:
        url = request.json.get('url', '').strip()
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Dự đoán bằng Random Forest
        rf_features = pd.DataFrame([extract_features(url)])  
        rf_prediction_proba = rf_url_model.predict_proba(rf_features)
        rf_prediction = float(rf_prediction_proba[:, 1][0])

        # Dự đoán bằng SVM
        svm_input = vectorizer.transform([url])
        svm_prediction_proba = svm_model.decision_function(svm_input)
        svm_confidence = 1 / (1 + np.exp(-svm_prediction_proba[0])) 

        # Trọng số cho ensemble
        rf_weight = 0.5
        svm_weight = 0.5
        ensemble_score = (rf_weight * rf_prediction) + (svm_weight * svm_confidence)
        result = "Phishing" if ensemble_score > 0.5 else "Legitimate"

        return jsonify({
            'rf_confidence': round(rf_prediction, 4),
            'svm_confidence': round(svm_confidence, 4),
            'ensemble_confidence': round(ensemble_score, 4),
            'url': url,
            'result': result
        })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
