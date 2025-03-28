import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyzbar.pyzbar import decode
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from Phishing_URL_Models.feature_extraction import extract_features
from Phishing_Image_Models.data_loader import preprocess_image         

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSION = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = r"D:\1. BaiTap\4. Python\CodeAIPhishing\models"

with open(os.path.join(MODEL_DIR, "random_forest_URL.pkl"), "rb") as f:
    rf_url_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "svm_model_URL.pkl"), "rb") as f:
    svm_url_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "vectorizer_URL.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "random_forest_file.pkl"), "rb") as f:
    rf_file_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "svm_model_file.pkl"), "rb") as f:
    svm_file_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler_file.pkl"), "rb") as f:
    scaler = pickle.load(f)

cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_phishing_image.keras"))
rf_image_model = joblib.load(os.path.join(MODEL_DIR, "rf_image_model.pkl"))

EXPECTED_FEATURES = [
    "PdfSize", "MetadataSize", "Pages", "XrefLength", "TitleCharacters",
    "isEncrypted", "EmbeddedFiles", "Images", "Text", "Header", "Obj",
    "Endobj", "Stream", "Endstream", "Xref", "Trailer", "StartXref",
    "PageNo", "Encrypt", "ObjStm", "JS", "Javascript", "AA", "OpenAction",
    "Acroform", "JBIG2Decode", "RichMedia", "Launch", "EmbeddedFile",
    "XFA", "Colors"
]

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def extract_pdf_features(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        features = {feature: 0 for feature in EXPECTED_FEATURES}
        features["PdfSize"] = os.path.getsize(pdf_path) / 1024.0
        features["MetadataSize"] = len(str(pdf.metadata)) if pdf.metadata else 0
        features["Pages"] = len(pdf.pages)
        features["TitleCharacters"] = len(pdf.metadata.get("/Title", "")) if pdf.metadata else 0
        features["isEncrypted"] = 1 if pdf.is_encrypted else 0
        features["PageNo"] = len(pdf.pages)
        text = extract_text(pdf_path, laparams=LAParams())
        features["Text"] = len(text) if text else 0
        features["JS"] = 1 if "javascript" in text.lower() else 0
        features["Javascript"] = features["JS"]
        features["Images"] = 1 if features["PdfSize"] / features["Pages"] > 50 else 0
        return features
    except Exception as e:
        raise Exception(f"Error extracting features from PDF: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    rf_weight = 0.6
    svm_weight = 0.4
    threshold = 0.5

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({"error": "Invalid image file"}), 400

            qr_codes = decode(image)
            if qr_codes:
                qr_results = []
                for qr in qr_codes:
                    url = qr.data.decode("utf-8")
                    print(f"QR từ ảnh: {url}")
                    rf_features = pd.DataFrame([extract_features(url)])
                    rf_pred = rf_url_model.predict_proba(rf_features)[:, 1][0]
                    svm_pred = svm_url_model.decision_function(vectorizer.transform([url]))
                    svm_conf = 1 / (1 + np.exp(-svm_pred[0]))
                    ensemble_score = (rf_weight * rf_pred) + (svm_weight * svm_conf)
                    res = "Phishing" if ensemble_score > threshold else "Legitimate"
                    qr_results.append({
                        "qr_url": url,
                        "rf_confidence": round(rf_pred, 4),
                        "svm_confidence": round(svm_conf, 4),
                        "ensemble_confidence": round(ensemble_score, 4),
                        "result": res
                    })
                return jsonify({"qr_results": qr_results})

            try:
                cnn_input = cv2.resize(image, (128, 128)).astype("float32") / 255.0
                cnn_input = np.expand_dims(cnn_input, axis=0)
                cnn_pred = float(cnn_model.predict(cnn_input)[0][0])
            except Exception as e:
                print(f"Lỗi CNN: {e}")
                cnn_pred = None

            try:
                rf_features_img = preprocess_image(file_path).flatten().reshape(1, -1)
                rf_pred_img = float(rf_image_model.predict_proba(rf_features_img)[:, 1][0])
            except Exception as e:
                print(f"Lỗi RF ảnh: {e}")
                rf_pred_img = None

            if cnn_pred is not None and rf_pred_img is not None:
                ensemble_score = (rf_weight * rf_pred_img) + (svm_weight * cnn_pred)
                res = "Phishing" if ensemble_score > threshold else "Legitimate"
            else:
                ensemble_score = None
                res = "Error"

            return jsonify({
                "rf_confidence": round(rf_pred_img, 4) if rf_pred_img is not None else "Error",
                "cnn_confidence": round(cnn_pred, 4) if cnn_pred is not None else "Error",
                "ensemble_confidence": round(ensemble_score, 4) if ensemble_score is not None else "Error",
                "result": res
            })

        elif allowed_file(filename, ALLOWED_PDF_EXTENSION):
            try:
                features = extract_pdf_features(file_path)
                feature_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)
                
                rf_prob = rf_file_model.predict_proba(feature_df)[0][1]
                features_scaled = scaler.transform(feature_df)
                svm_prob = svm_file_model.predict_proba(features_scaled)[0][1]

                ensemble_score = (rf_prob + svm_prob) / 2
                res = "Phishing" if ensemble_score > threshold else "Legitimate"
                os.remove(file_path)
                return jsonify({
                    "rf_confidence": round(rf_prob, 4),
                    "svm_confidence": round(svm_prob, 4),
                    "ensemble_confidence": round(ensemble_score, 4),
                    "result": res
                }), 200
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No valid input data"}), 400

    if "url" in data:
        url = data["url"].strip()
        if not url:
            return jsonify({"error": "URL is required"}), 400

        rf_features = pd.DataFrame([extract_features(url)])
        rf_pred = rf_url_model.predict_proba(rf_features)[:, 1][0]
        svm_pred = svm_url_model.decision_function(vectorizer.transform([url]))
        svm_conf = 1 / (1 + np.exp(-svm_pred[0]))
        ensemble_score = (rf_weight * rf_pred) + (svm_weight * svm_conf)
        res = "Phishing" if ensemble_score > threshold else "Legitimate"
        return jsonify({
            "rf_confidence": round(rf_pred, 4),
            "svm_confidence": round(svm_conf, 4),
            "ensemble_confidence": round(ensemble_score, 4),
            "url": url,
            "result": res
        })

    if "text" in data:
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400

        text_features = vectorizer.transform([text])
        text_pred = rf_url_model.predict_proba(text_features)[:, 1][0]
        res = "Phishing" if text_pred > threshold else "Legitimate"
        return jsonify({
            "text_confidence": round(text_pred, 4),
            "result": res
        })

    return jsonify({"error": "Invalid request format"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
