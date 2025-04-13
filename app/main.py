import os
import re
import cv2
import numpy as np
import joblib
import tensorflow as tf
import pickle
import pandas as pd
import logging
from datetime import datetime
from email import policy
from email.parser import BytesParser
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyzbar.pyzbar import decode
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from contextlib import contextmanager
from typing import Dict, Optional, List, Union
import email

# Giả định các module này được định nghĩa ở nơi khác
from Phishing_URL_Models.feature_extraction import extract_features
from Phishing_Image_Models.data_loader import preprocess_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Cấu hình
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSION = {"pdf"}
ALLOWED_EMAIL_EXTENSION = {"eml"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Sửa đường dẫn MODEL_DIR để trỏ đến thư mục models nằm ngoài app
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Danh sách đặc trưng email
EMAIL_FEATURES = [
    "missing_list-subscribe", "missing_precedence", "missing_list-help", "missing_list-post",
    "missing_list-unsubscribe", "missing_x-spam-status", "missing_list-id", "str_precedence_list",
    "missing_x-spam-checker-version", "time_zone", "missing_sender", "domain_match_to_received",
    "domain_match_errors-to_sender", "missing_x-mailer", "missing_errors-to", "missing_x-priority",
    "str_return-path_bounce", "missing_x-beenthere", "str_message-ID_dollar", "missing_delivered-to",
    "missing_references", "missing_content-disposition", "number_replies", "span_time",
    "date_comp_date_received", "x-priority", "missing_reply-to"
]

EXPECTED_FEATURES = [
    "PdfSize", "MetadataSize", "Pages", "XrefLength", "TitleCharacters",
    "isEncrypted", "EmbeddedFiles", "Images", "Text", "Header", "Obj",
    "Endobj", "Stream", "Endstream", "Xref", "Trailer", "StartXref",
    "PageNo", "Encrypt", "ObjStm", "JS", "Javascript", "AA", "OpenAction",
    "Acroform", "JBIG2Decode", "RichMedia", "Launch", "EmbeddedFile",
    "XFA", "Colors"
]

# Quản lý mô hình
class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
    
    def load_model(self, model_name: str, model_type: str):
        key = f"{model_name}_{model_type}"
        if key not in self.models:
            try:
                model_path = os.path.join(self.model_dir, f"{model_name}.{'pkl' if model_type != 'keras' else 'keras'}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file {model_path} does not exist")
                if model_type == "pickle":
                    self.models[key] = pickle.load(open(model_path, "rb"))
                elif model_type == "joblib":
                    self.models[key] = joblib.load(model_path)
                elif model_type == "keras":
                    self.models[key] = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded model: {key}")
            except Exception as e:
                logger.error(f"Failed to load model {key}: {e}")
                raise
        return self.models[key]

model_registry = ModelRegistry(MODEL_DIR)

@contextmanager
def temp_file(file, filename: str):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        file.save(file_path)
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up file {file_path}: {e}")

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def extract_email_features(headers: Union[str, Dict]) -> pd.DataFrame:
    if isinstance(headers, str):
        try:
            with open(headers, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            headers_dict = dict(msg.items())
        except Exception as e:
            logger.error(f"Failed to parse .eml file {headers}: {e}")
            raise ValueError(f"Invalid .eml file: {e}")
    else:
        headers_dict = headers

    is_missing = lambda h: int(h not in headers_dict)
    contains = lambda h, kw: int(kw.lower() in headers_dict.get(h, '').lower())
    domain_match = lambda h1, h2: int(
        bool(re.search(r'@([\w\.-]+)', headers_dict.get(h1, ''))) and
        bool(re.search(r'@([\w\.-]+)', headers_dict.get(h2, ''))) and
        re.search(r'@([\w\.-]+)', headers_dict.get(h1, '')).group(1) ==
        re.search(r'@([\w\.-]+)', headers_dict.get(h2, '')).group(1)
    )

    span_time = 0
    if "Date" in headers_dict and "Received" in headers_dict:
        try:
            dt = datetime.strptime(headers_dict["Date"][:31], "%a, %d %b %Y %H:%M:%S %z")
            rec = headers_dict.get("Received", "").split(";")[-1].strip()
            if rec:
                rec_dt = datetime.strptime(rec[:31], "%a, %d %b %Y %H:%M:%S %z")
                span_time = abs((dt - rec_dt).total_seconds())
        except (ValueError, IndexError):
            logger.debug("Failed to parse Date or Received for span_time")
            span_time = 0

    def get_x_priority(s: str) -> int:
        if not s:
            return 3
        part = s.split()[0]
        priority_map = {"High": 1, "Normal": 3, "Low": 5}
        if part in priority_map:
            return priority_map[part]
        try:
            return int(part)
        except ValueError:
            return 3

    feats = {
        "missing_list-subscribe": is_missing("List-Subscribe"),
        "missing_precedence": is_missing("Precedence"),
        "missing_list-help": is_missing("List-Help"),
        "missing_list-post": is_missing("List-Post"),
        "missing_list-unsubscribe": is_missing("List-Unsubscribe"),
        "missing_x-spam-status": is_missing("X-Spam-Status"),
        "missing_list-id": is_missing("List-ID"),
        "str_precedence_list": contains("Precedence", "list"),
        "missing_x-spam-checker-version": is_missing("X-Spam-Checker-Version"),
        "time_zone": int(bool(re.search(r'([+-]\d{4})', headers_dict.get("Date", "")))),
        "missing_sender": is_missing("SenderA"),
        "domain_match_to_received": domain_match("To", "Received"),
        "domain_match_errors-to_sender": domain_match("Errors-To", "Sender"),
        "missing_x-mailer": is_missing("X-Mailer"),
        "missing_errors-to": is_missing("Errors-To"),
        "missing_x-priority": is_missing("X-Priority"),
        "str_return-path_bounce": contains("Return-Path", "bounce"),
        "missing_x-beenthere": is_missing("X-Beenthere"),
        "str_message-ID_dollar": contains("Message-ID", "$"),
        "missing_delivered-to": is_missing("Delivered-To"),
        "missing_references": is_missing("References"),
        "missing_content-disposition": is_missing("Content-Disposition"),
        "number_replies": headers_dict.get("Subject", "").lower().count("re:"),
        "span_time": span_time,
        "date_comp_date_received": int("Date" in headers_dict and "Received" in headers_dict),
        "x-priority": get_x_priority(headers_dict.get("X-Priority", "")),
        "missing_reply-to": is_missing("Reply-To")
    }

    return pd.DataFrame([[feats[f] for f in EMAIL_FEATURES]], columns=EMAIL_FEATURES)

def extract_pdf_features(pdf_path: str) -> Dict:
    features = {feature: 0 for feature in EXPECTED_FEATURES}
    features["PdfSize"] = os.path.getsize(pdf_path) / 1024.0  # Always extract file size

    try:
        pdf = PdfReader(pdf_path)
        features["MetadataSize"] = len(str(pdf.metadata)) if pdf.metadata else 0
        features["Pages"] = len(pdf.pages)
        features["TitleCharacters"] = len(pdf.metadata.get("/Title", "")) if pdf.metadata else 0
        features["isEncrypted"] = 1 if pdf.is_encrypted else 0
        features["PageNo"] = len(pdf.pages)
        
        try:
            text = extract_text(pdf_path, laparams=LAParams())
            features["Text"] = len(text) if text else 0
            features["JS"] = 1 if "javascript" in text.lower() else 0
            features["Javascript"] = features["JS"]
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF {pdf_path}: {e}")
            features["Text"] = 0
        
        features["Images"] = 1 if features["Pages"] > 0 and features["PdfSize"] / features["Pages"] > 50 else 0
    except Exception as e:
        if "EOF marker not found" in str(e):
            logger.warning(f"PDF {pdf_path} has EOF marker issue, proceeding with partial features: {e}")
        else:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        # Keep features with defaults (PdfSize is already set, others remain 0)
    
    return features

def compute_ensemble_score(rf_prob: float, svm_prob: float, rf_weight: float = 0.6, svm_weight: float = 0.4) -> float:
    total_w = rf_weight + svm_weight
    return (rf_weight * rf_prob + svm_weight * svm_prob) / total_w

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    threshold = 0.5
    try:
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400

            filename = secure_filename(file.filename)
            with temp_file(file, filename) as file_path:
                # Xử lý file Email (.eml)
                if allowed_file(filename, ALLOWED_EMAIL_EXTENSION):
                    try:
                        with open(file_path, 'rb') as f:
                            msg = BytesParser(policy=policy.default).parse(f)
                        headers = dict(msg.items())
                        df = extract_email_features(headers)
                        
                        rf_model = model_registry.load_model("random_forest_email", "joblib")
                        svm_model = model_registry.load_model("svm_model_email", "joblib")
                        scaler = model_registry.load_model("scaler_email", "joblib")
                        
                        df_scaled = scaler.transform(df)
                        
                        rf_p = rf_model.predict_proba(df)[:, 1][0]
                        svm_p = svm_model.predict_proba(df_scaled)[:, 1][0]
                        
                        ensemble = compute_ensemble_score(rf_p, svm_p)
                        result = "Phishing" if ensemble > threshold else "Legitimate"
                        
                        return jsonify({
                            "rf_confidence": round(float(rf_p), 4),
                            "svm_confidence": round(float(svm_p), 4),
                            "ensemble_confidence": round(float(ensemble), 4),
                            "result": result
                        }), 200
                    except Exception as e:
                        logger.error(f"Error processing .eml file: {e}")
                        return jsonify({"error": f"Failed to process .eml file: {str(e)}"}), 400

                # Xử lý file ảnh
                elif allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                    image = cv2.imread(file_path)
                    if image is None:
                        return jsonify({"error": "Invalid image file"}), 400

                    qr_codes = decode(image)
                    if qr_codes:
                        qr_results = []
                        for qr in qr_codes:
                            url = qr.data.decode("utf-8")
                            logger.info(f"Detected QR code: {url}")
                            rf_features = pd.DataFrame([extract_features(url)])
                            rf_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(rf_features)[:, 1][0]
                            vectorizer = model_registry.load_model("vectorizer_URL", "pickle")
                            svm_pred = model_registry.load_model("svm_model_URL", "pickle").decision_function(vectorizer.transform([url]))
                            svm_conf = 1 / (1 + np.exp(-svm_pred[0]))
                            ensemble = compute_ensemble_score(rf_pred, svm_conf)
                            res = "Phishing" if ensemble > threshold else "Legitimate"
                            qr_results.append({
                                "qr_url": url,
                                "rf_confidence": round(float(rf_pred), 4),
                                "svm_confidence": round(float(svm_conf), 4),
                                "ensemble_confidence": round(float(ensemble), 4),
                                "result": res
                            })
                        return jsonify({"qr_results": qr_results}), 200

                    try:
                        cnn_input = cv2.resize(image, (128, 128)).astype("float32") / 255.0
                        cnn_input = np.expand_dims(cnn_input, axis=0)
                        cnn_pred = float(model_registry.load_model("cnn_phishing_image", "keras").predict(cnn_input)[0][0])
                    except Exception as e:
                        logger.error(f"CNN prediction failed: {e}")
                        cnn_pred = None

                    try:
                        rf_features_img = preprocess_image(file_path).flatten().reshape(1, -1)
                        rf_pred_img = float(model_registry.load_model("rf_image_model", "joblib").predict_proba(rf_features_img)[:, 1][0])
                    except Exception as e:
                        logger.error(f"RF image prediction failed: {e}")
                        rf_pred_img = None

                    if cnn_pred is not None and rf_pred_img is not None:
                        ensemble = compute_ensemble_score(rf_pred_img, cnn_pred)
                        res = "Phishing" if ensemble > threshold else "Legitimate"
                    else:
                        ensemble = None
                        res = "Error"

                    return jsonify({
                        "rf_confidence": round(float(rf_pred_img), 4) if rf_pred_img is not None else "Error",
                        "cnn_confidence": round(float(cnn_pred), 4) if cnn_pred is not None else "Error",
                        "ensemble_confidence": round(float(ensemble), 4) if ensemble is not None else "Error",
                        "result": res
                    }), 200

                # Xử lý file PDF
                elif allowed_file(filename, ALLOWED_PDF_EXTENSION):
                    features = extract_pdf_features(file_path)
                    feature_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)
                    rf_prob = model_registry.load_model("random_forest_file", "pickle").predict_proba(feature_df)[0][1]
                    scaler = model_registry.load_model("scaler_file", "pickle")
                    features_scaled = scaler.transform(feature_df)
                    svm_prob = model_registry.load_model("svm_model_file", "pickle").predict_proba(features_scaled)[0][1]
                    ensemble = compute_ensemble_score(rf_prob, svm_prob)
                    res = "Phishing" if ensemble > threshold else "Legitimate"
                    return jsonify({
                        "rf_confidence": round(float(rf_prob), 4),
                        "svm_confidence": round(float(svm_prob), 4),
                        "ensemble_confidence": round(float(ensemble), 4),
                        "result": res
                    }), 200

                else:
                    return jsonify({"error": "Invalid file type"}), 400

        # Xử lý JSON input (URL hoặc text)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid input data"}), 400

        if "url" in data:
            url = data["url"].strip()
            if not url:
                return jsonify({"error": "URL is required"}), 400
            rf_features = pd.DataFrame([extract_features(url)])
            rf_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(rf_features)[:, 1][0]
            vectorizer = model_registry.load_model("vectorizer_URL", "pickle")
            svm_pred = model_registry.load_model("svm_model_URL", "pickle").decision_function(vectorizer.transform([url]))
            svm_conf = 1 / (1 + np.exp(-svm_pred[0]))
            ensemble = compute_ensemble_score(rf_pred, svm_conf)
            res = "Phishing" if ensemble > threshold else "Legitimate"
            return jsonify({
                "rf_confidence": round(float(rf_pred), 4),
                "svm_confidence": round(float(svm_conf), 4),
                "ensemble_confidence": round(float(ensemble), 4),
                "url": url,
                "result": res
            }), 200

        if "text" in data:
            text = data["text"].strip()
            if not text:
                return jsonify({"error": "Text is required"}), 400
            vectorizer = model_registry.load_model("vectorizer_URL", "pickle")
            text_features = vectorizer.transform([text])
            text_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(text_features)[:, 1][0]
            res = "Phishing" if text_pred > threshold else "Legitimate"
            return jsonify({
                "text_confidence": round(float(text_pred), 4),
                "result": res
            }), 200

        return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_header", methods=["POST"])
def predict_header():
    threshold = 0.5
    try:
        data = request.get_json()
        if not data or "headers" not in data:
            return jsonify({"error": "Headers are required"}), 400

        headers = data["headers"]
        if not isinstance(headers, dict):
            return jsonify({"error": "Headers must be a dictionary"}), 400

        logger.info("Processing email headers")
        
        df = extract_email_features(headers)
        
        rf_model = model_registry.load_model("random_forest_email", "joblib")
        svm_model = model_registry.load_model("svm_model_email", "joblib")
        scaler = model_registry.load_model("scaler_email", "joblib")
        
        df_scaled = scaler.transform(df)
        
        rf_p = rf_model.predict_proba(df)[:, 1][0]
        svm_p = svm_model.predict_proba(df_scaled)[:, 1][0]
        
        ensemble = compute_ensemble_score(rf_p, svm_p)
        result = "Phishing" if ensemble > threshold else "Legitimate"
        
        return jsonify({
            "rf_confidence": round(float(rf_p), 4),
            "svm_confidence": round(float(svm_p), 4),
            "ensemble_confidence": round(float(ensemble), 4),
            "result": result
        }), 200

    except Exception as e:
        logger.error(f"Error in /predict_header: {e}")
        return jsonify({
            "result": "Error",
            "ensemble_confidence": 0.0,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)