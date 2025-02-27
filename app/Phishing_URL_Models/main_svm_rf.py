from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
from feature_extraction import extract_features

app = Flask(__name__)
CORS(app) 

# Load các mô hình và vector
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Hàm tiền xử lý URL cho SVM
def preprocess_url_svm(url, vectorizer):
    features = vectorizer.transform([url])
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Dự đoán bằng Random Forest
    rf_features = pd.DataFrame([extract_features(url)]) 
    rf_prediction_proba = rf_model.predict_proba(rf_features)
    rf_prediction = float(rf_prediction_proba[:, 1][0])

    # Dự đoán bằng SVM
    svm_input = preprocess_url_svm(url, vectorizer)
    svm_prediction = svm_model.predict(svm_input)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
