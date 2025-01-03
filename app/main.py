from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from feature_extraction import extract_features
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models
lstm_model = load_model("models/lstm_model.keras")
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_url(url, tokenizer, max_len=200):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequence = tokenizer.texts_to_sequences([url])
    return pad_sequences(sequence, maxlen=max_len)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Phishing Detection API! a'})

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    lstm_input = preprocess_url(url, tokenizer)
    lstm_prediction = float(lstm_model.predict(lstm_input)[0][0])

    rf_features = pd.DataFrame([extract_features(url)])
    rf_prediction_proba = rf_model.predict_proba(rf_features)
    rf_prediction = float(rf_prediction_proba[:, 1][0])

    ensemble_score = (lstm_prediction + rf_prediction) / 2
    result = "Phishing" if ensemble_score > 0.5 else "Legitimate"

    return jsonify({
        'url': url,
        'result': result,
        'lstm_confidence': round(lstm_prediction, 4),
        'rf_confidence': round(rf_prediction, 4),
        'ensemble_confidence': round(ensemble_score, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
