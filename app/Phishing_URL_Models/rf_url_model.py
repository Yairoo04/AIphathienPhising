import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features

# Load dataset
data_path = '../data/phishing.csv'
data = pd.read_csv(data_path)

if 'URL' not in data.columns or 'Label' not in data.columns:
    raise ValueError("Dataset không chứa các cột bắt buộc: 'URL' và 'Label'")

data = data.dropna(subset=['URL', 'Label']) 
data = data[data['Label'].isin([0, 1])]

# Gán nhãn: 0 - Legitimate, 1 - Phishing
label_mapping = {0: 'Legitimate', 1: 'Phishing'}
data['Label'] = data['Label'].map(label_mapping)

# Trích xuất các đặc trưng từ URL
features_list = []
valid_labels = []

for url, label in zip(data['URL'], data['Label']):
    features = extract_features(url)
    if features and all(value != -1 for value in features.values()):
        features_list.append(features)
        valid_labels.append(label)

if not features_list:
    raise ValueError("Không có URL hợp lệ nào được trích xuất. Kiểm tra dữ liệu đầu vào!")

features_df = pd.DataFrame(features_list)
features_df['Label'] = valid_labels

X = features_df.drop(columns=['Label'])
y = features_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Số lượng mẫu train: {len(X_train)}, Số lượng mẫu test: {len(X_test)}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

model_path = "../models/random_forest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"Model đã được lưu tại: {model_path}")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của Random Forest: {accuracy:.2f}")
