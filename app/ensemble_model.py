import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features
import pickle

# Tạo thư mục lưu model
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv('data/phishing.csv')

# Kiểm tra và làm sạch dữ liệu
data = data.dropna(subset=['URL', 'Label'])
data = data[data['Label'].isin([0, 1])]  # Chỉ giữ nhãn 0 và 1

# Trích xuất các đặc trưng
extracted_features = []
valid_indices = []

for idx, url in enumerate(data['URL']):
    features = extract_features(url)  # Gọi hàm extract_features
    if features is not None:  # Chỉ thêm các URL hợp lệ
        extracted_features.append(features)
        valid_indices.append(idx)

# Kiểm tra nếu không có URL hợp lệ
if not extracted_features:
    raise ValueError("Không có URL hợp lệ nào được trích xuất. Kiểm tra dữ liệu đầu vào!")

# Tạo DataFrame từ các đặc trưng hợp lệ
features_df = pd.DataFrame(extracted_features)
features_df['Label'] = data.iloc[valid_indices]['Label'].values

# Train-test split
X = features_df.drop(columns=['Label'])
y = features_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")
