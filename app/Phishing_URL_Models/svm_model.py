import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load dataset
data_path = '../data/phishing.csv'
data = pd.read_csv(data_path)

# Kiểm tra và xử lý dữ liệu
if 'URL' not in data.columns or 'Label' not in data.columns:
    raise ValueError("Dataset không chứa các cột bắt buộc: 'URL' và 'Label'")

data = data.dropna(subset=['URL', 'Label'])
data = data[data['Label'].isin([0, 1])] 

# Gán nhãn: 0 - Legitimate, 1 - Phishing
label_mapping = {0: 'Legitimate', 1: 'Phishing'}
data['Label'] = data['Label'].map(label_mapping)

# Cấu hình các tham số
max_words = 5000

# vectorizer
vectorizer = TfidfVectorizer(max_features=max_words)
X = vectorizer.fit_transform(data['URL'])
y = data['Label']

vectorizer_path = "../models/vectorizer.pkl"
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Vectorizer đã được lưu tại: {vectorizer_path}")

# Tách dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Số lượng mẫu train: {X_train.shape[0]}, Số lượng mẫu test: {X_test.shape[0]}")

# Xây dựng mô hình SVM
svm_model = SVC(kernel='linear', random_state=42)
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')

# Vẽ đồ thị accuracy của cross-validation
plt.plot(cv_scores, label='Accuracy per fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('SVM Model Accuracy (Cross-validation)')
plt.legend(loc='upper right')
plt.show()

# Huấn luyện
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của SVM trên tập test: {accuracy:.2f}")

# Lưu
model_path = "../models/svm_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(svm_model, f)

print(f"Model đã được lưu tại: {model_path}")
