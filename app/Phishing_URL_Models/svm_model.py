import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

data_path = '../dataset_URL/phishing_URL.csv'
data = pd.read_csv(data_path)

if 'URL' not in data.columns or 'Label' not in data.columns:
    raise ValueError("Dataset không chứa các cột bắt buộc: 'URL' và 'Label'")

data = data.dropna(subset=['URL', 'Label'])
data = data[data['Label'].isin([0, 1])] 

label_mapping = {0: 'Legitimate', 1: 'Phishing'}
data['Label'] = data['Label'].map(label_mapping)

max_words = 5000
vectorizer = TfidfVectorizer(max_features=max_words)
X = vectorizer.fit_transform(data['URL'])
y = data['Label']

vectorizer_path = "../models/vectorizer_URL.pkl"
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer đã được lưu tại: {vectorizer_path}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Số lượng mẫu train: {X_train.shape[0]}, Số lượng mẫu test: {X_test.shape[0]}")

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)

cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
plt.plot(cv_scores, label='Accuracy per fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('SVM Model Accuracy (Cross-validation)')
plt.legend(loc='upper right')
plt.show()

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
labels = ['Phishing', 'Legitimate']
cm = confusion_matrix(y_test, y_pred, labels=labels)

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) != 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) != 0 else 0

print("\nĐánh giá mô hình:")
print(f"- Độ chính xác (Accuracy): {accuracy_manual:.2f}")
print(f"- Độ chính xác dự đoán Phishing (Precision): {precision_manual:.2f}")
print(f"- Khả năng nhận diện Phishing đúng (Recall): {recall_manual:.2f}")

model_path = "../models/svm_model_URL.pkl"
with open(model_path, "wb") as f:
    pickle.dump(svm_model, f)
print(f"Model đã được lưu tại: {model_path}")
