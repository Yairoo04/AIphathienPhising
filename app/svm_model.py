import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt

# Tạo thư mục models nếu chưa tồn tại
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv('data/phishing.csv')

# Kiểm tra và làm sạch dữ liệu
data = data.dropna(subset=['URL', 'Label'])
data = data[data['Label'].isin([0, 1])]

# Cấu hình các tham số
max_words = 5000

# Tạo TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=max_words)
X = vectorizer.fit_transform(data['URL'])
y = data['Label']

# Lưu vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Tách dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình SVM
svm_model = SVC(kernel='linear', random_state=42)

# Sử dụng cross-validation để theo dõi accuracy trong suốt quá trình huấn luyện
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')

# Vẽ đồ thị accuracy của cross-validation
plt.plot(cv_scores, label='Accuracy per fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('SVM Model Accuracy (Cross-validation)')
plt.legend(loc='upper right')
plt.show()

# Huấn luyện mô hình trên toàn bộ dữ liệu training
svm_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = svm_model.predict(X_test)

# Tính accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# Lưu mô hình
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
