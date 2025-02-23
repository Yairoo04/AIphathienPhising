import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_dataset_paths, preprocess_image
import pickle


# Đường dẫn dataset
dataset_path = "../dataset"

# Load dữ liệu từ thư mục
image_paths, labels = load_dataset_paths(dataset_path)

# Debug: Kiểm tra xem có ảnh nào được load không
if len(image_paths) == 0:
    raise ValueError("Không tìm thấy ảnh trong thư mục dataset! Kiểm tra lại đường dẫn và cấu trúc thư mục.")

# Chuyển ảnh thành đặc trưng (feature vectors)
X_features = np.array([preprocess_image(img_path).flatten() for img_path in image_paths])
y_labels = np.array(labels)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Debug: Kiểm tra số lượng mẫu sau khi chia
print(f"Số lượng mẫu train: {len(X_train)}, Số lượng mẫu test: {len(X_test)}")

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Độ chính xác của Random Forest: {accuracy:.4f}")

# Save model
with open("../models/rf_image_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Mô hình Random Forest đã được lưu thành công!")
