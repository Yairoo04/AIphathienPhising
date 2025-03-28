import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import preprocess_data, load_data

CSV_PATH = "../dataset_File/data_File.csv"

def svm_model():
    df = load_data(CSV_PATH)
    
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"+ Số lượng mẫu train: {len(X_train)}")
    print(f"+ Số lượng mẫu test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision_manual = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) != 0 else 0

    print("\nSử dụng công thức tính toán thủ công:")
    print(f"- Độ chính xác (Accuracy): {accuracy_manual:.2f}")
    print(f"- Độ chính xác dự đoán Phishing (Precision): {precision_manual:.2f}")
    print(f"- Khả năng nhận diện Phishing đúng (Recall): {recall_manual:.2f}")
    
    model_path = "../models/svm_model_file.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    scaler_path = "../models/scaler_file.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model đã được lưu tại: {model_path}")
    print(f"Scaler đã được lưu tại: {scaler_path}")

if __name__ == "__main__":
    svm_model()
