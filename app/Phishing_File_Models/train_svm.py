import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"- Độ chính xác: {accuracy:.4f}")
        
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