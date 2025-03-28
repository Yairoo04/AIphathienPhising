import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data, load_data

CSV_PATH = "../dataset_File/data_File.csv"

def rf_model():
    df = load_data(CSV_PATH)
    print("- Thống kê file Benign:")
    print(df[df['Class'].str.lower() == 'benign'].describe())
    
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"+ Số lượng mẫu train: {len(X_train)}")
    print(f"+ Số lượng mẫu test: {len(X_test)}")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "../models/random_forest_file.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Model đã được lưu tại: {model_path}")

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"- Random Forest Accuracy: {acc:.4f}")

if __name__ == "__main__":
    rf_model()