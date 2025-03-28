import pandas as pd

def preprocess_data(df):
    if 'FileName' in df.columns:
        df = df.rename(columns={"FileName": "filename"})
    
    df['label'] = df['Class'].apply(lambda x: 1 if str(x).strip().lower() == 'malicious' else 0)
    
    X = df.drop(columns=["filename", "label", "Class"], errors="ignore")
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0)
    
    return X, df["label"]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    print("- Dữ liệu đầu vào (5 dòng):")
    print(df.head())
    
    return df