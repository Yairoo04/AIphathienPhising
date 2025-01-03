import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features
import pickle

os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv('data/phishing.csv')

# Kiểm tra và làm sạch dữ liệu
data = data.dropna(subset=['URL', 'Label'])
data = data[data['Label'].isin([0, 1])]

# Extract features
features = pd.DataFrame([extract_features(url) for url in data['URL']])
features['Label'] = data['Label']
features = features.dropna()

# Train-test split
X = features.drop(columns=['Label'])
y = features['Label']
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
