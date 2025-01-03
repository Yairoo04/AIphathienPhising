import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
max_len = 200

# Tokenize URLs
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['URL'])
sequences = tokenizer.texts_to_sequences(data['URL'])
X = pad_sequences(sequences, maxlen=max_len)
y = data['Label']

# Lưu tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Tách dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình LSTM
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_callback = ModelCheckpoint("models/best_model.keras", monitor='val_accuracy', 
                                       save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# Vẽ đồ thị accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()

# Lưu mô hình
model.save('models/lstm_model.keras')
