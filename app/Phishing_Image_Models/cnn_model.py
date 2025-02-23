import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, applications
import matplotlib.pyplot as plt
from data_loader import load_dataset

# Sử dụng Transfer Learning (EfficientNetB0)
def build_advanced_cnn(input_shape=(128, 128, 3)):
    base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Đóng băng các tầng để tránh overfitting

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Tham số huấn luyện
dataset_path = "../dataset"
epochs = 30
batch_size = 32

# Load dataset với augmentation
train_dataset, val_dataset = load_dataset(dataset_path, batch_size=batch_size, augment=True)

# Xây dựng mô hình mới
model = build_advanced_cnn()
model.summary()

# Callbacks để tối ưu
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Huấn luyện mô hình
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Vẽ biểu đồ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Lưu mô hình dưới định dạng mới
model.save("../models/cnn_phishing_image.keras")
print("Mô hình đã được lưu thành công!")
