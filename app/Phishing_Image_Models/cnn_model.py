import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, applications
import matplotlib.pyplot as plt
import os

def load_dataset(dataset_path, image_size=(128, 128), batch_size=32):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode='binary',  # Nhãn nhị phân: 0 (legitimate), 1 (phishing)
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=42
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42
    )
    
    return train_dataset, val_dataset

# Sử dụng Transfer Learning (EfficientNetB0)
def build_cnn(input_shape=(128, 128, 3)):
    base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False 

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

dataset_path = "../dataset"
epochs = 30
batch_size = 32

train_dataset, val_dataset = load_dataset(dataset_path, batch_size=batch_size)

model = build_cnn()
model.summary()

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

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

model.save("../models/cnn_phishing_image.keras")
print("Mô hình đã được lưu thành công!")