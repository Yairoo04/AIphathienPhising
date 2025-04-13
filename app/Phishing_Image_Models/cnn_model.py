import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from data_loader import load_dataset
import os


def load_dataset(dataset_path, img_size=(128, 128), batch_size=32, augment=True):
    datagen_params = {
        "rescale": 1.0 / 255,
        "validation_split": 0.2  
    }

    if augment:
        datagen_params.update({
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True
        })

    train_datagen = ImageDataGenerator(**datagen_params)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

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

dataset_path = "../dataset_Image"
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