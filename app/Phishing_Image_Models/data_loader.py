import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

def load_dataset(dataset_path, img_size=(128, 128), batch_size=32, augment=True):
    """
    Load dataset cho CNN từ thư mục, bao gồm augmentation và chia tập train/val.
    
    Args:
        dataset_path (str): Đường dẫn đến thư mục chứa dữ liệu.
        img_size (tuple): Kích thước ảnh đầu vào (mặc định 128x128).
        batch_size (int): Số lượng ảnh mỗi batch.
        augment (bool): Có sử dụng augmentation hay không.

    Returns:
        train_generator, val_generator: Bộ dữ liệu train và validation.
    """
    datagen_params = {
        "rescale": 1.0 / 255,
        "validation_split": 0.2  # 80% train, 20% validation
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

def load_dataset_paths(dataset_path):
    """
    Load danh sách đường dẫn của ảnh và nhãn tương ứng.
    - `phishing/` được gán nhãn 1
    - `legitimate/` được gán nhãn 0
    """
    image_paths = []
    labels = []

    # Duyệt qua các thư mục con
    for label, folder in enumerate(["legitimate", "phishing"]):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}, bỏ qua!")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Chỉ lấy file ảnh
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label)  # 1 nếu là phishing, 0 nếu là legitimate

    return image_paths, labels

def preprocess_image(image_path, img_size=(128, 128)):
    """
    Tiền xử lý ảnh: Resize về kích thước chuẩn và chuyển về dạng mảng numpy.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Lỗi: Không thể đọc ảnh {image_path}")
    
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
    return img

