import os
from typing import Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(data_dir: str, image_size: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = 1 if class_name.lower() == "mask" else 0
        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(tuple(VALID_EXTENSIONS)):
                continue
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(label)

    if not images:
        raise ValueError(f"No valid images found in {data_dir}")

    return np.array(images), np.array(labels)


def split_dataset(images: np.ndarray, labels: np.ndarray, test_size: float = 0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels)
