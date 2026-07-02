import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = load_model(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128)).astype("float32") / 255.0
    image_batch = np.expand_dims(image_resized, axis=0)

    prediction = model.predict(image_batch, verbose=0)[0][0]
    label = "mask" if prediction >= 0.5 else "no mask"
    color = (0, 255, 0) if label == "mask" else (0, 0, 255)
    cv2.putText(image, f"{label}: {prediction:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(args.output, image)
    print(f"Prediction: {label} ({prediction:.2f})")
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
