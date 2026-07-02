import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run live face mask detection from a webcam")
    parser.add_argument("--model", type=str, default="models/mask_detector.h5")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = None
    if os.path.exists(args.model):
        model = load_model(args.model)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load face detection model")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            if model is not None:
                resized = cv2.resize(face, (128, 128))
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                prediction = model.predict(np.expand_dims(resized_rgb, axis=0), verbose=0)[0][0]
                label = "Mask" if prediction >= 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.putText(frame, f"{label}: {prediction:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (255, 0, 0)
                cv2.putText(frame, "Model not loaded", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
