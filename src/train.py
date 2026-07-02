import argparse
import os

from tensorflow.keras.callbacks import ModelCheckpoint

from src.data_utils import load_dataset, split_dataset
from src.metrics import save_metrics_report
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face mask detector")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="models/mask_detector.h5")
    parser.add_argument("--history-output", type=str, default="reports/training_history.json")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_output), exist_ok=True)

    images, labels = load_dataset(args.data_dir)
    x_train, x_val, y_train, y_val = split_dataset(images, labels)

    model = build_model()
    checkpoint = ModelCheckpoint(args.output, save_best_only=True, monitor="val_accuracy")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
        verbose=1,
    )

    save_metrics_report(history.history, args.history_output)
    print(f"Training complete. Model saved to {args.output}")
    print(f"Training history saved to {args.history_output}")


if __name__ == "__main__":
    main()
