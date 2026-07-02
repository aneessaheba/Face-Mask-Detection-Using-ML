import argparse
import os

from tensorflow.keras.models import load_model

from src.data_utils import load_dataset, split_dataset
from src.metrics import compute_binary_metrics, save_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained face mask detector")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--metrics-output", type=str, default="reports/evaluation_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)

    model = load_model(args.model)
    images, labels = load_dataset(args.data_dir)
    _, x_test, _, y_test = split_dataset(images, labels, test_size=0.2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0).ravel()
    metrics = compute_binary_metrics(y_test, predictions)
    metrics["test_loss"] = float(loss)
    metrics["test_accuracy"] = float(accuracy)
    save_metrics_report(metrics, args.metrics_output)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved evaluation metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
