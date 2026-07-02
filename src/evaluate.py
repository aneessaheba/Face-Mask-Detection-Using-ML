import argparse

import numpy as np
from tensorflow.keras.models import load_model

from src.data_utils import load_dataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained face mask detector")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)
    images, labels = load_dataset(args.data_dir)
    x_train, x_test, y_train, y_test = split_dataset(images, labels, test_size=0.2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
