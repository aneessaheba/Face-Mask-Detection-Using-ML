import argparse
import sys

from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.infer import main as infer_main


def create_parser():
    parser = argparse.ArgumentParser(description="Face Mask Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--output", type=str, default="models/mask_detector.h5")
    train_parser.add_argument("--history-output", type=str, default="reports/training_history.json")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", type=str, required=True)
    eval_parser.add_argument("--data-dir", type=str, required=True)
    eval_parser.add_argument("--metrics-output", type=str, default="reports/evaluation_metrics.json")

    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument("--model", type=str, required=True)
    infer_parser.add_argument("--image", type=str, required=True)
    infer_parser.add_argument("--output", type=str, default="output.jpg")

    return parser


def train(args=None):
    parser = create_parser()
    if args is None:
        args = sys.argv[1:]
    parsed = parser.parse_args(args)
    if parsed.command == "train":
        sys.argv = ["train.py", f"--data-dir={parsed.data_dir}", f"--epochs={parsed.epochs}", f"--batch-size={parsed.batch_size}", f"--output={parsed.output}", f"--history-output={parsed.history_output}"]
        train_main()


def evaluate(args=None):
    parser = create_parser()
    if args is None:
        args = sys.argv[1:]
    parsed = parser.parse_args(args)
    if parsed.command == "evaluate":
        sys.argv = ["evaluate.py", f"--model={parsed.model}", f"--data-dir={parsed.data_dir}", f"--metrics-output={parsed.metrics_output}"]
        evaluate_main()


def infer(args=None):
    parser = create_parser()
    if args is None:
        args = sys.argv[1:]
    parsed = parser.parse_args(args)
    if parsed.command == "infer":
        sys.argv = ["infer.py", f"--model={parsed.model}", f"--image={parsed.image}", f"--output={parsed.output}"]
        infer_main()


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        sys.argv = ["train.py", f"--data-dir={args.data_dir}", f"--epochs={args.epochs}", f"--batch-size={args.batch_size}", f"--output={args.output}", f"--history-output={args.history_output}"]
        train_main()
    elif args.command == "evaluate":
        sys.argv = ["evaluate.py", f"--model={args.model}", f"--data-dir={args.data_dir}", f"--metrics-output={args.metrics_output}"]
        evaluate_main()
    elif args.command == "infer":
        sys.argv = ["infer.py", f"--model={args.model}", f"--image={args.image}", f"--output={args.output}"]
        infer_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
