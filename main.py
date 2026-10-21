"""Command-line entry point for the educational image-classification demo."""
import argparse
import sys

if sys.version_info[:2] != (3, 12):
    sys.exit(
        "This project is configured for Python 3.12. "
        "Windows: run setup_windows.bat, then run_windows.bat <command>. "
        "macOS/Linux: use .venv/bin/python main.py <command>."
    )

from src.evaluate import evaluate_model
from src.predict import predict_image
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Educational PneumoniaMNIST CNN demo")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="download data and train a CNN")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = commands.add_parser("evaluate", help="evaluate saved model on test split")
    eval_parser.add_argument("--model", default="models/pneumonia_cnn.weights.h5")

    predict_parser = commands.add_parser("predict", help="score one image (not a diagnosis)")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--model", default="models/pneumonia_cnn.weights.h5")

    args = parser.parse_args()
    if args.command == "train":
        train_model(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    elif args.command == "evaluate":
        evaluate_model(args.model)
    elif args.command == "predict":
        predict_image(args.model, args.image)


if __name__ == "__main__":
    main()
