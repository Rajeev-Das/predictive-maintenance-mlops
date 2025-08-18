"""
Main entry point for Predictive Maintenance MLOps workflow.

This script provides a command-line interface to train or evaluate the LSTM model
for Remaining Useful Life (RUL) prediction using the CMAPSS dataset.
"""
# main.py

import argparse
from src.models.train_lstm import main as train_lstm
from src.models.evaluate import main as eval_lstm


def run_train():
    """
    Run the training workflow for the LSTM model.

    This function prints a message indicating the start of training and calls the main
    training function from the LSTM training module.
    """
    print("\n=== Training LSTM Model ===")
    train_lstm()


def run_evaluate():
    """
    Run the evaluation workflow for the trained LSTM model.

    This function prints a message indicating the start of evaluation and calls the main
    evaluation function from the LSTM evaluation module.
    """
    print("\n=== Evaluating Trained LSTM Model on Test Set ===")
    eval_lstm()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance - Main Workflow Controller"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        required=True,
        help="Choose 'train' to train the LSTM, 'evaluate' for test set evaluation",
    )
    args = parser.parse_args()
    if args.mode == "train":
        run_train()
    elif args.mode == "evaluate":
        run_evaluate()
