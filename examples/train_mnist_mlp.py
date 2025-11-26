#!/usr/bin/env python3
"""Train a Multi-Layer Perceptron (MLP) on MNIST dataset.

This script trains a simple feedforward neural network on the MNIST digit
classification task. It's designed to be used with hyperparameter optimization
sweeps, accepting all key hyperparameters as command-line arguments.

The script can run standalone or as part of a sweep benchmark. It outputs
the final validation accuracy to stdout in a parseable format.

Usage:
    # Basic usage with default parameters
    python train_mnist_mlp.py

    # Specify hyperparameters
    python train_mnist_mlp.py --learning_rate 0.01 --hidden_units 128 --num_layers 2

    # Set random seed for reproducibility
    python train_mnist_mlp.py --seed 42

Requirements:
    - torch
    - torchvision
    - numpy
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST classification.

    A flexible feedforward neural network with configurable depth and width.
    Uses ReLU activations and dropout for regularization.

    Args:
        input_size: Number of input features (784 for flattened MNIST images)
        num_layers: Number of hidden layers
        hidden_units: Number of units in each hidden layer
        dropout: Dropout probability (applied between hidden layers)
        num_classes: Number of output classes (10 for MNIST)
    """

    def __init__(
        self,
        input_size=784,
        num_layers=2,
        hidden_units=128,
        dropout=0.2,
        num_classes=10,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout_prob = dropout
        self.num_classes = num_classes

        layers = []

        if num_layers == 0:
            layers.append(nn.Linear(input_size, num_classes))
        else:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_units, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.network(x)


def load_data(data_dir, batch_size, val_split=0.1, seed=None):
    """Load and prepare MNIST dataset.

    Downloads MNIST data (if needed), applies transformations, and creates
    train/validation/test DataLoaders.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducible train/val split

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion):
    """Train model for one epoch.

    Args:
        model: Neural network model
        device: Device to train on (cpu or cuda)
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, device, data_loader, criterion):
    """Evaluate model on a dataset.

    Args:
        model: Neural network model
        device: Device to evaluate on
        data_loader: Data loader for evaluation
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate for SGD"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=128,
        help="Number of units in each hidden layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for MNIST data"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed training progress"
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print("Using device: {}".format(device))
        print("Hyperparameters:")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Momentum: {args.momentum}")
        print(f"  Hidden units: {args.hidden_units}")
        print(f"  Number of layers: {args.num_layers}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")

    os.makedirs(args.data_dir, exist_ok=True)

    train_loader, val_loader, test_loader = load_data(
        args.data_dir, args.batch_size, val_split=0.1, seed=args.seed
    )

    model = MLP(
        input_size=784,
        num_layers=args.num_layers,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        num_classes=10,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, device, val_loader, criterion)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        if args.verbose:
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                "Val Acc: {:.2f}%".format(val_accuracy)
            )

    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

    if args.verbose:
        print("\nFinal Results:")
        print(f"  Best Validation Accuracy: {best_val_accuracy:.2f}%")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")

    print(f"RESULT: val_accuracy={best_val_accuracy:.4f}")

    return best_val_accuracy


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}", file=sys.stderr)
        sys.exit(1)
