import os
import time
import torch
from model import Model


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def train(args):
    # Initialize model
    print(f"Training started at {time.time()}"

    model = Model(args)
    for epoch in range(args.epochs):
        train_one_epoch(model)


if __name__ == "__main__":
    args = get_args()
    train(args)
