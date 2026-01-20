import os
import sys
import time
import logging

from utils import load_config


def main():
    setup_logging()
    config = load_config("config.yaml")
    print(f"Loaded config: {config}")


if __name__ == "__main__":
    main()
