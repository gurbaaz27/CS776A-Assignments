"""model.py: MLP model implementation"""

__author__ = "Gurbaaz Singh Nandra"


import os
import sys
import random
import logging
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm


CIFAR_DATASET_FILENAME = "cifar-10-batches-py"
IMAGES_PER_BATCH = 10000
NUM_TRAIN_BATCHES = 5
HEIGHT = 32
CHANNELS = 3


def enable_logging() -> logging.Logger:
    """
    Returns a formatted logger for stdout and file io.
    """
    LOG_FILE = "model.log"
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(levelname)s - %(asctime)s\n%(message)s")
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    log.addHandler(stdout_handler)
    log.addHandler(file_handler)

    return log


def unpickle(file):
    """
    Loads the data batch bytefile using pickle
    """
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def main():
    """
    Entry point of script
    """
    log = enable_logging()

    ## 4. Feature extraction

    ## 5. MLP implementation

    ## 6. Back-propagation

    ## 7. Performance evaluation


if __name__ == "__main__":
    sys.exit(main())
