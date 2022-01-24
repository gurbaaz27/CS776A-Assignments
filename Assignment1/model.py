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


UNAUGMENTED_TRAIN_SIZE = 100000
UNAUGMENTED_TRAIN_SIZE = 50000
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


def save_image(image, filename):
    """
    Expects a (channels=3, height=32, width=32) shaped numpy array and saves it as image
    """
    Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8)).save(
        filename
    )
    

def main():
    """
    Entry point of script
    """
    log = enable_logging()

    unaugmented_dataset = unpickle("unaugmented_dataset")
    augmented_dataset = unpickle("augmented_dataset")

    ## 4. Feature extraction

    ## 5. MLP implementation

    ## 6. Back-propagation

    ## 7. Performance evaluation


if __name__ == "__main__":
    sys.exit(main())
