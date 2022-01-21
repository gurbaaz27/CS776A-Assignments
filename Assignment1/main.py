"""main.py: A python file implementing image transformation functions and creating augmented dataset
using the transformations."""

__author__ = "Gurbaaz Singh Nandra"


import os
import sys
import random
import logging
import pickle


def enable_logging() -> logging.Logger:
    """
    Returns a formatted logger for stdout and file io.
    """
    LOG_FILE = "script.log"
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(asctime)s\n%(message)s")

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


def random_rotation(input_image):
    """
    Random Rotation in the range [−180 degree, +180 degree]

    Input:
        Input image
    Output:
        Transformed image
    """
    pass


def random_cutout(input_image):
    """
    Random cutout (randomly erase a block of pixels from the image with the width and height
    of the block in the range 0 to 16 pixels. The erased part (cutout) should be filled with a
    single value)

    Input:
        Input image
    Output:
        Transformed image
    """
    pass


def random_crop(input_image):
    """
    Random Crop (Add a padding of 2 pixels on all sides and randomly select a block of 32x32
    pixels from the padded image)

    Input:
        Input image
    Output:
        Transformed image
    """
    pass


def contrast_and_horizontal_flipping(input_image):
    """
    Contrast & Horizontal flipping. (First, change the contrast of the image with a factor of
    α randomly selected from the range (0.5, 2.0) and then flip the image horizontally with a
    probability of 0.5)

    Input:
        Input image
    Output:
        Transformed image
    """
    pass


def main():
    """
    Entry point of script
    """
    log = enable_logging()

    CIFAR_DATASET_FILENAME = "cifar-10-batches-py"
    IMAGES_PER_BATCH = 10000
    NUM_TRAIN_BATCHES = 5

    filenames = []
    labels = []
    images = []

    ## 1. Loading the dataset

    for file in os.listdir(CIFAR_DATASET_FILENAME):
        if file.startswith("data"):
            batch_dataset = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))

            for i in range(IMAGES_PER_BATCH):
                filenames.append(batch_dataset[b"filenames"][i])
                labels.append(batch_dataset[b"labels"][i])
                images.append(batch_dataset[b"data"][i])

        elif file.startswith("batches"):
            label_names = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))[
                b"label_names"
            ]

    train_dataset = {"filenames": filenames, "labels": labels, "images": images}

    log.info(len(train_dataset["filenames"]))
    log.info(label_names)

    ## 2. Image transformations

    example_idx = random.randint(0, NUM_TRAIN_BATCHES * IMAGES_PER_BATCH - 1)
    example_image = train_dataset["images"][example_idx]

    random_rotation(example_image)
    random_cutout(example_image)
    random_crop(example_image)
    contrast_and_horizontal_flipping(example_image)

    ## 3. Generating augmented training set

    ## 4. Feature extraction

    ## 5. MLP implementation

    ## 6. Back-propagation

    ## 7. Performance evaluation


if __name__ == "__main__":
    sys.exit(main())
