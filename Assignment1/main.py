"""main.py: A python file implementing image transformation functions and creating augmented dataset
using the transformations."""

__author__ = "Gurbaaz Singh Nandra"


import os
import sys
import random
import logging
import pickle
import numpy as np


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


def random_rotation(input_image, log):
    """
    Random Rotation in the range [−180 degree, +180 degree]

    Input:
        Input image
    Output:
        Transformed image
    """
    theta = random.randint(-180, 180)

    log.info(f"Rotating the image by {theta} degrees")


def random_cutout(input_image, log):
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


def random_crop(input_image, log):
    """
    Random Crop (Add a padding of 2 pixels on all sides and randomly select a block of 32x32
    pixels from the padded image)

    Input:
        Input image
    Output:
        Transformed image
    """
    pass


def contrast_and_horizontal_flipping(input_image, log):
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
                images.append(np.array(batch_dataset[b"data"][i]).reshape(3, 1024))

        elif file.startswith("batches"):
            label_names = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))[
                b"label_names"
            ]
            label_names = [label.decode("utf-8") for label in label_names]
        
        elif file.startswith("test_batch"):
            test_batch = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))
            test_dataset = {"filenames": [], "labels": [], "images": []}

            for i in range(IMAGES_PER_BATCH):
                test_dataset["filenames"].append(test_batch[b"filenames"][i])
                test_dataset["labels"].append(test_batch[b"labels"][i])
                test_dataset["images"].append(np.array(test_batch[b"data"][i]).reshape(3, 1024))

    train_dataset = {"filenames": filenames, "labels": labels, "images": images}

    log.info(f"Size of train dataset: {len(train_dataset['filenames'])}")
    log.info(f"Size of test dataset: {len(test_dataset['filenames'])}")
    log.info(f"Labels in dataset: {label_names}")

    ## 2. Image transformations

    example_idx = random.randint(0, NUM_TRAIN_BATCHES * IMAGES_PER_BATCH - 1)
    example_image = train_dataset["images"][example_idx]

    log.info(f"Example test image is {train_dataset['filenames'][example_idx].decode('utf-8')}")

    random_rotation(example_image, log)
    random_cutout(example_image, log)
    random_crop(example_image, log)
    contrast_and_horizontal_flipping(example_image, log)

    return 0
    ## 3. Generating augmented training set

    for i in range(NUM_TRAIN_BATCHES * IMAGES_PER_BATCH):
        operation = random.randint(0, 3)
        og_image = train_dataset["images"][i]

        if operation == 0:
            augmented_image = random_rotation(og_image, log)
        elif operation == 1:
            augmented_image = random_cutout(og_image, log)
        elif operation == 2:
            augmented_image = random_crop(og_image, log)
        else:
            augmented_image = contrast_and_horizontal_flipping(og_image, log)

        train_dataset["filenames"].append(b"augmented_" + train_dataset["filenames"][i])
        train_dataset["labels"].append(train_dataset["labels"][i])
        train_dataset["images"].append(augmented_image)

    log.info(len(train_dataset["filenames"]))

    ## 4. Feature extraction

    ## 5. MLP implementation

    ## 6. Back-propagation

    ## 7. Performance evaluation


if __name__ == "__main__":
    sys.exit(main())
