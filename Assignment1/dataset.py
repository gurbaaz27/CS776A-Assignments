"""dataset.py: A python file implementing image transformation functions and creating augmented dataset
using the transformations"""

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
    LOG_FILE = "script.log"
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


def random_rotation(input_image):
    """
    Random Rotation in the range [−180 degree, +180 degree]

    Input:
        Input image
    Output:
        Transformed image
    """
    theta = random.randint(-180, 180)

    theta = np.radians(-theta)

    output_image = np.zeros((3, 32, 32))

    for i in range(0, 32):
        for j in range(0, 32):
            cos, sin = np.cos(theta), np.sin(theta)
            ni = np.rint((i-15.5)*cos + (j-15.5)*sin + 15.5).astype('uint8')
            nj = np.rint((j-15.5)*cos - (i-15.5)*sin + 15.5).astype('uint8')
            if ni>=0 and ni<32 and nj>=0 and nj<32:
                for channel in range(CHANNELS):
                    output_image[channel][ni][nj] = input_image[channel][i][j]

    return output_image


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
    width = random.randint(0, 16)
    height = random.randint(0, 16)
    value = np.random.randint(0, 256, 3)

    ## top-left coordinates
    tlx = random.randint(0, 31 - width)
    tly = random.randint(0, 31 - height)

    ## bottom-right coordinates
    brx = random.randint(tlx, 31)
    bry = random.randint(tly, 31)

    output_image = np.copy(input_image)

    ## random cutout loop
    for channel in range(CHANNELS):
        for i in range(tlx, brx + 1):
            for j in range(tly, bry + 1):
                output_image[channel][i][j] = value[channel]

    return output_image


def random_crop(input_image):
    """
    Random Crop (Add a padding of 2 pixels on all sides and randomly select a block of 32x32
    pixels from the padded image)

    Input:
        Input image
    Output:
        Transformed image
    """
    ## top-left coordinates
    tlx = random.randint(0, 4)
    tly = random.randint(0, 4)

    ## bottom-right coordinates
    brx = tlx + 32
    bry = tly + 32

    padded_image = np.zeros((3, 36, 36))
    output_image = np.zeros((3, 32, 32))

    for channel in range(CHANNELS):
        padded_image[channel] = np.pad(
            input_image[channel], pad_width=2, mode="constant", constant_values=0
        )

    for channel in range(CHANNELS):
        output_image[channel] = padded_image[channel][tlx:brx, tly:bry]

    return output_image


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
    alpha = random.uniform(0.5, 2.0)

    output_image = np.zeros((3, 32, 32))

    ## perform contrast
    for channel in range(CHANNELS):
        for i in range(0, 32):
            for j in range(0, 32):
                output_image[channel][i][j] = (
                    alpha * (input_image[channel][i][j] - 128) + 128
                ) % 256

    probability = random.random()

    if probability > 0.5:
        ## perform horizontal flip
        for channel in range(CHANNELS):
            for i in range(0, 32):
                for j in range(0, 16):
                    output_image[channel][i][j], output_image[channel][i][31 - j] = (
                        input_image[channel][i][31 - j],
                        input_image[channel][i][j],
                    )

    return output_image


def main():
    """
    Entry point of script
    """
    log = enable_logging()

    filenames = []
    test_filenames = []
    labels = []
    test_labels = []
    images = []
    test_images = []

    ## 1. Loading the dataset
    log.info("## 1. Loading the dataset")

    for file in os.listdir(CIFAR_DATASET_FILENAME):
        if file.startswith("data"):
            batch_dataset = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))

            for i in range(IMAGES_PER_BATCH):
                filenames.append(batch_dataset[b"filenames"][i])
                labels.append(batch_dataset[b"labels"][i])
                images.append(np.array(batch_dataset[b"data"][i]).reshape(3, 32, 32))

        elif file.startswith("batches"):
            label_names = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))[
                b"label_names"
            ]
            label_names = [label.decode("utf-8") for label in label_names]

        elif file.startswith("test_batch"):
            test_batch = unpickle(os.path.join(CIFAR_DATASET_FILENAME, file))

            for i in range(IMAGES_PER_BATCH):
                test_filenames.append(test_batch[b"filenames"][i])
                test_labels.append(test_batch[b"labels"][i])
                test_images.append(np.array(test_batch[b"data"][i]).reshape(3, 32, 32))

    train_dataset = {
        "filenames": filenames,
        "labels": labels,
        "images": np.array(images),
    }
    test_dataset = {
        "filenames": test_filenames,
        "labels": test_labels,
        "images": np.array(test_images),
    }

    log.info(f"Size of train dataset: {len(train_dataset['filenames'])}")
    log.info(f"Size of test dataset: {len(test_dataset['filenames'])}")
    log.info(f"Labels in dataset: {label_names}")

    ## 2. Image transformations
    log.info("## 2. Image transformations")

    example_idx = random.randint(0, NUM_TRAIN_BATCHES * IMAGES_PER_BATCH - 1)
    example_image = train_dataset["images"][example_idx]

    log.info(
        f"Label of example image: {train_dataset['labels'][example_idx]} ({label_names[train_dataset['labels'][example_idx]-1]})"
    )
    log.info(
        f"Name of example image: {train_dataset['filenames'][example_idx].decode('utf-8')}"
    )
    log.info(f"Matrix shape of example image: {example_image.shape}")

    ## Naive approach
    # image_ = np.array([])
    # for j in range(HEIGHT * HEIGHT):
    #    image_ = np.append(image_, example_image[j :: HEIGHT * HEIGHT])

    # image_.resize(32, 32, 3)

    Image.fromarray(example_image.transpose(1, 2, 0).astype(np.uint8)).save(
        "example.png"
    )

    out1 = random_rotation(example_image)
    Image.fromarray(out1.transpose(1, 2, 0).astype(np.uint8)).save(
        "example_randomrotation.png"
    )

    out2 = random_cutout(example_image)
    Image.fromarray(out2.transpose(1, 2, 0).astype(np.uint8)).save(
        "example_randomcutout.png"
    )

    out3 = random_crop(example_image)
    Image.fromarray(out3.transpose(1, 2, 0).astype(np.uint8)).save(
        "example_randomcrop.png"
    )

    out4 = contrast_and_horizontal_flipping(example_image)
    Image.fromarray(out4.transpose(1, 2, 0).astype(np.uint8)).save(
        "example_contrastandhorizontalflip.png"
    )

    log.info("Example image and its transformation images have been saved as png")

    ## 3. Generating augmented training set
    log.info("## 3. Generating augmented training set")

    operations = np.random.randint(0, 4, NUM_TRAIN_BATCHES * IMAGES_PER_BATCH)
    augmented_images = []

    for i in tqdm(range(NUM_TRAIN_BATCHES * IMAGES_PER_BATCH)):
        og_image = train_dataset["images"][i]
        operation = operations[i]

        if operation == 0:
            augmented_image = random_rotation(og_image)
        elif operation == 1:
            augmented_image = random_cutout(og_image)
        elif operation == 2:
            augmented_image = random_crop(og_image)
        else:
            augmented_image = contrast_and_horizontal_flipping(og_image)

        train_dataset["filenames"].append(b"augmented_" + train_dataset["filenames"][i])
        train_dataset["labels"].append(train_dataset["labels"][i])
        augmented_images.append(augmented_image)
    
    train_dataset["images"].append(augmented_images)

    log.info(len(train_dataset["filenames"]))
    log.info(len(train_dataset["labels"]))
    log.info(len(train_dataset["images"]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
