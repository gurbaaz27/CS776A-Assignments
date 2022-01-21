"""main.py: A python file implementing image transformation functions and creating augmented dataset
using the transformations."""

__author__ = "Gurbaaz Singh Nandra"


import os
import pickle


def unpickle(file):
    """
    
    """
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def main():
    """
    Entry point of script
    """
    CIFAR_DATASET_FILENAME = "cifar-10-batches-py"

    for file in os.listdir(CIFAR_DATASET_FILENAME):
        # if file.startswith("data"):
        #     print(unpickle(os.path.join(CIFAR_DATASET_FILENAME, file)))
        #     print("==="*20)
        if file.startswith("batches"):
            print(unpickle(os.path.join(CIFAR_DATASET_FILENAME, file)))



if __name__ == "__main__":
    main()
