# Report: CS776A Assignment 1 
> Submitted on : 03/02/2022, by Gurbaaz Singh Nandra (190349)

### Table of Contents

1. [Models](#1-models)
2. [Hyperparameters](#2-hyperparameters)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Instructions for running the models](#4-instructions-for-running-the-models)
5. [Derivation of gradients and update expressions](#5-derivation-of-gradients-and-update-expressions)
6. [Performance Comparison](#6-performance-comparison)

## 1. Models

- Building the Model:
- Architecture of Models:

## 2. Hyperparameters

1. **Learning rate**: Different learning rates have been tried and experimented, where a lr of `1e-2` was a bit too fast, and `1e-4` a bit too slow. Finally learning rate of `1e-3` proved to be the best tradeoff.
2. **Epochs**: It has been observed that there is no significant change in model performance after ~20 epochs, though the model that has been finally evaluated on has been trained on 50 epochs.
3. **Batch Size**: The models have been trained with a batch size of `1`, that is, SGD has been used.

## 3. Evaluation metrics

## 4. Instructions for running the models

### i. Downloading and extracting CIFAR Dataset

1. Download the dataset using `wget` or `curl` (or manual download from the internet GUI)
    ```
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    ```

2. Extract the content with `tar` using the following command
    ```
    tar zxvf cifar-10-python.tar.gz 
    ```

This will generate a directory named `cifar-10-batches-py` containing data batches.


### ii. Running the code

Make sure you have `python` and `pip`/`conda` (package managers for `python`) installed.

1. Install the required dependencies,

    ```
    pip install -r requirements.txt
    ```

2. Run the `dataset.py` script. This will run image transformations and generate augmented training set.

    ```
    python dataset.py
    ```

3. Post execution, you will see 3 pickled data batches in your current directory, named `unaugmented_dataset`, `augmented_dataset` and `test_dataset`. These contain labels and numpy image array (of size 3x32x32) data of normal training set(50000 samples), augmented training set(50000 samples) and test set(10000 samples) respectively. These will be loaded and used by our next script contaning the implementation, training and testing of our MLP model.
    ```
    python model.py
    ```
    If you want to use the provided model weights and skip the training part, pass the `--no-train` flag.
    ```
    python model.py --no-train
    ```

## 5. Derivation of gradients and update expressions

\begin{equation}
    3+3
\end{equation}

## 6. Performance Comparison
    