## CS776A Assignment 1
> Submitted on : 03/02/2022, by Gurbaaz Singh Nandra (190349)

### 1. Downloading and extracting CIFAR Dataset

1. Download the dataset using `wget` or `curl` (or manual download from the internet GUI)

```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

2. Extract the content with `tar` using the following command

```
tar zxvf cifar-10-python.tar.gz 
```

This will generate a directory named `cifar-10-batches-py` containing data batches.


### 2. Running the code

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