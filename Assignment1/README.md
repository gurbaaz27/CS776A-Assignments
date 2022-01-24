## Solution

### Dataset

1. Download the dataset using `wget` or `curl`

```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

2. Extract the content with `tar`

```
tar zxvf cifar-10-python.tar.gz 
```

###

1. Install the dependencies

```
pip install -r requirements.txt
```

2. Run the image transformations script and create augmented dataset

```
python dataset.py
```
