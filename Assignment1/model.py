"""model.py: MLP model implementation"""

__author__ = "Gurbaaz Singh Nandra"


import argparse
import sys
import cv2
import logging
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models import resnet18


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(
            get_activation(outputs, output_layer_name)
        )


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=True):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [
            self.outputs[output_layer_name]
            for output_layer_name in self.output_layer_names
        ]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


class BBResNet18(object):
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        self.model = ModelWrapper(self.model, ["avgpool"], True)

        self.model.eval()
        self.model.to(self.device)

    def feature_extraction(self, x: np.ndarray):
        """
        param:
            x: numpy ndarray of shape: [None, 3, 224, 224] and dtype: np.float32

        return:
            numpy ndarray (feature vector) of shape: [None, 512] and dtype: np.float32
        """

        x = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            out = self.model(x).cpu().detach()
            out = out.view(out.size(0), -1)
            out = out.numpy()

        return out


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


def resize_image(image):
    return cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


class MLP:
    def __init__(
        self,
        batch_size=50000,
        lr=0.001,
        train_size=50000,
        input_nodes=512,
        hidden_nodes=64,
        output_labels=10,
    ):
        """
        Dimensions of different layers of the model
        """
        self.lr = lr
        self.batch_size = batch_size
        self.size = train_size
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_labels = output_labels

        """
        Initialising weights and biases
        """
        self.wh = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.01
        self.bh = np.zeros((self.hidden_nodes, 1))

        self.wo = np.random.randn(self.hidden_nodes, self.output_labels) * 0.01
        self.bo = np.zeros((self.output_labels, 1))

    def forward(self, x):
        """
        Forward pass
        """
        self.zh = np.dot(x, self.wh) + self.bh
        self.ah = relu(self.zh)
        self.zo = np.dot(self.ah, self.wo) + self.bo
        self.ao = softmax(self.zo)

    def backprop(self, x, y):
        """
        Backward pass
        """
        self.dcost_dzo = self.ao - y
        self.dzo_dwo = self.ah

        self.dcost_wo = (1 / self.batch_size) * np.dot(self.dzo_dwo.T, self.dcost_dzo)
        self.dcost_bo = (1 / self.batch_size) * self.dcost_dzo

        self.dzo_dah = self.wo
        self.dcost_dah = np.dot(self.dcost_dzo, self.dzo_dah.T)
        self.dah_dzh = relu_backward(self.zh)
        self.dzh_dwh = x

        self.dcost_wh = (1 / self.batch_size) * np.dot(
            self.dzh_dwh.T, self.dah_dzh * self.dcost_dah
        )
        self.dcost_bh = (1 / self.batch_size) * np.multiply(
            self.dcost_dah, self.dah_dzh
        )

    def train(self, X, Y, epochs):
        """
        Complete training pipeline of forward pass,
        backpropagation and updatation of weights and biases
        """
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                self.forward(x.reshape(1, self.input_nodes))
                self.backprop(
                    x.reshape(1, self.input_nodes), y.reshape(1, self.output_labels)
                )

                self.wh -= self.lr * self.dcost_wh
                self.bh -= self.lr * self.dcost_bh.sum(axis=0)

                self.wo -= self.lr * self.dcost_wo
                self.bo -= self.lr * self.dcost_bo.sum(axis=0)

            loss = -np.sum(np.multiply(np.log(self.ao), Y)) / self.size
            print(f"Epoch {epoch} || Loss := {loss}")

    def predict(self, x):
        """
        Predictions are computed using only the forward pass
        """
        self.forward(x)
        return np.round(self.ao).astype(np.int)


def main():
    """
    Entry point of script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Use the provided model weights and skip the training part",
    )
    args = parser.parse_args()

    log = enable_logging()

    unaugmented_dataset = unpickle("unaugmented_dataset")
    augmented_dataset = unpickle("augmented_dataset")
    test_dataset = unpickle("test_dataset")

    ## 4. Feature extraction
    log.info("## 4. Feature extraction")
    resnet_model = BBResNet18()

    unaugmented_trainset = unaugmented_dataset["images"].astype("float32")
    unaugmented_trainset = unaugmented_trainset.transpose(0, 2, 3, 1)

    augmented_trainset = augmented_dataset["images"].astype("float32")
    augmented_trainset = augmented_trainset.transpose(0, 2, 3, 1)

    testset = test_dataset["images"].astype("float32")
    testset = testset.transpose(0, 2, 3, 1)

    X_u = []
    for i in tqdm(unaugmented_trainset):
        feature = resnet_model.feature_extraction(
            resize_image(i).transpose(2, 0, 1).reshape(1, 3, 224, 224)
        )
        X_u.append(feature[0])

    X_u = np.array(X_u)

    unaugmented_labels = np.array(unaugmented_dataset["labels"]).reshape(
        50000,
    )
    Y_u = np.zeros((unaugmented_labels.size, unaugmented_labels.max() + 1))
    Y_u[np.arange(unaugmented_labels.size), unaugmented_labels] = 1

    X_a = []
    for i in tqdm(augmented_trainset):
        feature = resnet_model.feature_extraction(
            resize_image(i).transpose(2, 0, 1).reshape(1, 3, 224, 224)
        )
        X_a.append(feature[0])

    X_a = np.array(X_a)

    augmented_labels = np.array(augmented_dataset["labels"]).reshape(
        50000,
    )
    Y_a = np.zeros((augmented_labels.size, augmented_labels.max() + 1))
    Y_a[np.arange(augmented_labels.size), augmented_labels] = 1

    test_X = []
    for i in tqdm(testset):
        feature = resnet_model.feature_extraction(
            resize_image(i).transpose(2, 0, 1).reshape(1, 3, 224, 224)
        )
        test_X.append(feature[0])

    test_X = np.array(test_X)

    test_labels = np.array(test_dataset["labels"]).reshape(
        10000,
    )
    test_Y = np.zeros((test_labels.size, test_labels.max() + 1))
    test_Y[np.arange(test_labels.size), test_labels] = 1

    ## 5. MLP implementation

    # Implemented MLP as a class in python above main() function

    ## 6. Back-propagation

    # Implemented backprop() function in the MLP class

    ## 7. Performance evaluation

    if not args.no_train:
        log.info("Training on unaugmented trainset")
        mlp_u = MLP(1, 0.001, 50000, 512, 64, 10)
        mlp_u.train(X_u, Y_u, 100)

        mlp_predictions_u = mlp_u.predict(X_u)

        print(
            "MLP model accuracy(trained on unaugmented trainset) on training data: ",
            np.sum(mlp_predictions_u == unaugmented_labels)
            / unaugmented_labels.shape[0],
        )

        log.info("Saving model weights in mlp_unaugmented.npy file")
        with open("mlp_unaugmented.npy", "wb") as f:
            np.save(f, mlp_u.wh)
            np.save(f, mlp_u.bh)
            np.save(f, mlp_u.wo)
            np.save(f, mlp_u.bo)

        log.info("Training on augmented trainset")
        mlp_a = MLP(1, 0.001, 100000, 512, 64, 10)
        mlp_a.train(np.concatenate([X_u, X_a]), np.concatenate([Y_u, Y_a]), 100)

        mlp_predictions_a = mlp_u.predict(np.concatenate([X_u, X_a]))

        print(
            "MLP model accuracy(trained on augmented trainset) on training data: ",
            np.sum(mlp_predictions_a == np.concatenate([Y_u, Y_a]))
            / (2 * unaugmented_labels.shape[0]),
        )

        log.info("Saving model weights in mlp_augmented.npy file")
        with open("mlp_augmented.npy", "wb") as f:
            np.save(f, mlp_a.wh)
            np.save(f, mlp_a.bh)
            np.save(f, mlp_a.wo)
            np.save(f, mlp_a.bo)

    log.info("Loading weights of MLP (trained on unaugmented trainset)")
    mlp_u = MLP(1, 0.001, 50000, 512, 64, 10)
    with open("mlp_unaugmented.npy", "rb") as f:
        mlp_u.wh = np.load(f)
        mlp_u.bh = np.load(f)
        mlp_u.wo = np.load(f)
        mlp_u.bo = np.load(f)
    
    log.info("Calculating predictions of MLP (trained on unaugmented trainset)")
    mlp_predictions_u = mlp_u.predict(test_X)

    print(
            "MLP model accuracy(trained on unaugmented trainset) on test data: ",
            np.sum(mlp_predictions_u == test_labels)
            / test_labels.shape[0],
        )

    log.info("Loading weights of MLP (trained on augmented trainset)")
    mlp_a = MLP(1, 0.001, 100000, 512, 64, 10)
    with open("mlp_augmented.npy", "rb") as f:
        mlp_a.wh = np.load(f)
        mlp_a.bh = np.load(f)
        mlp_a.wo = np.load(f)
        mlp_a.bo = np.load(f)
    
    log.info("Calculating predictions of MLP (trained on augmented trainset)")
    mlp_predictions_a = mlp_a.predict(test_X)

    print(
            "MLP model accuracy(trained on augmented trainset) on test data: ",
            np.sum(mlp_predictions_a == test_labels)
            / test_labels.shape[0],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
