from typing import Optional, List

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .interface import MnistClassifierInterface, MNISTPredict


class FeedForwardNN(nn.Module):
    def __init__(self, p: float = 0.3):
        super(FeedForwardNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(100, 10),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)


class NeuralNetwork(MnistClassifierInterface):
    def __init__(self, model, lr: float = 0.001, weight_decay: float = 0.0, epochs: int = 40):
        """Base training loop and utilities for PyTorch MNIST models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs

    def _to_tensor(self, x, dtype):
        """Convert supported input types to a torch.Tensor with the given dtype.

        For image data (float dtype), this:
        - accepts shapes like (N, 784), (N, 28, 28), (N, 1, 28, 28)
        - scales values to [0, 1] if they appear to be in [0, 255].
        """
        if isinstance(x, Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            t = torch.tensor(x)
        else:
            raise TypeError(f"Unsupported data type for NeuralNetwork: {type(x)}")

        t = t.type(dtype)

        # Normalize to [0, 1] if necessary
        if dtype == torch.float32:
            if t.max() > 1.0:
                t = t / 255.0

        return t

    def _train_model(self, loader: DataLoader):
        """Run the training loop over the provided DataLoader."""
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch.to(self.device))
                loss = self.criterion(output, y_batch.to(self.device))
                loss.backward()
                self.optimizer.step()

    def _test_model(self, test: Tensor):
        """Run inference and return predicted labels as a NumPy array."""
        self.model.eval()
        with torch.no_grad():
            return self.model(test.to(self.device)).argmax(dim=1).cpu().numpy()


class FNN(NeuralNetwork):
    def __init__(self):
        model = FeedForwardNN()
        super().__init__(model)

    def train(self, data, target):
        """Train the feed-forward neural network on the provided data."""
        data_t = self._to_tensor(data, torch.float32)
        target_t = self._to_tensor(target, torch.long)
        dataset = TensorDataset(data_t, target_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self._train_model(loader)

    def predict(self, data, target=None):
        """Return predictions and optional per-class accuracy for the FNN model."""
        data_t = self._to_tensor(data, torch.float32)
        preds = self._test_model(data_t)

        class_acc: Optional[List[float]] = None
        if target is not None:
            target_t = self._to_tensor(target, torch.long)
            y_true = target_t.cpu().numpy()
            correct = preds == y_true
            class_acc = []
            for c in range(10):
                mask = y_true == c
                if mask.sum() == 0:
                    class_acc.append(0.0)
                else:
                    class_acc.append(float((correct & mask).sum() / mask.sum()))

        flat_data = data_t.cpu().numpy().reshape(len(data_t), -1)
        return MNISTPredict(data=list(flat_data), preds=list(preds), accuracy=class_acc)


class ConvolutionalNN(nn.Module):
    def __init__(self, in_channels: int = 1, p: float = 0.3):
        super(ConvolutionalNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 15, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(15, 20, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=p),
            nn.Linear(180, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(50, 10),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.features(x)
        return self.classifier(x)


class CNN(NeuralNetwork):
    def __init__(self):
        model = ConvolutionalNN()
        super().__init__(model)

    def train(self, data, target):
        """Train the convolutional neural network on the provided data."""
        data_t = self._to_tensor(data, torch.float32)
        target_t = self._to_tensor(target, torch.long)
        dataset = TensorDataset(data_t, target_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self._train_model(loader)

    def predict(self, data, target=None):
        """Return predictions and optional per-class accuracy for the CNN model."""
        data_t = self._to_tensor(data, torch.float32)
        preds = self._test_model(data_t)

        class_acc: Optional[List[float]] = None
        if target is not None:
            target_t = self._to_tensor(target, torch.long)
            y_true = target_t.cpu().numpy()
            correct = preds == y_true
            class_acc = []
            for c in range(10):
                mask = y_true == c
                if mask.sum() == 0:
                    class_acc.append(0.0)
                else:
                    class_acc.append(float((correct & mask).sum() / mask.sum()))

        flat_data = data_t.cpu().numpy().reshape(len(data_t), -1)
        return MNISTPredict(data=list(flat_data), preds=list(preds), accuracy=class_acc)

