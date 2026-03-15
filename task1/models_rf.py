from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor

from .interface import MnistClassifierInterface, MNISTPredict


class RandomForest(MnistClassifierInterface):
    def __init__(self):
        """Random Forest classifier wrapper around sklearn.RandomForestClassifier with default parameters."""
        self.model = RandomForestClassifier()

    def _prepare_features(self, X):
        """Convert input features to a normalized 2D NumPy array.

        Accepts tensors or arrays in common MNIST shapes:
        - (N, 784)
        - (N, 28, 28)
        - (N, 1, 28, 28)
        Values are scaled to [0, 1] if they appear to be in [0, 255].
        """
        if isinstance(X, Tensor):
            X = X.detach().cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, (list, tuple)):
            X = np.array(X)
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported feature type for RandomForest: {type(X)}")

        # Flatten per sample
        X = X.reshape(X.shape[0], -1)

        # Normalize to [0, 1] if necessary
        X = X.astype("float32")
        if X.max() > 1.0:
            X /= 255.0
        return X

    def _prepare_target(self, y):
        """Convert target labels to a 1D NumPy array."""
        if isinstance(y, Tensor):
            y = y.detach().cpu().numpy()
        elif isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, (list, tuple)):
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported target type for RandomForest: {type(y)}")

        y = y.reshape(-1)
        return y

    def train(self, data, target):
        """Fit the Random Forest model on the given data and labels."""
        data = self._prepare_features(data)
        target = self._prepare_target(target)
        self.model.fit(data, target)

    def predict(self, data, target=None):
        """Return predictions and optional per-class accuracy wrapped in MNISTPredict."""
        data = self._prepare_features(data)
        preds = self.model.predict(data)

        class_acc: Optional[List[float]] = None
        if target is not None:
            target = self._prepare_target(target)
            correct = preds == target
            class_acc = []
            for c in range(10):
                mask = target == c
                if mask.sum() == 0:
                    class_acc.append(0.0)
                else:
                    class_acc.append(float((correct & mask).sum() / mask.sum()))

        return MNISTPredict(data=list(data), preds=list(preds), accuracy=class_acc)

