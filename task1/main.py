"""Compatibility module keeping the old import path.

You can still do:

    from task1.main import MnistClassifier

but the actual implementations live in:
- interface.py        -> MNISTPredict, MnistClassifierInterface
- models_rf.py        -> RandomForest
- models_nn.py        -> FNN, CNN and helper architectures
- mnist_classifier.py -> MnistClassifier
"""

from .interface import MNISTPredict, MnistClassifierInterface
from .models_rf import RandomForest
from .models_nn import FNN, CNN
from .mnist_classifier import MnistClassifier

__all__ = [
    "MNISTPredict",
    "MnistClassifierInterface",
    "RandomForest",
    "FNN",
    "CNN",
    "MnistClassifier",
]