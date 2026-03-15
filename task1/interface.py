from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MNISTPredict:
    data: list
    preds: list
    accuracy: Optional[List[float]] = None

    def show(self, count: int):
        """Show `count` first samples as 28x28 images with predicted labels."""
        data_array = np.array(self.data)
        preds_array = np.array(self.preds)

        if count <= 0:
            return

        count = min(count, len(data_array))
        images = data_array[:count].reshape(count, 28, 28)
        labels = preds_array[:count]

        cols = min(count, 5)
        rows = int(np.ceil(count / cols))

        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(count):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.title(f"Label: {labels[i]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


class MnistClassifierInterface(ABC):
    """Abstract interface for MNIST classifiers."""

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

