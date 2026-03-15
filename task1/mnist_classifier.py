from .interface import MnistClassifierInterface, MNISTPredict
from .models_rf import RandomForest
from .models_nn import FNN, CNN


class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str):
        """Factory wrapper selecting one of the supported MNIST classifiers by name."""
        if algorithm == "cnn":
            self.model = CNN()
        elif algorithm == "rf":
            self.model = RandomForest()
        elif algorithm == "nn":
            self.model = FNN()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, X_train, y_train):
        """Train the selected model."""
        self.model.train(X_train, y_train)

    def predict(self, X, y=None) -> MNISTPredict:
        """Delegate prediction to the selected model and return MNISTPredict."""
        return self.model.predict(X, y)

