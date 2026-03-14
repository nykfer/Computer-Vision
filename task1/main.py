from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets
import numpy as np
from torch import Tensor
import torch 
import torch.nn as nn
from dataclasses import dataclass

#download dataset
train = datasets.MNIST(root='./data', train=True, download=True)
test = datasets.MNIST(root='./data', train=False, download=True)

def convert_data(data):
    if isinstance(data, Tensor):
        return data.numpy().reshape(-1, 784)
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid data type")

@dataclass
class MNISTPredict:
    data:list
    preds:list

class MnistClassifierInterface(ABC):
    
    @abstractmethod
    def train():
        pass
    
    @abstractmethod
    def predict():
        pass

class RandomForest(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier()
        self.mnist_predict = MNISTPredict()

    def train(self, data, target):

        data = convert_data(data)
        target = convert_data(target)

        self.model.fit(data, target)

    def test(self, data):
        data = convert_data(data)
        preds = self.model.predict(data)
        return MNISTPredict(data = data, preds=preds)

class FeedForwardNN(nn.Module):
  def __init__(self,p=0.3):
    super(FeedForwardNeuralNetwork, self).__init__()

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
    def __init__(self,  model, lr=0.001, weight_decay=0, epochs = 40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
    
    def _train_model(self, loader):
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:

                self.optimizer.zero_grad()
                output = self.model(X_batch.to(self.device))
                loss   = self.criterion(output, y_batch.to(self.device))
                loss.backward()
                self.optimizer.step()
    def _test_model(self, test):
        self.model.eval()
        with torch.no_grad():
            return self.model(test.to(self.device)).argmax(dim=1).cpu().numpy()

class FNN(NeuralNetwork):
    def __init__(self):
        self.model = FeedForwardNN()
        super(self.model).__init__()

    def train(self, data, target):
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self._train_model(loader)

    def test(self, data):
        return self._test_model(data)

class ConvolutionalNN(nn.Module):
  def __init__(self, in_channels=1, p=0.3):
    super(CNN, self).__init__()

    self.features = nn.Sequential(
        nn.Conv2d(in_channels, 15, kernel_size=3, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(15, 20, kernel_size=3, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
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
        self.model = ConvolutionalNN()
        super(self.model).__init__()

    def train(self, data, target):
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self._train_model(loader)

    def test(self, data):
        return self._test_model(data)

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str):
        if algorithm == "cnn":
            self.model = CNN()
        elif algorithm == "rf":
            self.model = RandomForest()
        elif algorithm == "nn":
            self.model = FNN()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)