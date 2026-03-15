## Task 1 – MNIST classification + OOP

This folder contains an object-oriented solution for the first internship task:
classification of the MNIST dataset using three different models wrapped under a
common interface.

### Implemented models

- **RandomForest**
  - Uses `sklearn.ensemble.RandomForestClassifier` with default parameters.
  - Accepts inputs as NumPy arrays, pandas `DataFrame`/`Series`, lists/tuples, or
    PyTorch tensors.
  - Internally flattens each image to a vector of size 784 (28×28) and normalizes
    pixel values to [0, 1] before training or prediction.

- **FNN (Feed-Forward Neural Network)**
  - Fully-connected network implemented with `torch.nn.Sequential` in
    `FeedForwardNN`.
  - Uses batch normalization, ReLU activations, and dropout.
  - Weights are initialized with **Kaiming (He) normal initialization**; biases
    are initialized to zero.
  - Input values are automatically normalized to [0, 1].
  - Trained with cross-entropy loss and the Adam optimizer.

- **CNN (Convolutional Neural Network)**
  - Convolutional architecture implemented in `ConvolutionalNN`.
  - Uses two convolutional blocks with ReLU + max-pooling, followed by a
    fully-connected classifier head.
  - Weights are initialized with **Kaiming (He) normal initialization**; biases
    are initialized to zero.
  - Input values are automatically normalized to [0, 1].

### Weight initialization

Both neural network models (FNN and CNN) use **Kaiming (He) normal
initialization** for their weight parameters. This method draws weights from a
normal distribution scaled by `sqrt(2 / fan_in)`, which keeps the variance of
activations stable across layers when using ReLU non-linearities. Without proper
initialization, deep networks can suffer from vanishing or exploding gradients
during the early stages of training.

Biases are initialized to zero, which is standard practice since the
asymmetry needed to break symmetry is already provided by the weight
initialization.

**Source:** Stanford CS231n — *Neural Networks Part 2: Setting up the Data and
the Loss*,
[cs231n.github.io/neural-networks-2/](https://cs231n.github.io/neural-networks-2/)
(see the *Weight Initialization* section).

### Common interface

All three models implement `MnistClassifierInterface`:

- **Methods**
  - `train(self, X, y)`: fits the model to the provided data and labels.
  - `predict(self, X, y=None)`: returns predictions for `X`. If `y` is provided,
    the implementation also computes per-class accuracy.

- **Wrapper**
  - `MnistClassifier(algorithm: str)`:
    - `algorithm="rf"` → `RandomForest`
    - `algorithm="nn"` → `FNN`
    - `algorithm="cnn"` → `CNN`
  - Provides a unified API:
    - `train(X_train, y_train)`
    - `predict(X_test, y_test=None)`

### Prediction output

- All `predict` methods return an instance of `MNISTPredict`:
  - `data`: list of flattened input samples.
  - `preds`: list of predicted labels.
  - `accuracy` (optional): list of per-class accuracies (for classes 0–9) when
    ground-truth labels are provided.
- `MNISTPredict.show(count)`:
  - Visualizes the first `count` samples as 28×28 grayscale images with their
    predicted labels as titles.

### How to use

1. **Install dependencies** (for example, in a virtual environment):

   ```bash
   pip install -r task1/requirements.txt
   ```

2. **Create train and test splits** for MNIST (for example, using
   `torchvision.datasets.MNIST`), and convert them to tensors or NumPy arrays.

3. **Train a model**:

   ```python
   from task1.mnist_classifier import MnistClassifier

   clf = MnistClassifier(algorithm="cnn")  # or "rf", "nn"
   clf.train(X_train, y_train)
   ```

4. **Run inference**:

   ```python
   preds = clf.predict(X_test, y_test)  # returns MNISTPredict
   preds.show(16)  # visualize first 16 predictions
   ```

