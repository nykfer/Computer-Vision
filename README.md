# Project description

## Repository Structure

```
├── task1/          # Image Classification + OOP (MNIST)
├── task2/          # Named Entity Recognition + Image Classification
└── README.md
```

## Task 1: Image Classification + OOP

Three classification models for the MNIST dataset, each implemented as a separate class behind a unified `MnistClassifier` interface:

- **Random Forest** (`rf`)
- **Feed-Forward Neural Network** (`nn`)
- **Convolutional Neural Network** (`cnn`)

All models implement `MnistClassifierInterface` with `train` and `predict` methods. The `MnistClassifier` wrapper class takes an algorithm name as input and provides predictions with the same interface regardless of the selected model.

See [task1/README.md](task1/README.md) for setup and usage details.

## Task 2: Named Entity Recognition + Image Classification

An ML pipeline that takes a text message and an image as input, and returns a boolean indicating whether the animal mentioned in the text matches the animal in the image.

The pipeline consists of two models:
- **NER model** — fine-tuned BERT (`dslim/bert-base-NER`) for extracting animal names from text
- **Image classifier** — fine-tuned ResNet-50 trained on the Animals-10 dataset (10 classes)

See [task2/README.md](task2/README.md) for setup and usage details.
