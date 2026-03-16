# Task 2: Named Entity Recognition + Image Classification

## Overview

An ML pipeline that verifies whether the animal mentioned in a text matches the animal shown in an image. It combines two models:

1. **NER model** — a fine-tuned `dslim/bert-base-NER` (BERT) that extracts the animal name from a sentence (e.g., "There is a cow in the picture." → `cow`).
2. **Image classifier** — a fine-tuned ResNet-50 trained on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset (10 classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel).

The pipeline returns `True` if both models agree on the same animal, `False` otherwise.

## Project Structure

```
task2/
├── main.py                  # Full pipeline: text + image → boolean
├── train_classifier.py      # Train the ResNet-50 image classifier
├── inference_classifier.py  # Run inference with the trained classifier
├── train_ner.py             # Train the BERT NER model
├── inference_ner.py         # Run inference with the trained NER model
├── train.json               # NER training data (500 samples)
├── test.json                # NER test data (100 samples)
├── animals.pth              # Trained classifier weights
├── eda.ipynb                # Exploratory data analysis of Animals-10 dataset
├── demo.ipynb               # Jupyter notebook with usage examples
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Image Classifier

Download the Animals-10 dataset and train:

```bash
python train_classifier.py --data_dir /path/to/animals10/raw-img \
                           --output_path animals.pth \
                           --epochs 10 \
                           --batch_size 32 \
                           --lr 0.001
```

Key parameters:
- `--data_dir` — path to the `raw-img` folder of Animals-10
- `--output_path` — where to save model weights (default: `animals.pth`)
- `--epochs` — number of training epochs (default: 10)
- `--batch_size` — batch size (default: 32)
- `--lr` — learning rate (default: 0.001)
- `--device` — `cpu` or `cuda` (auto-detected if omitted)

### NER Model

The NER training data is provided as `train.json` and `test.json`. Each entry has `tokens`, `tags`, and `text` fields. Feel free to make your own and use it. To train:

```bash
python train_ner.py --train_json train.json \
                    --test_json test.json \
                    --output_dir ner-model \
                    --epochs 3 \
                    --batch_size 16
```

Key parameters:
- `--train_json` / `--test_json` — paths to training and test data
- `--output_dir` — where to save the model (default: `ner-model`)
- `--base_model` — pretrained model to fine-tune (default: `dslim/bert-base-NER`)
- `--epochs` — number of training epochs (default: 3)
- `--batch_size` — batch size (default: 16)
- `--lr` — learning rate (default: 2e-5)

## Inference

### Full Pipeline

```bash
python main.py --text "There is a cow in the picture." \
               --image path/to/cow.jpg \
               --verbose
```

Output: `True` if the animal in the text matches the image, `False` otherwise.

### Individual Models

NER only:
```bash
python inference_ner.py --model_dir ner-model --text "I see a horse running."
```

Classifier only:
```bash
python inference_classifier.py --model_path animals.pth --image path/to/animal.jpg
```

## How It Works

1. The **NER model** tokenizes the input text with BERT's WordPiece tokenizer, predicts `B-ANIMAL` / `O` labels per token, and extracts the single animal name (merging sub-word pieces).
2. The **image classifier** preprocesses the image (resize, center crop, ImageNet normalization) and predicts one of 10 animal classes using ResNet-50.
3. The **pipeline** normalizes the extracted animal name (handling synonyms like "puppy" → "dog", "hen" → "chicken") and compares it against the classifier's prediction.

## Dataset

- **Animals-10** from Kaggle — 26,000+ images across 10 animal classes.
- **NER data** — custom JSON dataset with sentences containing animal names, labeled with `B-ANIMAL` and `O` tags (500 train / 100 test samples).

See `cnnModel.ipynb` and `demo.ipynb` for exploratory data analysis and usage examples.
