"""
Train a BERT-based NER model for extracting animal names from text.

Fine-tunes dslim/bert-base-NER on a custom dataset with B-ANIMAL / O tags.
Expects JSON files where each entry has 'tokens', 'tags', and 'text' fields.

Usage:
    python train_ner.py --train_json train.json --test_json test.json \
                        --output_dir ner-model --epochs 3 --batch_size 16
"""

import argparse
import json

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


def load_data(train_path: str, test_path: str, val_split: float, seed: int) -> DatasetDict:
    """
    Load train/test JSON files and create a train/validation/test DatasetDict.

    Args:
        train_path: Path to training JSON file.
        test_path: Path to test JSON file.
        val_split: Fraction of training data to use for validation.
        seed: Random seed for the split.

    Returns:
        DatasetDict with train, validation, and test splits.
    """
    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    # Validate token/tag alignment
    for i, sample in enumerate(train_data):
        if len(sample["tokens"]) != len(sample["tags"]):
            raise ValueError(
                f"Mismatch at train sample {i}: "
                f"{len(sample['tokens'])} tokens vs {len(sample['tags'])} tags"
            )

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    # Split train into train + validation
    split = dataset["train"].train_test_split(test_size=val_split, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": dataset["test"],
    })


def build_label_mappings(train_path: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Build label-to-id and id-to-label mappings from the training data.

    Returns:
        Tuple of (tag_names, label2id, id2label).
    """
    with open(train_path) as f:
        train_data = json.load(f)

    all_tags = set() 
    for sample in train_data:
        for tag in sample["tags"]:
            all_tags.add(tag)

    tag_names = sorted(all_tags)
    label2id = {label: idx for idx, label in enumerate(tag_names)}
    id2label = {idx: label for idx, label in enumerate(tag_names)}
    return tag_names, label2id, id2label


def tokenize_and_align_tags(examples, tokenizer, label2id):
    """
    Tokenize word-level inputs and align tags to sub-word tokens.

    BERT's WordPiece tokenizer may split a single word into multiple sub-tokens.
    For example, the word "butterfly" might be tokenized as ["butter", "##fly"].
    Since our NER tags are assigned per word, we need to decide which sub-tokens
    get the real tag and which are ignored during loss computation.

    Strategy:
      - The first sub-token of each word keeps the original tag.
      - Subsequent sub-tokens of the same word get -100 (ignored by CrossEntropyLoss).
      - Special tokens ([CLS], [SEP]) also get -100.

    Example:
      Words:      ["There", "is",  "a", "butterfly", "here"]
      Tags:       ["O",     "O",   "O", "B-ANIMAL",  "O"   ]
      Sub-tokens: ["[CLS]", "there", "is", "a", "butter", "##fly", "here", "[SEP]"]
      Labels:     [-100,     1,       1,    1,   0,        -100,    1,      -100   ]
                   ^special  ^O       ^O    ^O   ^B-ANIMAL ^cont.   ^O      ^special
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # special tokens [CLS] [SEP]
            elif word_id != prev_word_id:
                labels.append(label2id[tags[word_id]])  # first sub-token gets the real tag
            else:
                labels.append(-100)  # subsequent sub-tokens are ignored
            prev_word_id = word_id
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


def get_compute_metrics(tag_names):
    """Return a compute_metrics function for the Trainer."""
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [tag_names[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [tag_names[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train BERT NER model for animal extraction")
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to test data JSON file")
    parser.add_argument("--output_dir", type=str, default="ner-model",
                        help="Directory to save the trained model")
    parser.add_argument("--base_model", type=str, default="dslim/bert-base-NER",
                        help="Pretrained model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of training data for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting")
    args = parser.parse_args()

    # Build label mappings
    tag_names, label2id, id2label = build_label_mappings(args.train_json)
    print(f"Labels: {tag_names}")

    # Load and split data
    dataset = load_data(args.train_json, args.test_json, args.val_split, args.seed)
    print(dataset)

    # Load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenized_data = dataset.map(
        lambda examples: tokenize_and_align_tags(examples, tokenizer, label2id),
        batched=True,
    )

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(tag_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(tag_names),
    )

    # Train
    trainer.train()

    # Evaluate on test set
    results = trainer.evaluate(tokenized_data["test"])
    print(f"\nTest results: {results}")

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
