"""
Train an animal image classifier using a fine-tuned ResNet-50.

Downloads the Animals-10 dataset from Kaggle and trains a ResNet-50 model
with pretrained ImageNet weights. Splits data into train/val/test (80/10/10).

Usage:
    python train_classifier.py --data_dir /path/to/animals10/raw-img \
                               --output_path animals.pth \
                               --epochs 10 --batch_size 32 --lr 0.001
"""

import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader

# Italian-to-English class name translation for the Animals-10 dataset
TRANSLATE = {
    "cane":       "dog",
    "cavallo":    "horse",
    "elefante":   "elephant",
    "farfalla":   "butterfly",
    "gallina":    "chicken",
    "gatto":      "cat",
    "mucca":      "cow",
    "pecora":     "sheep",
    "ragno":      "spider",
    "scoiattolo": "squirrel",
}


def get_transforms():
    """Return the image preprocessing pipeline (ImageNet normalization)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_dataset(data_dir: str, batch_size: int, num_workers: int):
    """
    Load the Animals-10 dataset and split into train/val/test loaders.

    Args:
        data_dir: Path to the raw-img folder of Animals-10.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset).
    """
    transform = get_transforms()

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Remap Italian class names to English
    dataset.classes = [TRANSLATE[c] for c in dataset.classes]

    # 80% train, 10% val, 10% test
    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    train_data, temp_data = random_split(dataset, [train_size, temp_size])

    val_size = int(0.5 * temp_size)
    test_size = temp_size - val_size
    val_data, test_data = random_split(temp_data, [val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset, train_data, val_data, test_data


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create a ResNet-50 with a replaced final layer for the given number of classes."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Run one training epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model and return (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

    return running_loss / total_samples, correct / total_samples


def evaluate_per_class(model, test_loader, class_names, device):
    """Print per-class accuracy on the test set."""
    model.eval()
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                class_correct[label.item()] += (pred == label).item()

    print("\nPer-class accuracy:")
    for idx in sorted(class_total.keys()):
        correct = class_correct[idx]
        total = class_total[idx]
        print(f"  {class_names[idx]:<15} {correct}/{total}  ({100 * correct / total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 animal classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Animals-10 raw-img folder")
    parser.add_argument("--output_path", type=str, default="animals.pth",
                        help="Where to save trained model weights")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, dataset, train_data, val_data, test_data = \
        load_dataset(args.data_dir, args.batch_size, args.num_workers)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    # Build model
    model = build_model(num_classes=len(dataset.classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    evaluate_per_class(model, test_loader, dataset.classes, device)

    # Save weights
    torch.save(model.state_dict(), args.output_path)
    print(f"\nModel saved to {args.output_path}")


if __name__ == "__main__":
    main()
