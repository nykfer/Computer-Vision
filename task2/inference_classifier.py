"""
Image classification inference module for animal recognition.

Loads a fine-tuned ResNet-50 model and classifies animal images
into one of 10 categories.

Usage:
    python inference_classifier.py --model_path animals.pth --image path/to/image.jpg
"""

import argparse

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Class names corresponding to the 10-class Animals-10 dataset
# Sorted alphabetically to match torchvision ImageFolder ordering
CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]


class AnimalClassifierInference:
    """Runs inference with a fine-tuned ResNet-50 for animal classification."""

    def __init__(
        self,
        model_path: str = "animals.pth",
        device: str = None,
    ):
        """
        Args:
            model_path: Path to the saved model weights (.pth file).
            device: Device to run inference on ('cpu', 'cuda'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names =  CLASS_NAMES
        self.num_classes = len(self.class_names)

        # Build ResNet-50 architecture and replace final FC layer
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing (used during training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path: str) -> tuple[str, float, dict[str, float]]:
        """
        Classify an animal image.

        Args:
            image_path: Path to the input image.

        Returns:
            Tuple of (predicted_class, confidence, all_probabilities).
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        pred_idx = probabilities.argmax().item()
        pred_class = self.class_names[pred_idx]
        confidence = probabilities[pred_idx].item()

        all_probs = {
            name: probabilities[i].item()
            for i, name in enumerate(self.class_names)
        }

        return pred_class, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description="Animal image classification inference")
    parser.add_argument("--model_path", type=str, default="animals.pth",
                        help="Path to the trained model weights")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    classifier = AnimalClassifierInference(
        model_path=args.model_path, device=args.device
    )
    pred_class, confidence, all_probs = classifier.predict(args.image)

    print(f"Predicted class: {pred_class} (confidence: {confidence:.4f})")
    print("\nAll probabilities:")
    for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: {prob:.4f}")


if __name__ == "__main__":
    main()
