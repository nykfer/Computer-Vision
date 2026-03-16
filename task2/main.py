"""
Animal Verification Pipeline

Takes two inputs:
  1. A text message (e.g., "There is a cow in the picture.")
  2. An image of an animal

Returns a boolean: True if the animal mentioned in the text matches the
animal detected in the image, False otherwise.

The pipeline works in two stages:
  - NER stage: Extracts the animal name from the text using a fine-tuned BERT model.
  - Classification stage: Classifies the animal in the image using a fine-tuned ResNet-50.

Usage:
    python main.py --text "There is a cow in the picture." --image path/to/image.jpg
"""

import argparse
import sys

from inference_ner import NERInference
from inference_classifier import AnimalClassifierInference, CLASS_NAMES

# Mapping to normalize common synonyms/variations to canonical class names
ANIMAL_SYNONYMS = {
    "butterflies": "butterfly",
    "cats": "cat",
    "kitten": "cat",
    "kittens": "cat",
    "kitty": "cat",
    "chickens": "chicken",
    "hen": "chicken",
    "hens": "chicken",
    "rooster": "chicken",
    "roosters": "chicken",
    "cows": "cow",
    "bull": "cow",
    "bulls": "cow",
    "cattle": "cow",
    "calf": "cow",
    "dogs": "dog",
    "puppy": "dog",
    "puppies": "dog",
    "pup": "dog",
    "elephants": "elephant",
    "horses": "horse",
    "pony": "horse",
    "ponies": "horse",
    "mare": "horse",
    "stallion": "horse",
    "sheep": "sheep",
    "lamb": "sheep",
    "lambs": "sheep",
    "ram": "sheep",
    "spiders": "spider",
    "squirrels": "squirrel",
}


def normalize_animal_name(name: str) -> str:
    """Normalize an extracted animal name to a canonical class name."""
    name = name.lower().strip()
    if name in CLASS_NAMES:
        return name
    return ANIMAL_SYNONYMS.get(name, name)


def run_pipeline(
    text: str,
    image_path: str,
    ner_model_dir: str = "ner-model",
    classifier_model_path: str = "animals.pth",
    device: str = None,
    verbose: bool = False,
) -> bool:
    """
    Run the full animal verification pipeline.

    Args:
        text: Input text mentioning an animal.
        image_path: Path to the animal image.
        ner_model_dir: Path to the NER model directory.
        classifier_model_path: Path to the image classifier weights.
        device: Device for inference ('cpu', 'cuda', or None for auto).
        verbose: If True, print intermediate results.

    Returns:
        True if the animal in the text matches the animal in the image.
    """
    # Stage 1: Extract animal name from text
    ner = NERInference(model_dir=ner_model_dir, device=device)
    extracted_animal = ner.extract_animal(text)

    if verbose:
        print(f"[NER] Extracted animal: {extracted_animal}")

    if not extracted_animal:
        if verbose:
            print("[NER] No animal found in text -> False")
        return False

    # Normalize extracted name
    normalized_animal = normalize_animal_name(extracted_animal)
    if verbose:
        print(f"[NER] Normalized: {normalized_animal}")

    # Stage 2: Classify the animal in the image
    classifier = AnimalClassifierInference(
        model_path=classifier_model_path, device=device
    )
    predicted_class, confidence, all_probs = classifier.predict(image_path)

    if verbose:
        print(f"[Classifier] Predicted: {predicted_class} ({confidence:.4f})")

    # Stage 3: Compare
    match = normalized_animal == predicted_class

    if verbose:
        print(f"[Result] Match: {match}")

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Animal verification pipeline: text + image -> boolean"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Text mentioning an animal (e.g., 'There is a cow in the picture.')"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the animal image"
    )
    parser.add_argument(
        "--ner_model_dir", type=str, default="ner-model",
        help="Path to the NER model directory"
    )
    parser.add_argument(
        "--classifier_model_path", type=str, default="animals.pth",
        help="Path to the image classifier weights"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print intermediate pipeline results"
    )
    args = parser.parse_args()

    result = run_pipeline(
        text=args.text,
        image_path=args.image,
        ner_model_dir=args.ner_model_dir,
        classifier_model_path=args.classifier_model_path,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"\nResult: {result}")
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
