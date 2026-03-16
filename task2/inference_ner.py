"""
NER inference module for extracting animal names from text.

Loads a fine-tuned BERT token classification model and extracts
tokens labeled as B-ANIMAL from input sentences.

Usage:
    python inference_ner.py --model_dir ner-model --text "There is a cow in the picture."
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class NERInference:
    """Runs inference with a fine-tuned BERT NER model to extract animal entities."""

    def __init__(self, model_dir: str = "ner-model", device: str = None):
        """
        Args:
            model_dir: Path to the saved NER model directory.
            device: Device to run inference on ('cpu', 'cuda'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping from model config
        self.id2label = self.model.config.id2label

    def extract_animal(self, text: str) -> str | None:
        """
        Extract the animal name from the input text.

        Args:
            text: Input sentence (e.g., "There is a cow in the picture.").

        Returns:
            Extracted animal name string, or None if no animal found.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Collect tokens predicted as B-ANIMAL and merge sub-word pieces
        animal_tokens = []
        for token, pred_id in zip(tokens, predictions):
            label = self.id2label[pred_id.item()]
            if label == "B-ANIMAL":
                if token.startswith("##"):
                    animal_tokens.append(token[2:])
                else:
                    if animal_tokens:
                        break
                    animal_tokens.append(token)

        if not animal_tokens:
            return None

        animal = "".join(animal_tokens).strip(".,!?;:'\"").lower()
        return animal or None


def main():
    parser = argparse.ArgumentParser(description="NER inference for animal extraction")
    parser.add_argument("--model_dir", type=str, default="ner-model",
                        help="Path to the trained NER model")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text to extract animals from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    ner = NERInference(model_dir=args.model_dir, device=args.device)
    animal = ner.extract_animal(args.text)

    if animal:
        print(f"Extracted animal: {animal}")
    else:
        print("No animal found in the text.")


if __name__ == "__main__":
    main()
