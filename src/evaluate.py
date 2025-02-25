#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import torch
from tqdm import tqdm
import sys

from model import BiLSTMCRF
from preprocess import read_data


def load_model(model_path, device, vocabs_path=None):
    """
    Load trained model
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        vocabs_path: Path to the vocabs.pkl file (optional)
    Returns:
        model: Loaded model
        model_args: Model arguments
    """
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # For backwards compatibility with models saved without model_args
    if "model_args" not in checkpoint:
        print(
            "Warning: model_args not found in checkpoint. Using vocabs to determine model size."
        )

        # Load vocabs to get vocabulary sizes
        if vocabs_path is not None:
            try:
                with open(vocabs_path, "rb") as f:
                    vocabs = pickle.load(f)
                vocab_size = len(vocabs["word_to_idx"])
                tagset_size = len(vocabs["tag_to_idx"])
                char_vocab_size = len(vocabs["char_to_idx"])
                print(
                    f"Loaded vocabs - Words: {vocab_size}, Tags: {tagset_size}, Chars: {char_vocab_size}"
                )
            except Exception as e:
                print(f"Error loading vocabs: {e}")
                vocab_size = 10000
                tagset_size = 20
                char_vocab_size = 100
        else:
            vocab_size = 10000
            tagset_size = 20
            char_vocab_size = 100

        # Default model arguments
        model_args = {
            "vocab_size": vocab_size,
            "tagset_size": tagset_size,
            "embedding_dim": 100,
            "hidden_dim": 200,
            "num_layers": 1,
            "dropout": 0.5,
            "use_char_cnn": False,
            "char_params": None,
        }
    else:
        model_args = checkpoint["model_args"]
        print(f"Loaded model_args from checkpoint:")
        for key, value in model_args.items():
            if key != "char_params":  # Don't print the full char_params
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value is not None}")

    # Create model
    print("Creating model with the following parameters:")
    print(f"  vocab_size: {model_args['vocab_size']}")
    print(f"  tagset_size: {model_args['tagset_size']}")
    print(f"  embedding_dim: {model_args['embedding_dim']}")
    print(f"  hidden_dim: {model_args['hidden_dim']}")
    print(f"  num_layers: {model_args.get('num_layers', 1)}")
    print(f"  dropout: {model_args.get('dropout', 0.5)}")
    print(f"  use_char_cnn: {model_args.get('use_char_cnn', False)}")

    try:
        model = BiLSTMCRF(
            vocab_size=model_args["vocab_size"],
            tagset_size=model_args["tagset_size"],
            embedding_dim=model_args["embedding_dim"],
            hidden_dim=model_args["hidden_dim"],
            num_layers=model_args.get("num_layers", 1),
            dropout=model_args.get("dropout", 0.5),
            use_char_cnn=model_args.get("use_char_cnn", False),
            char_params=model_args.get("char_params", None),
        ).to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

    # Load model weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Check if keys match
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint["model_state_dict"].keys())
        missing_keys = model_keys - checkpoint_keys
        extra_keys = checkpoint_keys - model_keys

        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if extra_keys:
            print(f"Extra keys in checkpoint: {extra_keys}")

        print("Warning: Using model with random weights")

    model.eval()
    return model, model_args


def convert_to_indices(sentences, vocabs, char_level=False):
    """
    Convert sentences to indices
    Args:
        sentences: List of sentences (list of (word, tag) tuples)
        vocabs: Vocabulary dictionaries
        char_level: Whether to include character-level indices
    Returns:
        data: List of dictionaries with word and tag indices
    """
    word_to_idx = vocabs["word_to_idx"]
    tag_to_idx = vocabs["tag_to_idx"]
    char_to_idx = vocabs["char_to_idx"] if char_level else None

    data = []

    for sentence in sentences:
        words, tags = zip(*sentence)
        word_indices = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in words]

        # Process characters if needed
        char_indices = None
        if char_level:
            char_indices = []
            for word in words:
                word_chars = [char_to_idx.get(c, char_to_idx["<unk>"]) for c in word]
                char_indices.append(word_chars)

        tag_indices = [tag_to_idx.get(tag) for tag in tags]

        data.append(
            {
                "words": words,
                "word_indices": word_indices,
                "char_indices": char_indices,
                "tag_indices": tag_indices,
            }
        )

    return data


def predict(model, sentences, vocabs, device):
    """
    Predict tags for sentences
    Args:
        model: Trained model
        sentences: List of sentences (each sentence is a list of (word, tag) tuples)
        vocabs: Vocabulary dictionaries
        device: Device to use for prediction
    Returns:
        predictions: List of predicted tags for each sentence
    """
    model.eval()
    predictions = []

    word_to_idx = vocabs["word_to_idx"]
    char_to_idx = vocabs["char_to_idx"]
    idx_to_tag = vocabs["idx_to_tag"]

    use_char_cnn = model.use_char_cnn

    print(f"Predicting with model (use_char_cnn={use_char_cnn})")

    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Predicting"):
            # Extract words from (word, tag) tuples
            words = [word for word, _ in sentence]

            # Convert words to indices
            word_indices = [
                word_to_idx.get(word, word_to_idx["<unk>"]) for word in words
            ]
            word_indices_tensor = torch.tensor(
                [word_indices], dtype=torch.long, device=device
            )

            # Prepare character indices if needed
            char_indices_tensor = None
            if use_char_cnn:
                char_indices = []
                for word in words:
                    # Ensure word has at least 3 characters (for filter sizes up to 3)
                    # by padding if necessary
                    word_chars = [
                        char_to_idx.get(c, char_to_idx["<unk>"]) for c in word
                    ]

                    # If very short word, add padding to match minimum filter size
                    # Adding explicit padding for short words
                    min_len = 3  # Minimum padding length to handle filters of size 3
                    if len(word_chars) < min_len:
                        word_chars = word_chars + [0] * (min_len - len(word_chars))

                    char_indices.append(word_chars)

                # Padding character indices
                max_word_len = (
                    max(len(chars) for chars in char_indices)
                    if char_indices
                    else min_len
                )
                padded_chars = [
                    chars + [0] * (max_word_len - len(chars)) for chars in char_indices
                ]
                char_indices_tensor = torch.tensor(
                    [padded_chars], dtype=torch.long, device=device
                )

            # Create mask
            mask = torch.ones_like(word_indices_tensor, dtype=torch.bool, device=device)

            # Get predictions
            try:
                lstm_feats = model(
                    word_indices_tensor,
                    char_indices_tensor,
                    seq_lengths=torch.tensor([len(word_indices)], device=device),
                    mask=mask,
                )

                # Viterbi decoding
                path = model._viterbi_decode_batched(lstm_feats, mask)[0]

                # Convert indices to tags
                tags = [idx_to_tag[idx] for idx in path]

                predictions.append(tags)
            except Exception as e:
                print(f"Error predicting tags for sentence: {e}")
                print(f"Word indices shape: {word_indices_tensor.shape}")
                if char_indices_tensor is not None:
                    print(f"Char indices shape: {char_indices_tensor.shape}")
                # Use a blank prediction as fallback
                predictions.append(["O"] * len(words))

    return predictions


def write_output(sentences, predictions, output_file):
    """
    Write predictions to output file
    Args:
        sentences: List of sentences (list of (word, tag) tuples)
        predictions: List of predicted tags for each sentence
        output_file: Path to output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (sentence, tags) in enumerate(zip(sentences, predictions)):
            for j, ((word, _), tag) in enumerate(zip(sentence, tags)):
                f.write(f"{word}\t{tag}\n")
            f.write("\n")

    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)} sentences")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM-CRF model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--vocabs_path", type=str, required=True, help="Path to the vocabs pickle file"
    )
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to the test data file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save predictions"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with vocabs path
    model, model_args = load_model(args.model_path, device, args.vocabs_path)

    # Load vocabs for test data processing
    print(f"Loading vocabs from {args.vocabs_path}")
    try:
        with open(args.vocabs_path, "rb") as f:
            vocabs = pickle.load(f)
        print(f"Vocabulary sizes:")
        print(f"  Word vocabulary: {len(vocabs['word_to_idx'])} words")
        print(f"  Tag vocabulary: {len(vocabs['tag_to_idx'])} tags")
        print(f"  Char vocabulary: {len(vocabs['char_to_idx'])} chars")
    except Exception as e:
        print(f"Error loading vocabs: {e}")
        sys.exit(1)

    # Read test data
    print(f"Reading data from {args.data_file}")
    test_sentences = read_data(args.data_file)
    print(f"Read {len(test_sentences)} sentences from {args.data_file}")

    # Make predictions
    predictions = predict(model, test_sentences, vocabs, device)

    # Write output
    write_output(test_sentences, predictions, args.output_file)


if __name__ == "__main__":
    main()
