#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
from collections import Counter, defaultdict


def read_data(file_path):
    """
    Read the data file and return a list of sentences.
    Each sentence is a list of (word, tag) tuples.
    """
    sentences = []
    sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            if "\t" in line:
                word, tag = line.split("\t")
            else:
                # In test data, there might be no tags
                word = line
                tag = "O"

            sentence.append((word, tag))

        if sentence:  # Don't forget the last sentence
            sentences.append(sentence)

    return sentences


def build_vocab(sentences, min_freq=1):
    """
    Build word and tag vocabularies.
    """
    word_counter = Counter()
    char_counter = Counter()
    tag_counter = Counter()

    for sentence in sentences:
        for word, tag in sentence:
            word_counter[word] += 1
            tag_counter[tag] += 1
            for char in word:
                char_counter[char] += 1

    # Create word vocabulary with special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update(
        {
            w: i + 2
            for i, (w, _) in enumerate(word_counter.items())
            if word_counter[w] >= min_freq
        }
    )

    # Create character vocabulary
    char_vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    char_vocab.update({c: i + 4 for i, (c, _) in enumerate(char_counter.items())})

    # Create tag vocabulary
    tag_vocab = {t: i for i, (t, _) in enumerate(tag_counter.items())}

    # Create reverse mappings
    idx_to_word = {i: w for w, i in vocab.items()}
    idx_to_tag = {i: t for t, i in tag_vocab.items()}
    idx_to_char = {i: c for c, i in char_vocab.items()}

    return {
        "word_to_idx": vocab,
        "idx_to_word": idx_to_word,
        "tag_to_idx": tag_vocab,
        "idx_to_tag": idx_to_tag,
        "char_to_idx": char_vocab,
        "idx_to_char": idx_to_char,
    }


def preprocess_data(sentences, vocabs):
    """
    Convert sentences to sequences of indices.
    """
    word_to_idx = vocabs["word_to_idx"]
    tag_to_idx = vocabs["tag_to_idx"]
    char_to_idx = vocabs["char_to_idx"]

    processed_data = []

    for sentence in sentences:
        words, tags = zip(*sentence)
        word_indices = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in words]

        # Process characters
        char_indices = []
        for word in words:
            word_chars = [char_to_idx.get(c, char_to_idx["<unk>"]) for c in word]
            char_indices.append(word_chars)

        tag_indices = [tag_to_idx.get(tag) for tag in tags]

        processed_data.append(
            {
                "words": words,
                "word_indices": word_indices,
                "char_indices": char_indices,
                "tag_indices": tag_indices,
            }
        )

    return processed_data


def save_data(data, output_path):
    """
    Save processed data to a pickle file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess BioNLP NER data")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with the BioNLP data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed data",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Minimum word frequency to include in vocab",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit training to a subset of examples",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read data
    train_sentences = read_data(os.path.join(args.data_dir, "train"))
    dev_sentences = read_data(os.path.join(args.data_dir, "dev"))
    test_sentences = read_data(os.path.join(args.data_dir, "test"))

    # Limit to subset if specified
    if args.subset is not None and args.subset > 0:
        print(f"Using subset of {args.subset} training examples")
        train_sentences = train_sentences[: args.subset]

    print(
        f"Number of sentences - Train: {len(train_sentences)}, Dev: {len(dev_sentences)}, Test: {len(test_sentences)}"
    )

    # Build vocabularies
    vocabs = build_vocab(train_sentences, min_freq=args.min_freq)

    print(
        f"Vocabulary sizes - Words: {len(vocabs['word_to_idx'])}, Tags: {len(vocabs['tag_to_idx'])}, Chars: {len(vocabs['char_to_idx'])}"
    )

    # Preprocess data
    train_data = preprocess_data(train_sentences, vocabs)
    dev_data = preprocess_data(dev_sentences, vocabs)
    test_data = preprocess_data(test_sentences, vocabs)

    # Save data
    save_data(train_data, os.path.join(args.output_dir, "train.pkl"))
    save_data(dev_data, os.path.join(args.output_dir, "dev.pkl"))
    save_data(test_data, os.path.join(args.output_dir, "test.pkl"))
    save_data(vocabs, os.path.join(args.output_dir, "vocabs.pkl"))

    print("Preprocessing completed and saved to", args.output_dir)


if __name__ == "__main__":
    main()
