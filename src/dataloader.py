#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class NERDataset(Dataset):
    """
    Dataset class for NER data
    """

    def __init__(self, data_path):
        """
        Initialize the dataset
        Args:
            data_path: Path to the preprocessed data
        """
        self.data_path = data_path
        self.data_dir = os.path.dirname(data_path)
        
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_sequences(sequences, pad_idx=0):
    """
    Pad a list of sequences to the same length
    """
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return padded_seqs


def pad_char_sequences(char_sequences, pad_idx=0):
    """
    Pad character sequences (list of list of character indices)
    """
    # Find max word length and max sequence length
    max_seq_len = max(len(seq) for seq in char_sequences)
    max_word_len = max(max(len(word) for word in seq) for seq in char_sequences)

    # Pad sequences to max_seq_len and each word to max_word_len
    padded_chars = []
    for seq in char_sequences:
        # Pad each word to max_word_len
        padded_words = [word + [pad_idx] * (max_word_len - len(word)) for word in seq]
        # Pad the sequence with pad_idx words
        padded_words = padded_words + [[pad_idx] * max_word_len] * (
            max_seq_len - len(seq)
        )
        padded_chars.append(padded_words)

    return padded_chars


def collate_fn(batch, pad_idx=0):
    """
    Collate function for DataLoader
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: len(x["word_indices"]), reverse=True)

    # Extract batch data
    words = [sample["words"] for sample in batch]
    word_indices = [sample["word_indices"] for sample in batch]
    char_indices = [sample["char_indices"] for sample in batch]
    tag_indices = [sample["tag_indices"] for sample in batch]

    # Store original sequence lengths for unpadding later
    seq_lengths = [len(seq) for seq in word_indices]

    # Pad sequences
    padded_word_indices = pad_sequences(word_indices, pad_idx=pad_idx)
    padded_char_indices = pad_char_sequences(char_indices, pad_idx=pad_idx)
    padded_tag_indices = pad_sequences(tag_indices, pad_idx=pad_idx)

    # Convert to tensors
    word_indices_tensor = torch.LongTensor(padded_word_indices)
    char_indices_tensor = torch.LongTensor(padded_char_indices)
    tag_indices_tensor = torch.LongTensor(padded_tag_indices)
    seq_lengths_tensor = torch.LongTensor(seq_lengths)

    return {
        "words": words,
        "word_indices": word_indices_tensor,
        "char_indices": char_indices_tensor,
        "tag_indices": tag_indices_tensor,
        "seq_lengths": seq_lengths_tensor,
    }


def get_data_loader(data_path, batch_size, shuffle=True, num_workers=0):
    """
    Get DataLoader for the dataset
    """
    dataset = NERDataset(data_path)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x),
    )
