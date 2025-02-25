#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BiLSTMCRF
from dataloader import get_data_loader, NERDataset


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_time_per_epoch(start_time, end_time, batch_size, num_samples):
    """
    Compute time taken per epoch
    """
    time_taken = end_time - start_time
    return {
        "time_per_epoch (seconds)": time_taken,
        "time_per_sample (ms)": (time_taken * 1000) / num_samples,
        "samples_per_second": num_samples / time_taken,
        "batch_size": batch_size,
    }


def train(
    model,
    train_loader,
    dev_loader,
    optimizer,
    device,
    epochs,
    model_dir,
    loss_type="nll",
    cost_factor=1.0,
    evaluate_every=1,
):
    """
    Train the model
    Args:
        model: BiLSTMCRF model
        train_loader: Training data loader
        dev_loader: Development data loader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        model_dir: Directory to save model
        loss_type: Type of loss function (nll, max_margin, softmax_margin)
        cost_factor: Cost factor for cost-sensitive losses
        evaluate_every: Evaluate on dev set every n epochs
    Returns:
        model: Trained model
    """
    model.train()
    best_f1 = 0.0
    training_stats = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch in train_loader:
                model.zero_grad()

                # Get batch data
                word_indices = batch["word_indices"].to(device)
                char_indices = (
                    batch["char_indices"].to(device)
                    if "char_indices" in batch
                    else None
                )
                tag_indices = batch["tag_indices"].to(device)
                seq_lengths = batch["seq_lengths"]

                # Create mask
                mask = torch.zeros_like(word_indices, dtype=torch.uint8).to(device)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1

                # Forward pass
                lstm_feats = model(word_indices, char_indices, seq_lengths, mask)

                # Compute loss
                if loss_type == "nll":
                    loss = model.neg_log_likelihood(lstm_feats, tag_indices, mask)
                elif loss_type == "max_margin":
                    loss = model.max_margin_loss(
                        lstm_feats, tag_indices, mask, cost_factor
                    )
                elif loss_type == "softmax_margin":
                    loss = model.softmax_margin_loss(
                        lstm_feats, tag_indices, mask, cost_factor
                    )
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss/pbar.n:.4f}"})

        # Compute time taken
        epoch_end_time = time.time()
        time_stats = compute_time_per_epoch(
            epoch_start_time,
            epoch_end_time,
            train_loader.batch_size,
            len(train_loader.dataset),
        )

        # Evaluate on dev set
        if epoch % evaluate_every == 0:
            f1, precision, recall = evaluate(model, dev_loader, device)
            print(
                f"Epoch {epoch} - Dev F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                save_model(model, optimizer, epoch, model_dir, "best_model.pt")
                print(f"New best model saved with F1: {f1:.4f}")

        # Save latest model
        save_model(model, optimizer, epoch, model_dir, "latest_model.pt")

        # Save training stats
        training_stats.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(train_loader),
                "f1": f1 if epoch % evaluate_every == 0 else None,
                "precision": precision if epoch % evaluate_every == 0 else None,
                "recall": recall if epoch % evaluate_every == 0 else None,
                **time_stats,
            }
        )

        # Save training stats
        with open(os.path.join(model_dir, "training_stats.pkl"), "wb") as f:
            pickle.dump(training_stats, f)

    return model


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the given data
    Args:
        model: BiLSTMCRF model
        data_loader: Data loader
        device: Device to evaluate on
    Returns:
        f1: F1 score
        precision: Precision score
        recall: Recall score
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            # Get batch data
            word_indices = batch["word_indices"].to(device)
            char_indices = (
                batch["char_indices"].to(device) if "char_indices" in batch else None
            )
            tag_indices = batch["tag_indices"].to(device)
            seq_lengths = batch["seq_lengths"]

            # Create mask
            mask = torch.zeros_like(word_indices, dtype=torch.uint8).to(device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1

            # Forward pass
            lstm_feats = model(word_indices, char_indices, seq_lengths, mask)

            # Viterbi decoding
            _, best_paths = model._viterbi_decode(lstm_feats, mask)

            # Convert tag indices to tags and add to lists
            for i, (path, length) in enumerate(zip(best_paths, seq_lengths)):
                y_true.extend(tag_indices[i, :length].tolist())
                y_pred.extend(path[:length])

    # Compute metrics
    f1, precision, recall = compute_metrics(y_true, y_pred)

    model.train()
    return f1, precision, recall


def compute_metrics(y_true, y_pred):
    """
    Compute F1, precision, and recall scores
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
    Returns:
        f1: F1 score
        precision: Precision score
        recall: Recall score
    """
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0

    for true, pred in zip(y_true, y_pred):
        if pred != "O" and true == pred:
            tp += 1
        elif pred != "O" and true != pred:
            fp += 1
        elif pred == "O" and true != "O":
            fn += 1

    # Compute metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1, precision, recall


def save_model(model, optimizer, epoch, model_dir, filename):
    """
    Save model
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        model_dir: Directory to save model
        filename: Filename to save model
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Extract model arguments
    model_args = {
        "vocab_size": model.vocab_size,
        "tagset_size": model.tagset_size,
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.lstm.num_layers,  # Get num_layers from LSTM
        "dropout": model.dropout.p,  # Get dropout probability from dropout layer
        "use_char_cnn": hasattr(model, "char_cnn"),
    }

    # Add char_params if present
    if hasattr(model, "char_cnn") and model.char_cnn is not None:
        # Extract filter sizes from the convolutions
        filter_sizes = []
        for conv in model.char_cnn.convs:
            filter_sizes.append(conv.kernel_size[0])

        char_embedding_dim = model.char_cnn.char_embedding.embedding_dim

        model_args["char_params"] = {
            "char_vocab_size": model.char_cnn.char_embedding.num_embeddings,
            "char_emb_dim": char_embedding_dim,
            "char_hidden_dim": model.char_cnn.convs[
                0
            ].out_channels,  # Assuming all convs have same out_channels
            "filter_sizes": filter_sizes,
        }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_args": model_args,
        },
        os.path.join(model_dir, filename),
    )


def load_model(model, optimizer, model_path, device):
    """
    Load model
    Args:
        model: Model to load
        optimizer: Optimizer to load
        model_path: Path to load model from
        device: Device to load model on
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: Epoch when model was saved
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def main():
    parser = argparse.ArgumentParser(description="Train Bi-LSTM-CRF model for NER")

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with processed data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save model"
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bilstm_crf", "bilstm_crf_char_cnn", "bilstm_crf_alt_loss"],
        default="bilstm_crf",
        help="Type of model to train",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=100, help="Dimension of word embeddings"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Dimension of LSTM hidden state"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )

    # Character CNN parameters
    parser.add_argument(
        "--char_emb_dim", type=int, default=30, help="Dimension of character embeddings"
    )
    parser.add_argument(
        "--char_hidden_dim",
        type=int,
        default=50,
        help="Dimension of character CNN output",
    )
    parser.add_argument(
        "--filter_sizes",
        type=str,
        default="3,4,5",
        help="Comma-separated list of CNN filter sizes",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["nll", "max_margin", "softmax_margin"],
        default="nll",
        help="Type of loss function",
    )
    parser.add_argument(
        "--cost_factor",
        type=float,
        default=1.0,
        help="Cost factor for cost-sensitive losses",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=1,
        help="Evaluate on dev set every n epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load vocabs
    with open(os.path.join(args.data_dir, "vocabs.pkl"), "rb") as f:
        vocabs = pickle.load(f)

    # Get data loaders
    train_loader = get_data_loader(
        os.path.join(args.data_dir, "train.pkl"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = get_data_loader(
        os.path.join(args.data_dir, "dev.pkl"),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    vocab_size = len(vocabs["word_to_idx"])
    tagset_size = len(vocabs["tag_to_idx"])

    char_params = None
    if args.model_type in ["bilstm_crf_char_cnn"]:
        char_params = {
            "char_vocab_size": len(vocabs["char_to_idx"]),
            "char_emb_dim": args.char_emb_dim,
            "char_hidden_dim": args.char_hidden_dim,
            "filter_sizes": [int(s) for s in args.filter_sizes.split(",")],
        }

    model = BiLSTMCRF(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_char_cnn=(args.model_type in ["bilstm_crf_char_cnn"]),
        char_params=char_params,
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    model = train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_dir=args.output_dir,
        loss_type=args.loss_type if args.model_type == "bilstm_crf_alt_loss" else "nll",
        cost_factor=args.cost_factor,
        evaluate_every=args.evaluate_every,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
