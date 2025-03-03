#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import subprocess
from datetime import datetime


def run_command(command):
    """
    Run a shell command and print output
    """
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Print output in real-time
    for line in process.stdout:
        print(line.decode().strip())

    process.wait()
    return process.returncode


def preprocess_data(args):
    """
    Preprocess the data
    """
    cmd = f"python src/preprocess.py --data_dir {args.data_dir} --output_dir {args.output_dir}/data"

    # Add subset parameter if specified
    if args.subset is not None:
        cmd += f" --subset {args.subset}"

    return run_command(cmd)


def train_model(args):
    """
    Train the model
    """
    model_dir = f"{args.output_dir}/models/{args.model_type}"

    # For alternative loss model, include the loss type in the model directory
    if args.model_type == "bilstm_crf_alt_loss" and hasattr(args, "loss_type"):
        model_dir = f"{args.output_dir}/models/{args.model_type}_{args.loss_type}"

    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cmd = (
        f"python src/train.py --data_dir {args.output_dir}/data --output_dir {model_dir} "
        f"--model_type {args.model_type} --embedding_dim {args.embedding_dim} "
        f"--hidden_dim {args.hidden_dim} --num_layers {args.num_layers} "
        f"--dropout {args.dropout} --batch_size {args.batch_size} "
        f"--lr {args.lr} --epochs {args.epochs} --seed {args.seed}"
    )

    # Add extra parameters for character CNN
    if args.model_type == "bilstm_crf_char_cnn":
        cmd += (
            f" --char_emb_dim {args.char_emb_dim} --char_hidden_dim {args.char_hidden_dim} "
            f"--filter_sizes {args.filter_sizes}"
        )

    # Add extra parameters for alternative loss
    if args.model_type == "bilstm_crf_alt_loss":
        cmd += f" --loss_type {args.loss_type} --cost_factor {args.cost_factor}"

    return run_command(cmd)


def evaluate_model(args):
    """
    Evaluate the model
    """
    model_dir = f"{args.output_dir}/models/{args.model_type}"

    # For alternative loss model, include the loss type in the model directory
    if args.model_type == "bilstm_crf_alt_loss" and hasattr(args, "loss_type"):
        model_dir = f"{args.output_dir}/models/{args.model_type}_{args.loss_type}"

    results_dir = f"{args.output_dir}/results"

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # For alternative loss model, include the loss type in the output filename
    model_name = args.model_type
    if args.model_type == "bilstm_crf_alt_loss" and hasattr(args, "loss_type"):
        model_name = f"{args.model_type}_{args.loss_type}"

    # Run evaluation on dev set
    dev_output = f"{results_dir}/{model_name}_dev.output"
    cmd = (
        f"python src/evaluate.py --model_path {model_dir}/best_model.pt "
        f"--vocabs_path {args.output_dir}/data/vocabs.pkl "
        f"--data_file {args.data_dir}/dev "
        f"--output_file {dev_output}"
    )
    run_command(cmd)

    # Run official evaluation script on dev set
    run_command(f"{args.data_dir}/evalIOB2.pl {args.data_dir}/dev.answers {dev_output}")

    # Run evaluation on test set
    test_output = f"{results_dir}/{model_name}_test.output"
    cmd = (
        f"python src/evaluate.py --model_path {model_dir}/best_model.pt "
        f"--vocabs_path {args.output_dir}/data/vocabs.pkl "
        f"--data_file {args.data_dir}/test "
        f"--output_file {test_output}"
    )
    run_command(cmd)

    # Run official evaluation script on test set if test answers are available
    if os.path.exists(f"{args.data_dir}/test.answers"):
        run_command(
            f"{args.data_dir}/evalIOB2.pl {args.data_dir}/test.answers {test_output}"
        )

    return 0


def benchmark_batch_sizes(args):
    """
    Benchmark different batch sizes
    """
    results_dir = f"{args.output_dir}/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    batch_sizes = [8, 16, 32, 64, 128]
    results = {}

    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")

        # Set new batch size
        args.batch_size = batch_size

        # Train for one epoch only
        original_epochs = args.epochs
        args.epochs = 1

        # Train
        start_time = time.time()
        train_model(args)
        end_time = time.time()

        # Restore original epochs
        args.epochs = original_epochs

        # Record time
        training_time = end_time - start_time
        results[batch_size] = training_time

        print(f"Batch size {batch_size}: {training_time:.2f} seconds")

    # Write results to file
    with open(f"{results_dir}/batch_size_benchmark.txt", "w") as f:
        f.write("Batch Size\tTime (seconds)\tSamples per second\n")
        for batch_size, training_time in results.items():
            # Calculate samples per second (training has 18,546 sentences)
            samples_per_second = 18546 / training_time
            f.write(f"{batch_size}\t{training_time:.2f}\t{samples_per_second:.2f}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the full BioNLP NER pipeline")

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, default="A2-data", help="Directory with original data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory for outputs"
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
        "--hidden_dim", type=int, default=200, help="Dimension of LSTM hidden state"
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

    # Alternative loss parameters
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["nll", "max_margin", "softmax_margin", "ramp_loss", "soft_ramp_loss"],
        default="nll",
        help="Type of loss function",
    )
    parser.add_argument(
        "--cost_factor",
        type=float,
        default=1.0,
        help="Cost factor for cost-sensitive losses",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Pipeline control
    parser.add_argument(
        "--run_preprocessing", action="store_true", help="Run preprocessing step"
    )
    parser.add_argument("--run_training", action="store_true", help="Run training step")
    parser.add_argument(
        "--run_evaluation", action="store_true", help="Run evaluation step"
    )
    parser.add_argument(
        "--run_batch_benchmark", action="store_true", help="Run batch size benchmark"
    )
    parser.add_argument("--run_all", action="store_true", help="Run all steps")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit training to a subset of examples",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Default to running all steps if none are specified
    if not (
        args.run_preprocessing
        or args.run_training
        or args.run_evaluation
        or args.run_batch_benchmark
    ):
        args.run_all = True

    # Run all steps if requested
    if args.run_all:
        args.run_preprocessing = True
        args.run_training = True
        args.run_evaluation = True
        args.run_batch_benchmark = True

    # Timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting pipeline at {timestamp}")
    print(f"Model type: {args.model_type}")

    # Run preprocessing
    if args.run_preprocessing:
        print("\n=== Running Preprocessing ===")
        preprocess_data(args)

    # Run training
    if args.run_training:
        print("\n=== Running Training ===")
        train_model(args)

    # Run evaluation
    if args.run_evaluation:
        print("\n=== Running Evaluation ===")
        evaluate_model(args)

    # Run batch size benchmark
    if args.run_batch_benchmark:
        print("\n=== Running Batch Size Benchmark ===")
        benchmark_batch_sizes(args)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
