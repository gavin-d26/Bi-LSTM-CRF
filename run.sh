#!/bin/bash

# Simple shell script to run the Bi-LSTM-CRF pipeline
# Usage: ./run.sh [model_type] [option]
# model_type: bilstm_crf, bilstm_crf_char_cnn, bilstm_crf_alt_loss
# option: full, preprocess, train, evaluate, benchmark

# Default values
MODEL_TYPE="bilstm_crf"
OPTION="full"
OUTPUT_DIR="output"
DATA_DIR="A2-data"
BATCH_SIZE=32
EPOCHS=10
SUBSET="" # Use empty string for full dataset, or "--subset 1000" for a subset

# Parse arguments
if [ $# -ge 1 ]; then
    MODEL_TYPE=$1
fi

if [ $# -ge 2 ]; then
    OPTION=$2
fi

# Validate model type
if [ "$MODEL_TYPE" != "bilstm_crf" ] && [ "$MODEL_TYPE" != "bilstm_crf_char_cnn" ] && [ "$MODEL_TYPE" != "bilstm_crf_alt_loss" ]; then
    echo "Invalid model type. Choose from: bilstm_crf, bilstm_crf_char_cnn, bilstm_crf_alt_loss"
    exit 1
fi

# Create command based on option
CMD="python src/main.py --model_type $MODEL_TYPE --batch_size $BATCH_SIZE --epochs $EPOCHS --data_dir $DATA_DIR --output_dir $OUTPUT_DIR"

case $OPTION in
    "full")
        echo "Running full pipeline with model: $MODEL_TYPE"
        CMD="$CMD --run_all"
        ;;
    "preprocess")
        echo "Running preprocessing step only"
        CMD="$CMD --run_preprocessing"
        ;;
    "train")
        echo "Running training step with model: $MODEL_TYPE"
        CMD="$CMD --run_training"
        ;;
    "evaluate")
        echo "Running evaluation step with model: $MODEL_TYPE"
        CMD="$CMD --run_evaluation"
        ;;
    "benchmark")
        echo "Running batch size benchmark with model: $MODEL_TYPE"
        CMD="$CMD --run_batch_benchmark"
        ;;
    *)
        echo "Invalid option. Choose from: full, preprocess, train, evaluate, benchmark"
        exit 1
        ;;
esac

# Add extra parameters for specific model types
if [ "$MODEL_TYPE" = "bilstm_crf_alt_loss" ]; then
    # Default to max_margin loss for alternative loss model
    CMD="$CMD --loss_type max_margin --cost_factor 1.0"
fi

echo "Executing: $CMD"
eval $CMD 