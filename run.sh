#!/bin/bash

# Enhanced script to run the full Bi-LSTM-CRF pipeline with all required models and loss functions
# This script will automatically run all tests required for the assignment rubric

# Configuration
OUTPUT_DIR="output"
DATA_DIR="A2-data"
BATCH_SIZE=32
EPOCHS=10
SUBSET="" # Use empty string for full dataset, or "--subset 1000" for a subset

export CUDA_VISIBLE_DEVICES=5

echo "==================================================================="
echo "RUNNING COMPLETE BI-LSTM-CRF PIPELINE FOR ASSIGNMENT REQUIREMENTS"
echo "==================================================================="

# Step 1: Run preprocessing (only needs to be done once)
echo -e "\n\n=== PREPROCESSING DATA ==="
python src/main.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --run_preprocessing

# Step 2: Run batch size benchmark with base model (only needs to be done once)
echo -e "\n\n=== RUNNING BATCH SIZE BENCHMARK ==="
python src/main.py --model_type bilstm_crf --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --run_batch_benchmark

# Step 3: Train and evaluate the base BiLSTM-CRF model
echo -e "\n\n=== TRAINING & EVALUATING BASE BiLSTM-CRF MODEL ==="
python src/main.py --model_type bilstm_crf --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --run_training --run_evaluation

# Step 4: Train and evaluate the BiLSTM-CRF model with character CNN
echo -e "\n\n=== TRAINING & EVALUATING BiLSTM-CRF WITH CHARACTER CNN ==="
python src/main.py --model_type bilstm_crf_char_cnn --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --run_training --run_evaluation

# Step 5: Train and evaluate the BiLSTM-CRF model with alternative loss functions
# First with max_margin loss
echo -e "\n\n=== TRAINING & EVALUATING BiLSTM-CRF WITH MAX MARGIN LOSS ==="
python src/main.py --model_type bilstm_crf_alt_loss --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --loss_type max_margin --cost_factor 1.0 \
  --run_training --run_evaluation

# Then with ramp_loss
echo -e "\n\n=== TRAINING & EVALUATING BiLSTM-CRF WITH RAMP LOSS ==="
python src/main.py --model_type bilstm_crf_alt_loss --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --loss_type ramp_loss --cost_factor 1.0 \
  --run_training --run_evaluation

# Step 6: Collect all outputs for submission
echo -e "\n\n=== CREATING SUBMISSION DIRECTORY ==="
mkdir -p submission
cp output/results/bilstm_crf_dev.output submission/
cp output/results/bilstm_crf_test.output submission/
cp output/results/bilstm_crf_char_cnn_dev.output submission/
cp output/results/bilstm_crf_char_cnn_test.output submission/
cp output/results/bilstm_crf_alt_loss_max_margin_dev.output submission/
cp output/results/bilstm_crf_alt_loss_max_margin_test.output submission/
cp output/results/bilstm_crf_alt_loss_ramp_loss_dev.output submission/
cp output/results/bilstm_crf_alt_loss_ramp_loss_test.output submission/
cp output/results/batch_size_benchmark.txt submission/

echo -e "\n\n=== PIPELINE COMPLETED ==="
echo "All required models have been trained and evaluated."
echo "Output files are available in the submission/ directory."
echo "These satisfy the rubric requirements for:"
echo "  1. Base BiLSTM-CRF model"
echo "  2. BiLSTM-CRF with character CNN"
echo "  3. BiLSTM-CRF with alternative loss functions (max margin and ramp loss)"
echo "  4. Batch size benchmarking"
echo "  5. Results on both dev and test sets for all models" 