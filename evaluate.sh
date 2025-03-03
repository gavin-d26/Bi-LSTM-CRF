#!/bin/bash

# Create or clear the evaluation results log file
echo "Evaluation Results" > evaluation_results.log
echo "==================" >> evaluation_results.log
echo "" >> evaluation_results.log
echo "Date: $(date)" >> evaluation_results.log
echo "" >> evaluation_results.log

# Function to evaluate a model's predictions
evaluate_model() {
    local model_name=$1
    local pred_file=$2
    local gold_file=$3
    local dataset_name=$4

    echo "Evaluating $model_name on $dataset_name dataset..." | tee -a evaluation_results.log
    
    # Run the evaluation script
    perl A2-data/evalIOB2.pl "$gold_file" "$pred_file" | tee -a evaluation_results.log
    
    echo "" >> evaluation_results.log
    echo "----------------------------------------" >> evaluation_results.log
    echo "" >> evaluation_results.log
}

# List of models to evaluate
echo "Starting evaluation of all models..." | tee -a evaluation_results.log
echo "" >> evaluation_results.log

# Evaluate BiLSTM-CRF model
evaluate_model "BiLSTM-CRF" "submission/bilstm_crf_dev.output" "A2-data/dev.answers" "dev"
evaluate_model "BiLSTM-CRF" "submission/bilstm_crf_test.output" "A2-data/test_answers/test.answers" "test"

# Evaluate BiLSTM-CRF with Character CNN
evaluate_model "BiLSTM-CRF with Character CNN" "submission/bilstm_crf_char_cnn_dev.output" "A2-data/dev.answers" "dev"
evaluate_model "BiLSTM-CRF with Character CNN" "submission/bilstm_crf_char_cnn_test.output" "A2-data/test_answers/test.answers" "test"

# Evaluate BiLSTM-CRF with Max Margin Loss
evaluate_model "BiLSTM-CRF with Max Margin Loss" "submission/bilstm_crf_alt_loss_max_margin_dev.output" "A2-data/dev.answers" "dev"
evaluate_model "BiLSTM-CRF with Max Margin Loss" "submission/bilstm_crf_alt_loss_max_margin_test.output" "A2-data/test_answers/test.answers" "test"

# Evaluate BiLSTM-CRF with Ramp Loss
evaluate_model "BiLSTM-CRF with Ramp Loss" "submission/bilstm_crf_alt_loss_ramp_loss_dev.output" "A2-data/dev.answers" "dev"
evaluate_model "BiLSTM-CRF with Ramp Loss" "submission/bilstm_crf_alt_loss_ramp_loss_test.output" "A2-data/test_answers/test.answers" "test"

echo "Evaluation complete. Results saved to evaluation_results.log" | tee -a evaluation_results.log

# Make the script executable
chmod +x evaluate.sh
