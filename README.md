# Bi-LSTM-CRF for Biomedical Named Entity Recognition

This project implements a Bidirectional LSTM-CRF model for biomedical named entity recognition on the BioNLP/NLPBA 2004 corpus. The task involves identifying 5 types of entities: DNA, RNA, protein, cell line, and cell type.

## Project Structure
```
.
├── A2-data/              # Original dataset
├── src/                  # Source code
├── models/               # Saved model checkpoints
├── data/                 # Processed data
├── results/              # Output results
└── cursor_management/    # Project management files
```

## Requirements

```
pip install -r requirements.txt
```

## Running the Code

### 1. Data Preprocessing

```
python src/preprocess.py --data_dir A2-data --output_dir data
```

### 2. Training

#### Basic Bi-LSTM-CRF

```
python src/train.py --model_type bilstm_crf --data_dir data --output_dir models/bilstm_crf
```

#### Bi-LSTM-CRF with Character CNN

```
python src/train.py --model_type bilstm_crf_char_cnn --data_dir data --output_dir models/bilstm_crf_char_cnn
```

#### Bi-LSTM-CRF with Alternative Loss Function

```
python src/train.py --model_type bilstm_crf_alt_loss --data_dir data --output_dir models/bilstm_crf_alt_loss --loss_type [max_margin|softmax_margin|ramp|soft_ramp]
```

### 3. Evaluation

```
python src/evaluate.py --model_path models/[MODEL_NAME]/best_model.pt --data_file A2-data/dev --output_file results/dev.output
```

### 4. Run Official Evaluation Script

```
A2-data/evalIOB2.pl A2-data/dev.answers results/dev.output
```

## Model Variants

1. **Bi-LSTM-CRF**: Basic model with bidirectional LSTM and CRF layer
2. **Bi-LSTM-CRF with Character CNN**: Adds character-level CNN embeddings
3. **Bi-LSTM-CRF with Alternative Loss**: Uses cost-sensitive loss functions