#!/bin/bash

set -e

# Configuration
INPUT_DIR="data/1_validated-audio"
OUTPUT_FILE="data/dataset_consolidated.h5"
METADATA_FILE="data/data-file/validated.tsv"

# Spectrogram parameters
N_FFT=512
WIN_LENGTH=20
HOP_LENGTH=10
N_MELS=80
F_MIN=50
F_MAX=7600
SEGMENT_DURATION=0.1
OVERLAP=0.5

# Step 1: Create consolidated dataset with Log-Mel spectrograms
python -m src.preprocessing.create_consolidated_dataset \
    --input-dir "$INPUT_DIR" \
    --output-file "$OUTPUT_FILE" \
    --metadata-file "$METADATA_FILE" \
    --n-fft $N_FFT \
    --win-length $WIN_LENGTH \
    --hop-length $HOP_LENGTH \
    --n-mels $N_MELS \
    --f-min $F_MIN \
    --f-max $F_MAX \
    --segment-duration $SEGMENT_DURATION \
    --overlap $OVERLAP

# Step 2: Train model
python main.py --use-consolidated \
               --dataset-path data/dataset_consolidated.h5 \
               --batch-size 128 \
               --num-workers 8  \
               --epochs 100 \
               --lr 1e-4 \
               --checkpoint-freq 10 \
               --experiment-name "spectrogram-dreamer-v1" \
               --h-state-size 200 \
               --z-state-size 30 \
               --action-size 128

# Step 3: Test inference with best model
if [ -d "checkpoints" ]; then
    LATEST_CHECKPOINT=$(find checkpoints -name "best_model.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    if [ -n "$LATEST_CHECKPOINT" ]; then
        TEST_AUDIO=$(find "$INPUT_DIR" -type f \( -name "*.mp3" -o -name "*.wav" \) | head -1)
        if [ -n "$TEST_AUDIO" ]; then
            python infer.py \
                --model "$LATEST_CHECKPOINT" \
                --input "$TEST_AUDIO" \
                --mode recon \
                --use_log
        fi
    fi
fi
