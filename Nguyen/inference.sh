#!/bin/bash
# inference.sh - Script chạy inference cho hackathon
# BTC sẽ chạy script này trong Docker container

# File test từ BTC (sẽ được mount vào container)
TEST_FILE="${TEST_FILE:-/code/private_test.json}"

# Chạy prediction
python3 predict.py --test-file "$TEST_FILE"

# Kiểm tra kết quả
if [ -f "/code/submission.csv" ] && [ -f "/code/submission_time.csv" ]; then
    echo "Inference completed successfully!"
    echo "Output files:"
    echo "  - submission.csv"
    echo "  - submission_time.csv"
else
    echo "Error: Output files not found!"
    exit 1
fi
