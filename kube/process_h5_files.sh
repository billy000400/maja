#!/bin/bash
# Script to process h5 files and add gen_mass_logits
# This should be run inside the Kubernetes pod

set -e

# Default values
MASS_CLASSES="200,300,400,500,600,700,800,900,1000"
H5_FILE=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --h5-file)
            H5_FILE="$2"
            shift 2
            ;;
        --mass-classes)
            MASS_CLASSES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$H5_FILE" ]; then
    echo "Error: --h5-file is required"
    exit 1
fi

# Check if we're in the pod
if [ ! -d "/maad-vol/SPANet" ]; then
    echo "Error: /maad-vol/SPANet not found. Are you running this in the pod?"
    exit 1
fi

cd /maad-vol/SPANet

# Copy the script to the pod if it doesn't exist
SCRIPT_PATH="/maad-vol/SPANet/add_gen_mass_logits.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Note: add_gen_mass_logits.py not found at $SCRIPT_PATH"
    echo "Please copy it there or run it from the kube directory"
fi

# Run the script
python3 "$SCRIPT_PATH" "$H5_FILE" --mass-classes "$MASS_CLASSES" ${OUTPUT_DIR:+--output "$OUTPUT_DIR"}

echo "Successfully processed $H5_FILE"
