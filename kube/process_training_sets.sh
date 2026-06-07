#!/bin/bash
# Script to copy add_gen_mass_logits.py to pod and process h5 files in /maad-vol/data/training_sets

set -e

POD_NAME="maja-pod-billy"
SCRIPT_NAME="add_gen_mass_logits.py"
TRAINING_SETS_DIR="/maad-vol/data/training_sets"
MASS_CLASSES="200,300,400,500,600,700,800,900,1000"

echo "Step 1: Checking if pod $POD_NAME exists..."
if ! kubectl get pod "$POD_NAME" &>/dev/null; then
    echo "Error: Pod $POD_NAME not found. Please create it first with:"
    echo "  kubectl create -f pod.yaml"
    exit 1
fi

echo "Step 2: Checking if pod is ready..."
POD_STATUS=$(kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}')
if [ "$POD_STATUS" != "Running" ]; then
    echo "Warning: Pod status is $POD_STATUS. Waiting for it to be Running..."
    echo "You may need to wait a few minutes for the pod to be ready."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 3: Copying $SCRIPT_NAME to pod..."
kubectl cp "$SCRIPT_NAME" "$POD_NAME:/maad-vol/SPANet/$SCRIPT_NAME"

echo "Step 4: Making script executable in pod..."
kubectl exec "$POD_NAME" -- chmod +x "/maad-vol/SPANet/$SCRIPT_NAME"

echo "Step 5: Finding h5 files in $TRAINING_SETS_DIR..."
H5_FILES=$(kubectl exec "$POD_NAME" -- find "$TRAINING_SETS_DIR" -name "*.h5" -type f 2>/dev/null | tr -d '\r')

if [ -z "$H5_FILES" ]; then
    echo "Warning: No h5 files found in $TRAINING_SETS_DIR"
    echo "Listing directory contents:"
    kubectl exec "$POD_NAME" -- ls -la "$TRAINING_SETS_DIR" 2>/dev/null || echo "Directory may not exist"
    exit 1
fi

echo "Found h5 files:"
echo "$H5_FILES" | while read -r file; do
    echo "  - $file"
done

echo ""
echo "Step 6: Processing h5 files..."
echo "$H5_FILES" | while read -r file; do
    if [ -n "$file" ]; then
        echo "Processing: $file"
        kubectl exec "$POD_NAME" -- python3 "/maad-vol/SPANet/$SCRIPT_NAME" \
            "$file" \
            --mass-classes "$MASS_CLASSES"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully processed $file"
        else
            echo "  ✗ Failed to process $file"
        fi
        echo ""
    fi
done

echo "Done! All h5 files have been processed."
echo ""
echo "Next steps:"
echo "1. Test with one epoch: kubectl create -f jobs/global_classification_only/test_one_epoch.yaml"
echo "2. If test succeeds, launch full training: kubectl create -f jobs/global_classification_only/global_classification_only.yaml"
