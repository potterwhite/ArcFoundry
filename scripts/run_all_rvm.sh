#!/bin/bash

# Run all RVM configurations and sync to evboard

echo "=== Running all RVM configurations ==="
echo "Total: 19 configurations"
echo ""

# Get list of all RVM config files
CONFIG_FILES=$(ls configs/rvm/*.yaml)

# Counter
SUCCESS=0
FAILED=0

# Run each configuration
for config in $CONFIG_FILES; do
    echo "----------------------------------------"
    echo "Running: $config"
    echo "----------------------------------------"

    # Run ArcFoundry
    if ./arc "$config"; then
        echo "✓ Success: $config"
        ((SUCCESS++))
    else
        echo "✗ Failed: $config"
        ((FAILED++))
    fi

    echo ""
done

echo "========================================"
echo "Summary:"
echo "  Success: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Total: 19"
echo "========================================"

# Find all generated .rknn files
echo ""
echo "=== Generated .rknn files ==="
find output/ -name "*.rknn" -type f | sort

# Sync to evboard
echo ""
echo "=== Syncing to evboard:~ ==="
echo "Password: 123"

# Create a list of .rknn files
RKNN_FILES=$(find output/ -name "*.rknn" -type f)

if [ -n "$RKNN_FILES" ]; then
    # Sync using rsync with sshpass
    echo "$RKNN_FILES" | rsync -avz --progress -e "sshpass -p 123 ssh -o StrictHostKeyChecking=no" --files-from=- . evboard:~/
    echo "✓ Sync completed"
else
    echo "✗ No .rknn files found"
fi

echo ""
echo "=== Done ==="
