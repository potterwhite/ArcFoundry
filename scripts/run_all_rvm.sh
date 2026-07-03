#!/bin/bash
# Copyright (c) 2026 ArcFoundry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
