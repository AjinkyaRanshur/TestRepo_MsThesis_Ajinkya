#!/bin/bash

echo "Enter commit message:"
read message

branch=$(git rev-parse --abbrev-ref HEAD)
echo "[INFO] Current branch: $branch"

# Stage everything
git add -A

# Explicitly unstage heavy files
git reset \
    *.png *.jpg *.jpeg *.svg *.pdf \
    *.pth *.pt *.ckpt *.h5 *.onnx \
    *.npy *.npz *.csv *.tsv *.hdf5 \
    *.log 2>/dev/null

# Commit if something remains
if git diff --cached --quiet; then
    echo "[INFO] No lightweight changes to commit."
else
    git commit -m "$message"
    git push origin "$branch"
fi

echo "Done. Press Enter..."
read

