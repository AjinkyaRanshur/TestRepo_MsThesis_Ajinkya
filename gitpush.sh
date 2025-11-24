#!/bin/bash

echo "Enter commit message:"
read message

# Determine current branch
branch=$(git rev-parse --abbrev-ref HEAD)

echo "[INFO] Current branch: $branch"

# Stage changes
git add -A

# Commit
git commit -m "$message"

# Push to the same branch
git push origin "$branch"

echo "Done. Press Enter..."
read

