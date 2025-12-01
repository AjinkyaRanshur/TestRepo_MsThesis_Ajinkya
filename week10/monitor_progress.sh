#!/bin/bash

# ============================================================
# Monitor Experiment Progress
# Usage: ./monitor_progress.sh [log_directory]
# ============================================================

LOG_DIR=${1:-"experiment_logs"}

clear
echo "============================================================"
echo "Experiment Progress Monitor"
echo "============================================================"
echo "Monitoring directory: $LOG_DIR"
echo "Press Ctrl+C to exit"
echo "============================================================"
echo ""

# Function to display progress
show_progress() {
    # Find all progress files
    PROGRESS_FILES=$(find "$LOG_DIR" -name "progress_*.txt" -type f 2>/dev/null)
    
    if [ -z "$PROGRESS_FILES" ]; then
        echo "No progress files found in $LOG_DIR"
        return
    fi
    
    # Display each progress file
    for file in $PROGRESS_FILES; do
        if [ -f "$file" ]; then
            echo ""
            echo "─────────────────────────────────────────────────────────"
            echo "File: $(basename $file)"
            echo "─────────────────────────────────────────────────────────"
            cat "$file"
        fi
    done
    
    echo ""
    echo "============================================================"
    echo "SLURM Queue Status"
    echo "============================================================"
    squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
    echo ""
}

# Monitor in a loop
while true; do
    clear
    echo "============================================================"
    echo "Experiment Progress Monitor - $(date)"
    echo "============================================================"
    
    show_progress
    
    echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done
