#!/bin/bash

start=0
end=323
N=10  # Number of parallel jobs

pids=()

for ((i=start;i<=end;i++)); do
    config="config$i"
    python3 main.py --config "$config" &  # Run in background
    pids+=($!)  # Store PID

    # Wait if number of background jobs reaches N
    if (( ${#pids[@]} >= N )); then
        wait -n  # Wait for *any* one to finish
        # Remove completed PIDs from array (optional, for cleanup)
        tmp=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                tmp+=("$pid")
            fi
        done
        pids=("${tmp[@]}")
    fi
done

# Wait for remaining background jobs to finish
wait

