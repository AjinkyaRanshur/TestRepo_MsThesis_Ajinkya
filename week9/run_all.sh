#!/bin/bash

# Run all configurations in parallel
echo "Running all configurations in parallel..."

python3 main.py --config configclass10 &  
pid1=$!

python3 main.py --config configclass &  
pid2=$!

python3 main.py --config configill10 &  
pid3=$!

python3 main.py --config configill &  
pid4=$!

# Wait for all to finish
wait $pid1 $pid2 $pid3 $pid4

echo "All runs completed."

