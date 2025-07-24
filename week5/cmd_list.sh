#!/bin/bash

start=0
end=2


for ((i=start;i<=end;i++)); do
    config="config$i"
    python3 main.py --config "$config"
done

