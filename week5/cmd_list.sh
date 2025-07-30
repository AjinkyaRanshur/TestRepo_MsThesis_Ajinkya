#!/bin/bash

start=1
end=3


for ((i=start;i<=end;i++)); do
    config="config$i"
    python3 main.py --config "$config"
done

