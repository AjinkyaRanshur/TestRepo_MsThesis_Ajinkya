#!/bin/bash

start=0
end=17


for ((i=start;i<=end;i++)); do
    config="config$i"
    python3 main.py --config "$config"
done

python3 main.py --config control_pc
python3 main.py --config control_fffb

