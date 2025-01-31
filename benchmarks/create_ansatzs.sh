#!/bin/env bash

# Qubits: 10, 20, 30
# Depth: 10, 30, 50, 70, 100

create_ansatzs () {
    local ansatzs_dir
    ansatzs_dir=$1
    for qubits in 10 20 30
    do 
        echo "Qubits: $qubits"
        for depth in 10 30 50 70 100
        do
            echo "Depth: $depth"
            python src/python/gen_instructs.py "$qubits" "$depth" "$ansatzs_dir/q${qubits}d${depth}.txt"
        done
    done
}

create_ansatzs "bench_circuits"
