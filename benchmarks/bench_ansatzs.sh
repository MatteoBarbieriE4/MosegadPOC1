#!/bin/env bash
# by Simone Rizzo (simone.rizzo@e4company.com)
# for E4 Computer Engineering SPA

# Usage: ./bench_ansatzs.sh raw_times_folder

if [[ -z $CUDA_QUANTUM_PATH ]]; then
    echo "CUDA_QUANTUM_PATH not set, nvq++ not available"
    exit 1
fi

script_dir=$(dirname "$0")
tmp_suffix=$(head /dev/urandom | md5sum | head -c 16)
build_dir="$script_dir/build-$(uname -m)-${tmp_suffix}"
# build_dir="$script_dir/bench_build"
circuits_dir="$script_dir/bench_circuits"
# times_dir="$script_dir/times"
times_dir="$1"
mkdir -p "$circuits_dir"
mkdir -p "$build_dir"
mkdir -p "$times_dir"

echo "Compiling executables..."
nvq++ "$script_dir/ansatz.cpp" -o "$build_dir/ansatz_gpu.out" --target nvidia && echo -ne "1/2"\\r
nvq++ "$script_dir/ansatz.cpp" -o "$build_dir/ansatz_cpu.out" --target qpp-cpu && echo "2/2"

# Qubits: 10, 20, 30
# Depth: 10, 30, 50, 70, 100

test_times () {
    local build_dir csv_file circuits_dir qubits depth start end runtime_c runtime_g exit_cpu exit_gpu
    build_dir=$1
    csv_file="$2/time_raw.csv"
    mkdir -p "$2"
    circuits_dir=$3
    echo "qubits,depth,cpu,gpu,exit_cpu,exit_gpu" > "$csv_file"

    for qubits in 30
    do 
        echo "Qubits: $qubits"
        for depth in 10 30 50 70 100
        do
            echo "Depth: $depth"
            for round in {1..10}
            do
                start=$(date +%s.%N)
                #echo "$build_dir/ansatz_cpu.out" "$circuits_dir/q${qubits}d${depth}.txt"
                "$build_dir/ansatz_cpu.out" "$circuits_dir/q${qubits}d${depth}.txt" > /dev/null
                exit_cpu=$?
                #exit_cpu=999
                end=$(date +%s.%N)
                runtime_c=$( echo "$end - $start" | bc -l )

                start=$(date +%s.%N)
                "$build_dir/ansatz_gpu.out" "$circuits_dir/q${qubits}d${depth}.txt" > /dev/null
                exit_gpu=$?
                end=$(date +%s.%N)
                runtime_g=$( echo "$end - $start" | bc -l )

                # Getting all data. Average (and stdev) can be calculated during postprocessing
                echo "$qubits,$depth,$runtime_c,$runtime_g,$exit_cpu,$exit_gpu" >> "$csv_file"
            done
        done
    done
    
    echo "Benchmark done"
}

echo "Test execution"

test_times "$build_dir" "$times_dir" "$circuits_dir"

rm -rf "$build_dir"
