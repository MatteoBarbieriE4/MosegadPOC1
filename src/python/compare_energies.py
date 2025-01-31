import argparse
import numpy as np
from pathlib import Path

from config import DATA_DIR

def compare(ref: Path, new: Path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("reference_file")
    # parser.add_argument("new_file")
    # args = parser.parse_args()

    with open(ref) as fp:
        ref = np.array(list(map(float, fp.read().strip().split("\n"))))
    
    with open(new) as fp:
        new = np.array(list(map(float, fp.read().strip().split("\n"))))
    
    if len(new) < len(ref):
        ref = ref[:len(new)]
    elif len(ref) < len(new):
        new = new[:len(ref)]
    diffs =new-ref
    print("Difference between new values and reference ones:")
    print(f"Mean:\t{diffs.mean()}")
    print(f"Std:\t{diffs.std()}")
    print(f"Min:\t{diffs.min()}")
    print(f"Max:\t{diffs.max()}")
    assert len(ref) == len(new)

if __name__ == "__main__":
    print("> System B (NH3) base QuAQ, nuovo E4 CUDA-Q")
    compare(ref = DATA_DIR / "misc" / "B_energy.txt", new= DATA_DIR / "misc" / "B_energy_e4.txt")
    print("> System B (NH3) base E4 Qiskit, nuovo E4 CUDA-Q")
    compare(ref = DATA_DIR / "misc" / "B_energy_qiskit_e4.txt", new= DATA_DIR / "misc" / "B_energy_e4.txt")
    print("> System C (spin) base QuAQ, nuovo E4 CUDA-Q")
    compare(ref = DATA_DIR / "misc" / "C_energy.txt", new= DATA_DIR / "misc" / "C_energy_e4.txt")
    print()
    print("> System B (NH3) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.7.1")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "B_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "B_GPU.txt")
    print("> System C (Spin) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.7.1")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "C_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "C_GPU.txt")
    print("> System D (N2) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.7.1")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "D_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "D_GPU.txt")
    print()
    print("> System B (NH3) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.9.0")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "B_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "B_GPU_0.9.0.txt")
    print("> System C (Spin) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.9.0")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "C_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "C_GPU_0.9.0.txt")
    print("> System D (N2) base CUDA-Q CPU, nuovo CUDA-Q GPU v0.9.0")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "D_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "D_GPU_0.9.0.txt")
    print()
    print("> System B (NH3) base CUDA-Q CPU, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "B_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "B_GPU_64.txt")
    print("> System C (Spin) base CUDA-Q CPU, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "C_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "C_GPU_64.txt")
    print("> System D (N2) base CUDA-Q CPU, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "energies_CUDA-Q" / "D_CPU.txt", new= DATA_DIR / "energies_CUDA-Q" / "D_GPU_64.txt")
    print()
    print("> System B (NH3) base QuAQ, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "misc" / "B_energy.txt", new= DATA_DIR / "energies_CUDA-Q" / "B_GPU_64.txt")
    print("> System C (Spin) base QuAQ, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "misc" / "C_energy.txt", new= DATA_DIR / "energies_CUDA-Q" / "C_GPU_64.txt")
    print("> System D (N2) base QuAQ, nuovo CUDA-Q GPU fp64")
    compare(ref = DATA_DIR / "misc" / "D_energy.txt", new= DATA_DIR / "energies_CUDA-Q" / "D_GPU_64.txt")
