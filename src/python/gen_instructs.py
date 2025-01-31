# /usr/bin/env python3
from QuAQ.tools.ansatz import CQ_Ansatz
from pathlib import Path
import csv
import argparse

OPS = ["rx", "ry", "rz", "cx"]


def flatten_to_ints(data: list[dict[str, tuple[str | int, int]]]) -> list[list[int]]:
    res = []

    for instruction_dict in data:
        for op, values in instruction_dict.items():
            line = [OPS.index(op)]
            if op == "cx":
                line.extend(values)
            else:
                line.extend([int(values[0], 2), values[1]])
            res.append(line)

    return res


def write_ints(data: list[list[int]], save_file: Path):
    with open(save_file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(data)


def get_entangler_map(qubit_count: int):
    all_numbers = list(range(qubit_count))
    all_pairs = zip(all_numbers[:-1], all_numbers[1:])
    res = [[list(i) for i in all_pairs]]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qubit_count")
    parser.add_argument("depth")
    parser.add_argument("savefile")
    args = parser.parse_args()

    num_qubits = int(args.qubit_count)
    depth = int(args.depth)
    entangler_map = get_entangler_map(qubit_count=num_qubits)
    savefile = Path(args.savefile)

    qc = CQ_Ansatz(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rx"],
        initial_state=[],
        entanglement_blocks="cx",
        entangler_map=entangler_map,
        depth=[depth],
        ansatz_type="heuristic",
        reduce_parameters=False,
    )
    ints = flatten_to_ints(data=qc.circuit)
    print(qc.circuit)
    savefile.parent.mkdir(exist_ok=True, parents=True)
    write_ints(data=ints, save_file=Path(savefile))


if __name__ == "__main__":
    main()  # "python gen_instructs.py 20 100 q20d100.txt" creates q20d100.txt
