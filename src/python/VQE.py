import csv
import time

import_start = time.time()
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import cudaq
import numpy as np
from config import DATA_DIR
from cudaq import spin
from cudaq.qis import *
from QuAQ.tools.ansatz import CQ_Ansatz

import_end = time.time()
print("Loaded imports in", import_end - import_start)


OPS = ["rx", "ry", "rz", "cx"]


@dataclass
class VQEResult:
    energy: float
    ham_time: float
    quaq_time: float
    manipulation_time: float
    vqe_time: float
    total_time: float


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


def transpose_matrix(in_data: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*in_data)]


"""class Instruct(object):    
    def __init__(self, op: int, src: int, dest: int):        
        self.operation = op       
        self.src = src        
        self.dest = dest
        
def MakeInstrArray(data: list[dict[str, tuple[str | int, int]]]):
    arr = []
    res = flatten_to_ints(data)
    for i in res:
        instr = Instruct(i[0],i[1],i[2])
        arr.append(instr)
    return arr"""

"""@cudaq.kernel
def MakeCircuit(qc:cudaq.qview, to_do_ops:list[Instruct], param:list[int]):
    for i in to_do_ops:
        if i.operation == 0:
            rx(param[i.src],qc[i.dest])
        elif i.operation == 1:
            ry(param[i.src],qc[i.dest])
        elif i.operation == 2:
            rz(param[i.src],qc[i.dest])
        elif i.operation == 3:
            x.ctrl(qc[i.src],qc[i.dest])"""

"""@cudaq.kernel
def MakeCircuit1(qc:cudaq.qview, to_do_ops:list[list[int]], param:list[int]):
    for i in to_do_ops:
        if i[0] == 0:
            rx(param[i[1]],qc[i[2]])
        elif i[0] == 1:
            ry(param[i[1]],qc[i[2]])
        elif i[0] == 2:
            rz(param[i[1]],qc[i[2]])
        elif i[0] == 3:
            x.ctrl(qc[i[1]],qc[i[2]])"""


@cudaq.kernel
def MakeCircuit2(
    qc: cudaq.qview, ops: list[int], src: list[int], tgt: list[int], param: list[float]
):
    for i in range(len(ops)):
        if ops[i] == 0:
            rx(param[src[i]], qc[tgt[i]])
        elif ops[i] == 1:
            ry(param[src[i]], qc[tgt[i]])
        elif ops[i] == 2:
            rz(param[src[i]], qc[tgt[i]])
        elif ops[i] == 3:
            x.ctrl(qc[src[i]], qc[tgt[i]])


"""@cudaq.kernel
def RunAnsatz(n_q:int,par: list[int],t_d_op:list[Instruct]):
    qc = cudaq.qvector(n_q)
    MakeCircuit(qc,t_d_op,par)
    mz(qc)"""


@cudaq.kernel
def RunAnsatz1(
    n_q: int, par: list[float], ops: list[int], src: list[int], tgt: list[int]
):
    qc = cudaq.qvector(n_q)
    MakeCircuit2(qc, ops, src, tgt, par)
    mz(qc)


def GetEntanglerMap(n_qubits) -> list[list[int]]:
    res = []
    for i in range(n_qubits - 1):
        res.append([i, i + 1])
    return res


def DoAnsatz():
    num_qubits = 6
    dep = 4
    qc_QuAQ = CQ_Ansatz(
        num_qubits=num_qubits,
        rotation_blocks=["ry"],
        initial_state=[],
        entanglement_blocks="cx",
        entangler_map=[GetEntanglerMap(num_qubits)],
        depth=[dep],
        ansatz_type="heuristic",
        reduce_parameters=False,
    )
    list_size = 2 * num_qubits * dep
    par = [random.uniform(0, 2) * np.pi for _ in range(list_size)]
    ints = flatten_to_ints(data=qc_QuAQ.circuit)
    ops, src, tgt = transpose_matrix(ints)
    res = cudaq.sample(RunAnsatz1, num_qubits, par, ops, src, tgt)
    print(res)


# DoAnsatz()


def MakeHamiltonian(
    h: dict[str, float]
):  # -> cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.SpinOperator
    res = 0
    for pauli_word, coeff in h.items():
        term = cudaq.SpinOperator.from_word(pauli_word)
        res += term * coeff

    #print(res)
    return res


# print(MakeHamiltonian(Ham))


@cudaq.kernel
def VQE_step(
    n_q: int, par: list[float], ops: list[int], src: list[int], tgt: list[int]
):
    qc = cudaq.qvector(n_q)
    MakeCircuit2(qc, ops, src, tgt, par)


def RunCustomHam(
    pauli_ops_file: Path, num_qubits: int, depth: int, params: list[float] | None = None
):
    ham_start = time.time()
    with open(pauli_ops_file, "rb") as fp:
        content = pickle.load(fp)

    first_key = next(iter(content))
    Ham = content[first_key]
    H = MakeHamiltonian(Ham)
    # print(H.to_string())

    quaq_start = time.time()
    qc_QuAQ = CQ_Ansatz(
        num_qubits=num_qubits,
        rotation_blocks=["ry"],
        initial_state=[],
        entanglement_blocks="cx",
        entangler_map=[GetEntanglerMap(num_qubits)],
        depth=[depth],
        ansatz_type="heuristic",
        reduce_parameters=False,
    )

    manipulation_start = time.time()
    list_size = num_qubits + num_qubits * depth
    if params is None:
        par = [random.uniform(0, 2) * np.pi for _ in range(list_size)]
    else:
        assert len(params) == list_size
        par = params

    ints = flatten_to_ints(data=qc_QuAQ.circuit)

    # with open("spin_ops.txt", "w") as fp:
    #     writer = csv.writer(fp)
    #     writer.writerows(ints)

    ops, srcs, dests = transpose_matrix(ints)

    vqe_start = time.time()
    energy = cudaq.observe(VQE_step, H, num_qubits, par, ops, srcs, dests).expectation()
    vqe_end = time.time()
    total_time = vqe_end - ham_start
    return VQEResult(
        energy=energy,
        ham_time=quaq_start - ham_start,
        quaq_time=manipulation_start - quaq_start,
        manipulation_time=vqe_start - manipulation_start,
        vqe_time=vqe_end - vqe_start,
        total_time=total_time,
    )


def RunSpinHam(params: list[float] | None = None) -> VQEResult:
    a = RunCustomHam(
        pauli_ops_file=DATA_DIR / "pauli_ops" / "spin_cuda_2x4_isotropic.pkl",
        num_qubits=8,
        depth=20,
        params=params,
    )
    return a


def RunN2Ham(params: list[float] | None = None) -> VQEResult:
    b = RunCustomHam(
        pauli_ops_file=DATA_DIR / "pauli_ops" / "N2_cuda_sto-3g_R-HF.pkl",
        num_qubits=20,
        depth=10,
        params=params,
    )
    return b


def RunNH3Ham(params: list[float] | None = None) -> VQEResult:
    c = RunCustomHam(
        pauli_ops_file=DATA_DIR / "pauli_ops" / "NH3_cuda_sto-3g_R-HF_is_frozen.pkl",
        # pauli_ops_file=DATA_DIR / "pauli_ops" / "pauli_op_nh3_random_sm_dict.pkl",
        num_qubits=14,
        depth=6,
        params=params,
    )
    return c


if __name__ == "__main__":
    # with open(DATA_DIR / "misc" / "C_final_parameters.pkl", "rb") as fp:
    #     c_params = pickle.load(fp)
    # for i in range(50): 
    #     print(RunSpinHam(params=c_params[i]).energy)

    # with open(DATA_DIR / "misc" / "D_initial_parameters.pkl", "rb") as fp:
    #     d_params = pickle.load(fp)
    # for i in range(len(d_params)): 
    #     print(RunN2Ham(params=d_params[i]).energy)

    with open(DATA_DIR / "misc" / "B_final_parameters.pkl", "rb") as fp:
        b_params = pickle.load(fp)
    for i in range(50): 
        print(RunNH3Ham(params=b_params[i]).energy)
