from qiskit.quantum_info import SparsePauliOp
# from qiskit.primitives import Estimator, BackendEstimator
from qiskit_aer.primitives import Estimator
import pickle
from qiskit.circuit import Parameter
from qiskit import Aer, QuantumCircuit
from QuAQ.tools.ansatz import Ansatz
import numpy as np

from config import DATA_DIR


def GetEntanglerMap(n_qubits) -> list[list[int]]:
    res = []
    for i in range(n_qubits - 1):
        res.append([i, i + 1])
    return res


def energy_eval(
    paulis: dict[str, complex], num_qubits: int, depth: int, theta_values: list[float]
):
    hamiltonian_sparse = SparsePauliOp.from_list(
        [(p[::-1], coeff) for p, coeff in paulis.items()]
    )

    qc_QuAQ = Ansatz(
        num_qubits=num_qubits,
        rotation_blocks=["ry"],
        initial_state=[],
        entanglement_blocks="cx",
        entangler_map=[GetEntanglerMap(num_qubits)],
        depth=[depth],
        ansatz_type="heuristic",
        reduce_parameters=False,
    )

    quaq_circuit = qc_QuAQ.var_form

    n_params = num_qubits + num_qubits * depth

    # print(n_params, len(theta_values))
    assert n_params == len(theta_values)

    #theta_params = [Parameter(f"theta_{i}") for i in range(len(theta_values))]
    theta_params = quaq_circuit.parameters

    # qc = QuantumCircuit(num_qubits)

    # for instruction_dict in quaq_circuit:
    #     for op, values in instruction_dict.items():
    #         if op == "ry":
    #             qc.ry(theta_params[int(values[0], 2)], values[1])
    #         elif op == "cx":
    #             qc.cx(*values)
    # Puoi farci qc.draw(), ha la stessa forma di come generiamo noi

    bound_qc = quaq_circuit.assign_parameters(
        {param: value for param, value in zip(theta_params, theta_values)}
    )

    # simulator = Aer.get_backend('aer_simulator_statevector')
    # simulator.se
    # print(simulator)
    # estimator = BackendEstimator(backend=simulator) # Not deterministic results
    estimator = Estimator(approximation=True)

    # Perform a single energy evaluation
    job = estimator.run([bound_qc], [hamiltonian_sparse], shots=None)
    energy = job.result().values[0].real

    print(energy)
    # h_mat = hamiltonian_sparse.to_matrix()
    # eig_va, eig_vec = np.linalg.eig(h_mat)
    # eig_va.sort()
    # print(eig_va[0])



def energy_eval_nh3():
    with open(
        #DATA_DIR / "pauli_ops" / "NH3_cuda_sto-3g_R-HF_is_frozen.pkl", "rb"
        DATA_DIR / "pauli_ops" / "pauli_op_nh3_random_sm_dict.pkl", "rb"
    ) as fp:
        paulis = pickle.load(fp)["pauli_op"]
    with open(DATA_DIR / "misc" / "B_final_parameters.pkl", "rb") as fp:
        b_params = pickle.load(fp)
    # for params_idx in range(len(b_params)):
    #     energy_eval(
    #         paulis=paulis, num_qubits=14, depth=6, theta_values=b_params[params_idx]
    #     )
    custom_vals = [2.760253401436559, 1.3954308483209563, 1.4005917922961018, -2.364965744699991, 0.44410801282275925, -3.0676037734663586, -0.9654622674617821, 0.6650285970575278, 0.2672677863410806, 0.460691060406206, 2.2762421091615916, -2.429527224292359, -1.6997399353197435, 2.1716449111900307, 0.2569489256439246, -1.1365539126222917, -1.970646723605526, -0.7513414771621729, -2.8732102963715214, 0.23158521822846856, 0.5226730305926539, 0.8780178863581556, -1.8738448399838283, 2.036879992462617, -1.382415767437493, -2.535288039847024, -2.9710055327769576, -2.790381100540941, 0.6723354944532551, 0.7758326149199934, 0.8280885101869209, 1.0988176646459258, 0.3510073437573378, -2.004002694179805, -1.1936416811201784, 2.676924519837337, -3.0585101612880257, 0.279782316624674, 2.619344087171581, 1.5860161835861613, -0.404560319955229, -1.4375336411697033, -1.9860428208148528, -0.7630822548083263, -1.6243279128539445, 2.6718032132609952, -0.6463898470116978, 1.6932747155764631, 1.689414601601639, 2.5807247624426104, -2.1116901421463368, -0.5997594129873858, 2.0930355397918836, -0.5062217171806389, -2.973883917659779, 0.6948128937577152, 1.1935750465799786, -2.6020556057251873, 1.1791525146988875, 1.8043268016643323, -0.5619335160745962, -1.0227925096501584, 2.7577484683490026, -0.1274258490180138, 0.2543200452792935, -1.9833738269406167, -1.9755395195726617, 0.4709334266747782, -2.1943331791090825, -0.014364675032664032, -1.1353871160019757, -1.928451599025605, -0.4961957377143875, 0.7452580956260646, -3.0152449390183955, -0.21321524075681175, -1.1286756020985713, -0.3424705232437777, 0.9762412274826353, 1.9702214466158665, 1.8026247674506346, 1.0298182134370135, 1.453985564304638, 1.2065002069666668, -2.3973649819837517, 3.1358369711293346, 0.501811061423532, 0.4017662362657881, -1.948234470584179, -0.6282681376307115, 2.3037757766849447, 1.4446479787411326, 2.114078651224754, 2.9626039207667727, 1.5510102101824617, 3.029192309070678, -1.5600965073034825, 0.3918000699202895]
    energy_eval(
        paulis=paulis, num_qubits=14, depth=6, theta_values=custom_vals
    )


if __name__ == "__main__":
    energy_eval_nh3()
