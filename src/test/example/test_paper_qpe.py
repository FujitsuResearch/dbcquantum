import cmath
import math
import typing

import numpy.linalg as LA
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate, PhaseGate
from qiskit.opflow import H, OperatorBase, StateFn, T, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils
from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.utils import eq_state

# from .paper_qft import make_qft


def controlled_Phase_Gate(l: int) -> Gate:
    return PhaseGate(2 * math.pi / (2**l)).control(num_ctrl_qubits=1)


def make_qft(num_qubits: int) -> AssertQuantumCircuit:
    circuit = AssertQuantumCircuit(num_qubits)

    for i in range(0, num_qubits):
        circuit.append(HGate(), [i])
        for j in range(1, num_qubits - i):
            circuit.append(
                utils.decompose(
                    controlled_Phase_Gate(j + 1),
                    basis_gates=["h", "rx", "rz", "cx"],
                ),
                [i + j, i],
            )

    return circuit


def make_qpe(
    U_gate: Gate, U: OperatorBase, n_precision: int
) -> AssertQuantumCircuit[float]:
    n_total: int = U_gate.num_qubits + n_precision
    circuit: AssertQuantumCircuit[float] = AssertQuantumCircuit(n_total)

    for i in range(n_precision):
        circuit.append(HGate(), [i])
        circuit.append(
            utils.decompose(
                U_gate.power(2**i).control(num_ctrl_qubits=1),
                basis_gates=["h", "rx", "rz", "cx"],
            ),
            [i] + list(range(n_precision, n_precision + U_gate.num_qubits)),
        )

    circuit.append(
        make_qft(n_precision).inverse(),
        range(n_precision),
    )

    def post_state_is_eigen_vector(
        pre_state, post_state: Statevector, param: float
    ) -> bool:
        atol = param
        psi: StateFn = StateFn(
            utils.partial_state(
                pre_state, range(n_precision, n_precision + U_gate.num_qubits)
            )
        )

        state_js = [
            0 if eq_state(state, Zero) else 1
            for state in utils.split_each_qubit_states(
                utils.partial_state(post_state, range(n_precision))
            )
        ]

        eigen = cmath.exp(1j * 2 * math.pi * utils.bin_frac_to_dec(state_js))
        return eq_state(U @ psi, eigen * psi, atol=atol)  # type: ignore

    circuit.add_condition_use_param(
        "[qpe] post state is eigen vector", post_state_is_eigen_vector
    )

    return circuit


def test_qpe_1():
    n_precision: int = 5

    U: OperatorBase = T ^ H  # type: ignore
    c = QuantumCircuit(2)
    c.t(1)
    c.h(0)
    U_gate: Gate = c.to_gate()

    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        # calculate eigenvalue and eigenstate
        eigenstate = StateFn(eigen[1][:, i])

        init_state: OperatorBase = StateFn("")
        for _ in range(n_precision):
            init_state = typing.cast(OperatorBase, init_state ^ Zero)
        init_state = eigenstate ^ init_state  # type: ignore

        circuit = make_qpe(U_gate, U, n_precision)
        circuit.run(init_state=init_state, param=1e-1)


def test_qpe_2():
    n_precision: int = 5

    U: OperatorBase = T ^ H  # type: ignore
    c = QuantumCircuit(2)
    c.t(1)
    c.h(0)
    U_gate: Gate = c.to_gate()

    eigen = LA.eig(U.to_matrix())

    phi = None
    eigen_states = []
    Zeros = StateFn(utils.make_zeros_state(n_precision))

    for i in range(len(eigen[0])):
        # calculate eigenvalue and eigenstate
        eigenstate = StateFn(eigen[1][:, i])
        phi = (phi + eigenstate * (1 / 2)) if phi is not None else (eigenstate * (1 / 2))  # type: ignore
        eigen_states.append((1 / 2, eigenstate ^ Zeros))

    init_state = phi ^ Zeros  # type: ignore

    qpe = make_qpe(U_gate, U, n_precision)

    circuit = AssertQuantumCircuit(U_gate.num_qubits + n_precision)
    circuit.append_superposition(
        "qpe_superposition",
        qpe,
        pre_superposition_states=eigen_states,
        post_superposition_states=None,
        qargs=range(U_gate.num_qubits + n_precision),
    )
    circuit.run(init_state=init_state, param=1e-1)


# def test_qpe_3():
#     n_precision: int = 5

#     U: OperatorBase = T  # type: ignore
#     c = QuantumCircuit(1)
#     c.h(0)
#     U_gate: Gate = c.to_gate()

#     eigen = LA.eig(U.to_matrix())

#     phi = None
#     eigen_states = []
#     Zeros = StateFn(utils.make_zeros_state(n_precision))

#     for i in range(len(eigen[0])):
#         # calculate eigenvalue and eigenstate
#         eigenstate = StateFn(eigen[1][:, i])
#         phi = (phi + eigenstate * (1 / 2)) if phi is not None else (eigenstate * (1 / 2))  # type: ignore
#         eigen_states.append((1 / 2, eigenstate ^ Zeros))

#     init_state = phi ^ Zeros  # type: ignore

#     qpe = make_qpe(U_gate, U, n_precision)

#     circuit = AssertQuantumCircuit(U_gate.num_qubits + n_precision)
#     circuit.append_superposition(
#         "qpe_superposition",
#         qpe,
#         pre_superposition_states=eigen_states,
#         post_superposition_states=None,
#         qargs=range(U_gate.num_qubits + n_precision),
#     )
#     circuit.run(init_state=init_state, param=1e-1)
