import cmath
import math

from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, PhaseGate
from qiskit.opflow import One, StateFn, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils
from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.utils import eq_state


def controlled_Phase_Gate(l: int) -> Gate:
    return PhaseGate(2 * math.pi / (2**l)).control(num_ctrl_qubits=1)


def make_qft(num_qubits: int) -> AssertQuantumCircuit:
    circuit = AssertQuantumCircuit(num_qubits)

    for i in range(0, num_qubits):
        circuit.append(HGate(), [i])
        for j in range(1, num_qubits - i):
            circuit.append(controlled_Phase_Gate(j + 1), [i + j, i])

    circuit.add_pre_condition(
        "all qft input states must be |0> or |1>",
        lambda pre_state: all(
            eq_state(s, Zero) or eq_state(s, One)
            for s in utils.split_each_qubit_states(pre_state)
        ),
    )

    def condition(pre_state: Statevector, post_state: Statevector) -> bool:
        splitted_input_state: list[(Statevector)] = utils.split_each_qubit_states(
            pre_state
        )

        input_state_01: list[int] = [
            0 if eq_state(state, Zero) else 1 for state in splitted_input_state
        ]

        a: complex = cmath.exp(
            1j * 2 * math.pi * utils.bin_frac_to_dec(input_state_01, True)
        )

        desired_state: StateFn = (1 / (math.sqrt(2))) * (Zero + a * One)  # type: ignore

        for i in range(1, num_qubits):
            a = cmath.exp(
                1j
                * 2
                * math.pi
                * utils.bin_frac_to_dec(list(input_state_01[i:num_qubits]), True)
            )

            desired_state = ((1 / (math.sqrt(2))) * (Zero + a * One)) ^ desired_state  # type: ignore

        return eq_state(post_state, desired_state)

    circuit.add_condition("fqt state condition", condition)
    return circuit
