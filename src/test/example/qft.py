import cmath
import math

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, PhaseGate
from qiskit.opflow import One, StateFn, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils
from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.utils import eq_state


def controlled_R_Gate(l: int) -> Gate:
    return PhaseGate(2 * math.pi / (2**l)).control(num_ctrl_qubits=1)


def make_qft(num_qubits: int) -> AssertQuantumCircuit:
    circuit = AssertQuantumCircuit(num_qubits)

    for i in range(0, num_qubits):
        inner_circuit = AssertQuantumCircuit(num_qubits - i)
        inner_circuit.append(HGate(), [0])
        for j in range(1, num_qubits - i):
            inner_circuit.append(controlled_R_Gate(j + 1), [j, 0])

        inner_circuit.add_pre_condition(
            "[qft inner condition: qubit "
            + str(i)
            + "] input state of qft must be |0> or |1>",
            lambda pre_state: all(
                eq_state(s, Zero) or eq_state(s, One)
                for s in utils.split_each_qubit_states(pre_state)
            ),
        )

        inner_circuit.add_condition(
            "[qft inner condition: qubit "
            + str(i)
            + "] input and output states must be the same except for the first qubit",
            (  # See
                # https://stackoverflow.com/questions/19837486/lambda-in-a-loop
                lambda i: lambda pre_state, post_state: eq_state(
                    utils.partial_state(pre_state, range(1, num_qubits - i)),
                    utils.partial_state(post_state, range(1, num_qubits - i)),
                )
            )(i),
        )

        def inner_circuit_condition_first_qubit(
            pre_state: Statevector, post_state: Statevector
        ) -> bool:
            splitted_input_state: list[(Statevector)] = utils.split_each_qubit_states(
                pre_state
            )

            input_state_01: list[int] = [
                0 if eq_state(state, Zero) else 1 for state in splitted_input_state
            ]

            a_inner: complex = cmath.exp(
                1j * 2 * math.pi * utils.bin_frac_to_dec(input_state_01, True)
            )

            state: Statevector = 1 / math.sqrt(2) * (Zero + (a_inner * One))  # type: ignore
            return eq_state(state, utils.partial_state(post_state, [0]))

        inner_circuit.add_condition(
            "[qft inner condition: qubit "
            + str(i)
            + "] the first qubit's state condition",
            inner_circuit_condition_first_qubit,
        )

        circuit.append(inner_circuit, range(i, inner_circuit.num_qubits + i))
        circuit.append(Barrier(num_qubits), range(num_qubits))

    # for i in range(num_qubits // 2):
    #     circuit.append(SwapGate(), [i, num_qubits - i - 1])

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
            # desired_state = desired_state ^ ((1 / (math.sqrt(2))) * (Zero + a * One))  # type: ignore

        return eq_state(post_state, desired_state)

    circuit.add_condition("fqt state condition", condition)
    return circuit
