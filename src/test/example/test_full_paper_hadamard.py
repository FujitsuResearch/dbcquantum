import cmath
import math
import typing

import pytest
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, TGate
from qiskit.opflow import One, OperatorBase, Plus, StateFn, T, Zero
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
from qiskit.result.result import Result
from qiskit_aer import AerSimulator

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.err import MeasureConditionError
from dbcquantum.utils import decompose, eq_state, partial_state

# https://dojo.qulacs.org/ja/latest/notebooks/2.2_Hadamard_test.html
# https://utokyo-icepp.github.io/qc-workbook/quantum_computation.html

backed = AerSimulator()


def make_circ_hadamard_test(U_gate: Gate) -> AssertQuantumCircuit[OperatorBase]:
    circ: AssertQuantumCircuit[OperatorBase] = AssertQuantumCircuit(
        U_gate.num_qubits + 1
    )
    circ.append(HGate(), [0])
    ctrl_U = decompose(U_gate.control(), basis_gates=["h", "rx", "rz", "cx"])
    circ.append(ctrl_U, range(U_gate.num_qubits + 1))
    circ.append(HGate(), [0])

    def condition(
        pre_state: Statevector, post_state: Statevector, param: OperatorBase
    ) -> bool:
        U: OperatorBase = param
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))

        state_0 = ((psi + (U @ psi)) / 2) ^ Zero  # type: ignore
        state_1 = ((psi - (U @ psi)) / 2) ^ One  # type: ignore
        return eq_state(post_state, state_0 + state_1)

    circ.add_condition_use_param("condition1", condition)
    return circ


def test_hadamard_test_1():
    U_gate: Gate = TGate()
    circ_hadamard_test = make_circ_hadamard_test(U_gate)

    psi = Plus
    # U: OperatorBase = T
    U: OperatorBase = PrimitiveOp(Operator([[1, 0], [0, cmath.exp(1j * math.pi / 4)]]))

    circ_hadamard_test.run(init_state=psi ^ Zero, param=U)


def make_hadamard_test(
    psi_circuit: AssertQuantumCircuit, U_gate: Gate
) -> AQCMeasure[float, tuple[OperatorBase, OperatorBase, float]]:
    assert psi_circuit.num_qubits == U_gate.num_qubits

    circuit: AssertQuantumCircuit[OperatorBase] = AssertQuantumCircuit(
        U_gate.num_qubits + 1
    )
    circuit.append(psi_circuit, range(1, U_gate.num_qubits + 1))
    circuit.append(make_circ_hadamard_test(U_gate), range(U_gate.num_qubits + 1))

    def calc_exp_value(result: Result) -> float:
        counts: Counts = typing.cast(Counts, result.get_counts())

        p0: float = counts["0"] / (counts["0"] + counts["1"])
        p1: float = counts["1"] / (counts["0"] + counts["1"])

        estimated_exp = p0 - p1
        return estimated_exp

    def measure_condition(
        pre_measure_state: Statevector,
        result: Result,
        estimated_exp: float,
        param: tuple[OperatorBase, OperatorBase, float],
    ):
        U, psi, abs_tol = param
        actual_exp: float = ((~psi) @ U @ psi).eval().real  # type: ignore
        return cmath.isclose(actual_exp, estimated_exp, abs_tol=abs_tol)

    circuit_measure: AQCMeasure[float, tuple[OperatorBase, OperatorBase, float]] = (
        AQCMeasure(
            circuit,
            postprocess=calc_exp_value,
            qubit=[0],
            param_converter=lambda param: param[0],
        )
    )

    circuit_measure.add_condition_use_param(
        "eigenvalue real estimation", measure_condition
    )

    return circuit_measure


def test_hadamard_test_measure_1():
    U: OperatorBase = T
    U_gate: Gate = TGate()

    psi = Plus
    psi_circuit = AssertQuantumCircuit(1)
    psi_circuit.append(HGate(), [0])
    psi_circuit.add_post_condition(
        "psi is constructed", lambda post_state: eq_state(post_state, psi)
    )

    circuit_measure = make_hadamard_test(psi_circuit, U_gate)

    ans = circuit_measure.run(shots=100_000, param=(U, psi, 1e-02))
    print(ans)


def test_hadamard_test_measure_fail1():
    U: OperatorBase = T
    U_gate: Gate = TGate()

    psi = Plus
    psi_circuit = AssertQuantumCircuit(1)
    psi_circuit.append(HGate(), [0])
    psi_circuit.add_post_condition(
        "psi is constructed", lambda post_state: eq_state(post_state, psi)
    )

    circuit_measure = make_hadamard_test(psi_circuit, U_gate)

    with pytest.raises(MeasureConditionError) as e:
        ans = circuit_measure.run(shots=100_000, param=(U, psi, 1e-05))
        print(ans)

    assert "Condition Error occured in 'eigenvalue real estimation'" in str(e.value)
