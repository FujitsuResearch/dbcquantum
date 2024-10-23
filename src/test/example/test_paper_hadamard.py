import cmath
import math
import typing

import pytest
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, TGate, XGate
from qiskit.compiler import transpile
from qiskit.opflow import One, OperatorBase, Plus, StateFn, T, Zero
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
from qiskit.result.result import Result
from qiskit_aer import AerSimulator

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.err import StateConditionError
from dbcquantum.utils import decompose, eq_state, partial_state

# https://dojo.qulacs.org/ja/latest/notebooks/2.2_Hadamard_test.html
# https://utokyo-icepp.github.io/qc-workbook/quantum_computation.html


def make_circ_hadamal_test(U_gate: Gate, U: OperatorBase) -> AssertQuantumCircuit:
    circ = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circ.append(HGate(), [0])
    ctrl_U = decompose(U_gate.control(), basis_gates=["h", "rx", "rz", "cx"])
    circ.append(ctrl_U, range(U_gate.num_qubits + 1))
    circ.append(HGate(), [0])

    def condition(pre_state: Statevector, post_state: Statevector) -> bool:
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))

        state_0 = ((psi + (U @ psi)) / 2) ^ Zero  # type: ignore
        state_1 = ((psi - (U @ psi)) / 2) ^ One  # type: ignore
        return eq_state(post_state, state_0 + state_1)

    circ.add_condition("condition1", condition)
    return circ


backed = AerSimulator()


def test_hadamal_test_1():
    # U: OperatorBase = T
    U: OperatorBase = PrimitiveOp(Operator([[1, 0], [0, cmath.exp(1j * math.pi / 4)]]))
    U_gate: Gate = TGate()
    circ_hadamard_test = make_circ_hadamal_test(U_gate, U)

    psi = Plus
    circ_hadamard_test.run(init_state=psi ^ Zero)


def test_hadamal_test_measure_1():
    U: OperatorBase = T
    U_gate: Gate = TGate()
    real_estimation_circuit = make_circ_hadamal_test(U_gate, U)

    psi = Plus
    psi_circuit = AssertQuantumCircuit(1)
    psi_circuit.append(HGate(), [0])
    psi_circuit.add_post_condition(
        "psi is constructed", lambda post_state: eq_state(post_state, psi)
    )

    circuit = AssertQuantumCircuit(2)
    circuit.append(psi_circuit, [1])
    circuit.append(real_estimation_circuit, [0, 1])

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
        param: tuple[float, float],
    ):
        actual_exp, abs_tol = param
        return cmath.isclose(actual_exp, estimated_exp, abs_tol=abs_tol)

    circuit_measure: AQCMeasure = AQCMeasure(
        circuit, postprocess=calc_exp_value, qubit=[0]
    )

    circuit_measure.add_condition_use_param(
        "eigenvalue real estimation", measure_condition
    )

    actual_exp: complex = ((~psi) @ U @ psi).eval()  # type: ignore

    ans = circuit_measure.run(shots=100_000, param=(actual_exp.real, 1e-02))
    assert math.isclose(ans, actual_exp.real, abs_tol=1e-02)

    qiskit_circ, postprocess = circuit_measure.remove_assertions()
    # qiskit_circ.draw()

    transpiled_circ = transpile(qiskit_circ, backed)
    result = backed.run(transpiled_circ, shots=100_000).result()

    ans2 = postprocess(result)
    assert math.isclose(ans2, actual_exp.real, abs_tol=1e-02)


def make_circ_hadamal_test_fail(U_gate: Gate, U: OperatorBase) -> AssertQuantumCircuit:
    circ = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circ.append(XGate(), [0])
    ctrl_U = decompose(U_gate.control(), basis_gates=["h", "rx", "rz", "cx"])
    circ.append(ctrl_U, range(U_gate.num_qubits + 1))
    circ.append(XGate(), [0])

    def condition(pre_state: Statevector, post_state: Statevector) -> bool:
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))

        state_0 = ((psi + (U @ psi)) / 2) ^ Zero  # type: ignore
        state_1 = ((psi - (U @ psi)) / 2) ^ One  # type: ignore
        return eq_state(post_state, state_0 + state_1)

    circ.add_condition("condition1", condition)
    return circ


def test_hadamal_test_fail_1():
    # U: OperatorBase = T
    U: OperatorBase = PrimitiveOp(Operator([[1, 0], [0, cmath.exp(1j * math.pi / 4)]]))
    U_gate: Gate = TGate()
    circ_hadamard_test = make_circ_hadamal_test_fail(U_gate, U)

    psi = Plus

    with pytest.raises(StateConditionError) as e:
        circ_hadamard_test.run(init_state=psi ^ Zero)

    assert "Condition Error occured in 'condition1'" in str(e.value)
