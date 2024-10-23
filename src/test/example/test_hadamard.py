import cmath
import math
import typing

import numpy as np
import numpy.linalg as LA
import pytest
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, TGate, XGate
from qiskit.opflow import H, One, OperatorBase, StateFn, T, X, Zero
from qiskit.quantum_info import Statevector
from qiskit.result.counts import Counts
from qiskit.result.result import Result

import dbcquantum.err as err
from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.utils import eq_state, partial_state

seed_simulator = 10

# https://dojo.qulacs.org/ja/latest/notebooks/2.2_Hadamard_test.html
# https://utokyo-icepp.github.io/qc-workbook/quantum_computation.html


def make_circuit_hadamal_test(U_gate: Gate, U: OperatorBase) -> AssertQuantumCircuit:
    controlled_U_gate: Gate = U_gate.control(num_ctrl_qubits=1)

    circuit = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circuit.append(HGate(), [0])
    circuit.append(controlled_U_gate, range(U_gate.num_qubits + 1))
    circuit.append(HGate(), [0])

    circuit.add_pre_condition(
        "The first qubit's inital state must be |0>",
        lambda pre_state: eq_state(partial_state(pre_state, [0]), Zero),
    )

    def condition(pre_state: Statevector, post_state: Statevector) -> bool:
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))

        state_0 = ((psi + (U @ psi)) / 2) ^ Zero  # type: ignore
        state_1 = ((psi - (U @ psi)) / 2) ^ One  # type: ignore
        state = state_0 + state_1

        return eq_state(post_state, state)

    circuit.add_condition("hadamal_test_condition", condition)
    return circuit


def make_circuit_hadamal_test_eigen(
    U_gate: Gate, U: OperatorBase
) -> AssertQuantumCircuit[float]:
    circuit: AssertQuantumCircuit[float] = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circuit.append(make_circuit_hadamal_test(U_gate, U), range(U_gate.num_qubits + 1))

    def is_eigen(pre_state: Statevector, l: float) -> bool:
        eigen = cmath.exp(1j * l)
        x = (partial_state(pre_state, [1])).data
        return bool(np.all(np.isclose(eigen * x, np.dot(U.to_matrix(), x), atol=1e-08)))

    def condition(pre_state: Statevector, post_state: Statevector, l: float) -> bool:
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))
        eigen = cmath.exp(1j * l)

        state_q0 = (((1 + eigen)) / 2) * Zero + (((1 - eigen)) / 2) * One  # type: ignore

        state = psi ^ state_q0
        return eq_state(post_state, state)

    circuit.add_pre_condition_use_param("is_eigen", is_eigen)
    circuit.add_condition_use_param("condition_eigen_lambda", condition)
    return circuit


def test_hadamal_1():
    U: OperatorBase = H
    U_gate: Gate = HGate()
    circuit = make_circuit_hadamal_test(U_gate, U)
    circuit.run()


def test_hadamal_2():
    U: OperatorBase = H
    U_gate: Gate = XGate()
    circuit = make_circuit_hadamal_test(U_gate, U)

    with pytest.raises(err.StateConditionError) as e:
        circuit.run()
    assert "Condition Error occured in 'hadamal_test_condition'" in str(e.value)


def test_hadamal_3():
    U: OperatorBase = H
    U_gate: Gate = HGate()
    circuit = make_circuit_hadamal_test(U_gate, U)

    with pytest.raises(err.StateConditionError) as e:
        circuit.run(Zero ^ One)
    assert (
        "Condition Error occured in 'The first qubit's inital state must be |0>'"
        in str(e.value)
    )


def test_hadamal_eigen_1():
    U: OperatorBase = H
    U_gate: Gate = HGate()

    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        # give lambda as a runtime param
        circuit.run(init_state=eigenstate ^ Zero, param=l)


def test_hadamal_eigen_2():
    U: OperatorBase = H
    U_gate: Gate = HGate()

    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(X.to_matrix())  # mistake!

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        with pytest.raises(err.StateConditionError) as e:
            # give lambda as a runtime param
            circuit.run(init_state=eigenstate ^ Zero, param=l)
        assert "Condition Error occured in 'is_eigen'" in str(e.value)


def test_hadamal_eigen_3():
    U: OperatorBase = H
    U_gate: Gate = HGate()

    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        with pytest.raises(err.StateConditionError) as e:
            # give lambda as a runtime param
            circuit.run(init_state=eigenstate ^ One, param=l)  # mistake!
        assert (
            "Condition Error occured in 'The first qubit's inital state must be |0>'"
            in str(e.value)
        )


def mistake_make_circuit_hadamal_test_eigen(
    U_gate: Gate, U: OperatorBase
) -> AssertQuantumCircuit[float]:
    circuit: AssertQuantumCircuit[float] = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circuit.append(make_circuit_hadamal_test(U_gate, U), range(U_gate.num_qubits + 1))

    # mistake!
    circuit.append(XGate(), [0])

    def is_eigen(pre_state: Statevector, l: float) -> bool:
        eigen = cmath.exp(1j * l)
        x = (partial_state(pre_state, [1])).data
        return bool(np.all(np.isclose(eigen * x, np.dot(U.to_matrix(), x), atol=1e-08)))

    def condition(pre_state: Statevector, post_state: Statevector, l: float) -> bool:
        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))
        eigen = cmath.exp(1j * l)

        state_q0 = (((1 + eigen)) / 2) * Zero + (((1 - eigen)) / 2) * One  # type: ignore

        state = psi ^ state_q0
        return eq_state(post_state, state)

    circuit.add_pre_condition_use_param("is_eigen", is_eigen)
    circuit.add_condition_use_param("condition_eigen_lambda", condition)
    return circuit


def test_hadamal_eigen_mistake_circuit_1():
    U: OperatorBase = H
    U_gate: Gate = HGate()

    # mistake!
    circuit = mistake_make_circuit_hadamal_test_eigen(U_gate, U)

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        with pytest.raises(err.StateConditionError) as e:
            # give lambda as a runtime param
            circuit.run(init_state=eigenstate ^ Zero, param=l)
        assert "Condition Error occured in 'condition_eigen_lambda'" in str(e.value)


def test_hadamal_measure_1():
    U: OperatorBase = T
    U_gate: Gate = TGate()
    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    def calc_eigenvalue(result: Result) -> float:
        counts: Counts = typing.cast(Counts, result.get_counts())
        c_zero: int = counts["0"] if counts.get("0") is not None else 0
        c_one: int = counts["1"] if counts.get("1") is not None else 0

        p0: float = c_zero / (c_zero + c_one)
        # (1 + cos l) / 2 = p0
        # 1 + cos l = 2 * p0
        # cos l = 2 * p0 - 1
        estimated_l: float = math.acos(2 * p0 - 1)

        return estimated_l

    aqc_measure: AQCMeasure[float, tuple[float, float]] = AQCMeasure(
        circuit,
        postprocess=calc_eigenvalue,
        qubit=[0],
        param_converter=(lambda l_tol: l_tol[0]),
    )

    def eigenvalue_estimation_condition(
        pre_measure_state: Statevector,
        result: Result,
        estimated_l: float,
        param: tuple[float, float],
    ):
        actual_l, abs_tol = param
        return cmath.isclose(actual_l, estimated_l, abs_tol=abs_tol)

    aqc_measure.add_condition_use_param(
        "eigenvalue estimation", eigenvalue_estimation_condition
    )

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l: float = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        # 0 <= l <= pi
        aqc_measure.run(
            shots=100_000,
            init_state=eigenstate ^ Zero,
            param=(l, 1e-02),
            seed_simulator=seed_simulator,
        )


def test_hadamal_measure_2():
    U: OperatorBase = T
    U_gate: Gate = TGate()
    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    def calc_eigenvalue(result: Result) -> float:
        counts: Counts = typing.cast(Counts, result.get_counts())
        c_zero: int = counts["0"] if counts.get("0") is not None else 0
        c_one: int = counts["1"] if counts.get("1") is not None else 0

        p0: float = c_zero / (c_zero + c_one)
        # (1 + cos l) / 2 = p0
        # 1 + cos l = 2 * p0
        # cos l = 2 * p0 - 1
        estimated_l: float = math.acos(2 * p0 - 1)

        return estimated_l

    aqc_measure: AQCMeasure[float, tuple[float, float]] = AQCMeasure(
        circuit,
        postprocess=calc_eigenvalue,
        qubit=[0],
        param_converter=(lambda l_tol: l_tol[0]),
    )

    def eigenvalue_estimation_condition(
        pre_measure_state: Statevector,
        result: Result,
        estimated_l: float,
        param: tuple[float, float],
    ):
        actual_l, abs_tol = param
        return cmath.isclose(actual_l, estimated_l, abs_tol=abs_tol)

    aqc_measure.add_condition_use_param(
        "eigenvalue estimation", eigenvalue_estimation_condition
    )

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    with pytest.raises(err.MeasureConditionError) as e:
        for i in range(len(eigen[0])):
            eigenstate = StateFn(eigen[1][:, i])
            eigenvalue = eigen[0][i]

            # e^(i*lambda) = eigenvalue
            # lambda = log(eigenvalue) / i
            l: float = (cmath.log(eigenvalue) / 1j).real

            # check (eigenvalue / 1j) is a real number
            assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

            # 0 <= l <= pi

            aqc_measure.run(
                shots=100,
                init_state=eigenstate ^ Zero,
                param=(l, 1e-02),
                seed_simulator=seed_simulator,
            )

    assert "Condition Error occured in 'eigenvalue estimation'" in str(e.value)


def test_hadamal_measure_3():
    U: OperatorBase = T
    U_gate: Gate = TGate()
    circuit = make_circuit_hadamal_test_eigen(U_gate, U)

    def calc_eigenvalue(result: Result) -> float:
        counts: Counts = typing.cast(Counts, result.get_counts())
        c_zero: int = counts["0"] if counts.get("0") is not None else 0
        c_one: int = counts["1"] if counts.get("1") is not None else 0

        p0: float = c_zero / (c_zero + c_one)
        # (1 + cos l) / 2 = p0
        # 1 + cos l = 2 * p0
        # cos l = 2 * p0 - 1

        # mistake! acos -> asin
        estimated_l: float = math.asin(2 * p0 - 1)

        return estimated_l

    aqc_measure: AQCMeasure[float, tuple[float, float]] = AQCMeasure(
        circuit,
        postprocess=calc_eigenvalue,
        qubit=[0],
        param_converter=(lambda l_tol: l_tol[0]),
    )

    def eigenvalue_estimation_condition(
        pre_measure_state: Statevector,
        result: Result,
        estimated_l: float,
        param: tuple[float, float],
    ):
        actual_l, abs_tol = param
        return cmath.isclose(actual_l, estimated_l, abs_tol=abs_tol)

    aqc_measure.add_condition_use_param(
        "eigenvalue estimation", eigenvalue_estimation_condition
    )

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    with pytest.raises(err.MeasureConditionError) as e:
        for i in range(len(eigen[0])):
            eigenstate = StateFn(eigen[1][:, i])
            eigenvalue = eigen[0][i]

            # e^(i*lambda) = eigenvalue
            # lambda = log(eigenvalue) / i
            l: float = (cmath.log(eigenvalue) / 1j).real

            # check (eigenvalue / 1j) is a real number
            assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

            # 0 <= l <= pi

            aqc_measure.run(
                shots=100_000,
                init_state=eigenstate ^ Zero,
                param=(l, 1e-02),
                seed_simulator=seed_simulator,
            )

    assert "Condition Error occured in 'eigenvalue estimation'" in str(e.value)


def test_hadamal_measure_simple_1():
    U: OperatorBase = T
    U_gate: Gate = TGate()
    circuit = make_circuit_hadamal_test(U_gate, U)

    def calc_eigenvalue(result: Result) -> float:
        counts: Counts = typing.cast(Counts, result.get_counts())
        c_zero: int = counts["0"] if counts.get("0") is not None else 0
        c_one: int = counts["1"] if counts.get("1") is not None else 0

        p0: float = c_zero / (c_zero + c_one)
        # (1 + cos l) / 2 = p0
        # 1 + cos l = 2 * p0
        # cos l = 2 * p0 - 1
        estimated_l: float = math.acos(2 * p0 - 1)

        return estimated_l

    aqc_measure: AQCMeasure[float, tuple[float, float]] = AQCMeasure(
        circuit, postprocess=calc_eigenvalue, qubit=[0]
    )

    def eigenvalue_estimation_condition(
        pre_measure_state: Statevector,
        result: Result,
        estimated_l: float,
        param: tuple[float, float],
    ):
        actual_l, abs_tol = param
        return cmath.isclose(actual_l, estimated_l, abs_tol=abs_tol)

    aqc_measure.add_condition_use_param(
        "eigenvalue estimation", eigenvalue_estimation_condition
    )

    # calculate eigenvalue and eigenstate
    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        # e^(i*lambda) = eigenvalue
        # lambda = log(eigenvalue) / i
        l: float = (cmath.log(eigenvalue) / 1j).real

        # check (eigenvalue / 1j) is a real number
        assert cmath.isclose(l, cmath.log(eigenvalue) / 1j, abs_tol=1e-08)

        # 0 <= l <= pi

        aqc_measure.run(
            shots=100_000,
            init_state=eigenstate ^ Zero,
            param=(l, 1e-02),
            seed_simulator=seed_simulator,
        )
