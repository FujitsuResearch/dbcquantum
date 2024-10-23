import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, HGate, XGate
from qiskit.opflow import One, Plus, Zero

from dbcquantum.circuit import AssertQuantumCircuit, _QiskitInstructionSet
from dbcquantum.err import DbCQuantumError, StateConditionError
from dbcquantum.utils import eq_state, partial_state

# backend: StatevectorSimulator = Aer.get_backend("statevector_simulator")
bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_QiskitInstructionSet1():
    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    state = aqc.run()
    assert eq_state(state, bell)


def test_QiskitInstructionSet2():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)
    state = aqc.run()
    assert eq_state(state, bell ^ bell)
    assert len(aqc._instruction_set_list) == 2


def test_QiskitInstructionSet3():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    aqc = AssertQuantumCircuit(2)
    aqc.append(qc.to_gate(), [0, 1])
    state = aqc.run()
    assert eq_state(state, bell)


def test_QiskitInstructionSet_condition1():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_condition(
        "condition1",
        #   |0000>
        lambda pre_state, post_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    aqc.add_condition(
        "condition2",
        lambda pre_state, post_state: eq_state(post_state, bell ^ bell),
    )
    state = aqc.run()

    assert eq_state(state, bell ^ bell)


def test_QiskitInstructionSet_condition2():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ One),  # type: ignore
    )

    aqc.add_condition(
        "condition2",
        lambda pre_state, post_state: eq_state(post_state, bell ^ bell),
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_QiskitInstructionSet_condition3():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    aqc.add_condition(
        "condition2",
        lambda pre_state, post_state: eq_state(post_state, bell ^ Zero ^ One),
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_QiskitInstructionSet_condition_partial_state1():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])
    aqc.append(CXGate(), [1, 2])
    aqc.append(XGate(), [3])

    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    def condition2(pre_state, post_state):
        post_state_0 = partial_state(post_state, [0])
        post_state_12 = partial_state(post_state, [1, 2])
        post_state_3 = partial_state(post_state, [3])

        return (
            eq_state(post_state_0, Plus)
            and eq_state(post_state_12, bell)
            and eq_state(post_state_3, One)
        )

    aqc.add_condition("condition2", condition2)
    aqc.run()


def test_QiskitInstructionSet_condition_partial_state2():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])
    aqc.append(CXGate(), [1, 2])
    aqc.append(XGate(), [3])

    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    def condition2(pre_state, post_state):
        post_state_0 = partial_state(post_state, [0])
        post_state_12 = partial_state(post_state, [1, 2])
        post_state_3 = partial_state(post_state, [3])

        return (
            eq_state(post_state_0, Plus)
            and eq_state(post_state_12, bell)
            and eq_state(post_state_3, Zero)  # False
        )

    aqc.add_condition("condition2", condition2)

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert eq_state(e.value.info.pre_state, Zero ^ Zero ^ Zero ^ Zero)  # type: ignore
    assert eq_state(e.value.info.post_state, One ^ bell ^ Plus)  # type: ignore
    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_QiskitInstructionSet_condition_partial_state3():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])
    aqc.append(CXGate(), [1, 2])
    aqc.append(XGate(), [3])

    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    def condition2(pre_state, post_state):
        post_state_0 = partial_state(post_state, [0])
        post_state_1 = partial_state(post_state, [1])
        post_state_2 = partial_state(post_state, [2])
        post_state_3 = partial_state(post_state, [3])

        return (
            eq_state(post_state_0, Plus)
            and eq_state(post_state_1, Zero)
            and eq_state(post_state_2, Zero)
            and eq_state(post_state_3, Zero)  # False
        )

    aqc.add_condition("condition2", condition2)

    with pytest.raises(DbCQuantumError) as e:
        aqc.run()

    assert "The specified qubits are entangled with the other qubits!" in str(e.value)
