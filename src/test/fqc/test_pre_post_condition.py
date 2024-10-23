import math

import pytest
from qiskit.circuit.library.standard_gates import CXGate, HGate, XGate
from qiskit.opflow import One, Plus, X, Zero

from dbcquantum.circuit import AssertQuantumCircuit, _QiskitInstructionSet
from dbcquantum.err import StateConditionError
from dbcquantum.utils import eq_state, partial_state

# backend: StatevectorSimulator = Aer.get_backend("statevector_simulator")
bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_pre_condition1():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_pre_condition(
        "condition1",
        #   |0000>
        lambda pre_state: eq_state(
            pre_state, Zero ^ Zero ^ Zero ^ Zero  # type: ignore
        ),
    )

    state = aqc.run()

    assert eq_state(state, bell ^ bell)


def test_pre_condition2():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ One),  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_post_condition1():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_post_condition(
        "condition1",
        lambda post_state: eq_state(post_state, bell ^ bell),
    )

    state = aqc.run()

    assert eq_state(state, bell ^ bell)


def test_post_condition2():
    aqc = AssertQuantumCircuit(4)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])
    aqc.append(HGate(), [2])
    qiskit_instruction_set = _QiskitInstructionSet(4)
    qiskit_instruction_set.append(CXGate(), [2, 3])
    aqc._instruction_set_list.append(qiskit_instruction_set)

    aqc.add_post_condition(
        "condition1",
        lambda post_state: eq_state(post_state, bell ^ Zero ^ One),
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)


#


def test_AQCInstructionSet_understandable_nest_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    # swap.add_condition(
    #     "swap_condition",
    #     lambda pre_state, post_state: eq_state(
    #         partial_state(pre_state, [0]), partial_state(post_state, [1])
    #     )
    #     and eq_state(partial_state(pre_state, [1]), partial_state(post_state, [0])),
    # )

    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])
    x.add_condition(
        "x_condition",
        lambda pre_state, post_state: eq_state(post_state, pre_state.evolve(X)),
    )

    aqc1 = AssertQuantumCircuit(3)
    aqc1.append(XGate(), [0])
    aqc1.append(HGate(), [2])
    aqc1.append(make_bell, [0, 2])
    aqc1.append(XGate(), [1])

    aqc1.add_pre_condition(
        "aqc1_condition1",
        lambda pre_state: eq_state(partial_state(pre_state, [0]), One),
    )
    aqc1.add_pre_condition(
        "aqc1_condition2",
        lambda pre_state: eq_state(partial_state(pre_state, [1]), Zero),
    )
    aqc1.add_pre_condition(
        "aqc1_condition3",
        lambda pre_state: eq_state(partial_state(pre_state, [2]), Plus),
    )
    aqc1.add_post_condition(
        "aqc1_condition4",
        lambda post_state: eq_state(partial_state(post_state, [1]), One),
    )
    aqc1.add_post_condition(
        "aqc1_condition5",
        lambda post_state: eq_state(partial_state(post_state, [0, 2]), bell),
    )

    aqc = AssertQuantumCircuit(4)
    # aqc.append(x, [3]) # removed

    # assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Zero)  # type: ignore

    aqc.append(HGate(), [0])
    # assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Plus)  # type: ignore

    aqc.append(aqc1, [3, 1, 0])
    # assert eq_state(partial_state(aqc.run(), [3, 0]), bell)
    # assert eq_state(partial_state(aqc.run(), [1]), One)

    aqc.append(swap, [2, 3])
    aqc.append(swap, [1, 2])
    # assert eq_state(aqc.run(), Zero ^ One ^ bell)

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'aqc1_condition1'" in str(e.value)


def test_focus_qubit_of_add_pre_condition():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "condition1", lambda pre: eq_state(pre, Plus), focus_qubits=[0]
    )

    aqc.append(HGate(), [1])
    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc.run()


def test_focus_qubit_of_add_pre_condition_fail():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "condition1", lambda pre: eq_state(pre, One), focus_qubits=[0]
    )

    aqc.append(HGate(), [1])
    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_focus_qubit_of_add_post_condition():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc.add_post_condition(
        "condition1", lambda post: eq_state(post, bell), focus_qubits=[1, 2]
    )

    aqc.run()


def test_focus_qubit_of_add_post_condition_fail():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc.add_post_condition(
        "condition1", lambda post: eq_state(post, Zero ^ Zero), focus_qubits=[1, 2]  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)
