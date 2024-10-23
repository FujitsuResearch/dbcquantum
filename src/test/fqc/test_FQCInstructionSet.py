import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, HGate, XGate, YGate, ZGate
from qiskit.opflow import One, Plus, X, Zero
from qiskit.quantum_info import Statevector

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import DbCQuantumError, StateConditionError
from dbcquantum.utils import eq_state, partial_state

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_AQCInstructionSet_1():
    aqc1 = AssertQuantumCircuit(1)
    aqc2 = AssertQuantumCircuit(2)

    aqc1.append(HGate(), [0])
    aqc2.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(aqc1, [0])
    aqc.append(aqc2, [0, 1])

    state = aqc.run()
    assert eq_state(state, bell)


def test_AQCInstructionSet_2():
    aqc1 = AssertQuantumCircuit(1)
    aqc2 = AssertQuantumCircuit(2)

    aqc1.append(HGate(), [0])
    aqc2.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(aqc1, [1])
    aqc.append(aqc2, [1, 0])

    state = aqc.run()
    assert eq_state(state, bell)


def test_AQCInstructionSet_3():
    aqc1 = AssertQuantumCircuit(1)
    aqc2 = AssertQuantumCircuit(2)

    aqc1.append(HGate(), [0])
    aqc2.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(aqc1, [1])
    aqc.append(aqc2, [0, 1])

    state = aqc.run()
    assert eq_state(state, Plus ^ Zero)  # type: ignore


def test_AQCInstructionSet_understandable_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition",
        lambda pre_state, post_state: eq_state(
            partial_state(pre_state, [0]), partial_state(post_state, [1])
        )
        and eq_state(partial_state(pre_state, [1]), partial_state(post_state, [0])),
    )

    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])
    x.add_condition(
        "x_condition",
        lambda pre_state, post_state: eq_state(post_state, pre_state.evolve(X)),
    )

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(HGate(), [2])
    aqc.append(x, [3])
    aqc.append(swap, [2, 3])

    state = aqc.run()
    assert eq_state(state, Plus ^ One ^ bell)  # type: ignore


def test_AQCInstructionSet_limitation_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition",
        lambda pre_state, post_state: eq_state(
            partial_state(pre_state, [0]), partial_state(post_state, [1])
        )
        and eq_state(partial_state(pre_state, [1]), partial_state(post_state, [0])),
    )

    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])
    x.add_condition(
        "x_condition",
        lambda pre_state, post_state: eq_state(post_state, pre_state.evolve(X)),
    )

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(HGate(), [2])
    aqc.append(x, [3])
    aqc.append(swap, [2, 3])

    aqc.append(swap, [1, 3])
    aqc.append(swap, [1, 3])

    # state = aqc.run()
    # assert eq_state(state, Plus ^ One ^ bell)  # type: ignore

    with pytest.raises(DbCQuantumError) as e:
        _ = aqc.run()

    assert "The specified qubits are entangled with the other qubits!" in str(e.value)


def test_AQCInstructionSet_understandable_nest_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
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

    aqc1.add_condition(
        "aqc1_condition1",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [0]), One),
    )
    aqc1.add_condition(
        "aqc1_condition2",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [1]), Zero),
    )
    aqc1.add_condition(
        "aqc1_condition3",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [2]), Plus),
    )
    aqc1.add_condition(
        "aqc1_condition4",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [1]), One),
    )
    aqc1.add_condition(
        "aqc1_condition5",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [0, 2]), bell),
    )

    aqc = AssertQuantumCircuit(4)
    aqc.append(x, [3])
    assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Zero)  # type: ignore

    aqc.append(HGate(), [0])
    assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Plus)  # type: ignore

    aqc.append(aqc1, [3, 1, 0])
    assert eq_state(partial_state(aqc.run(), [3, 0]), bell)
    assert eq_state(partial_state(aqc.run(), [1]), One)

    aqc.append(swap, [2, 3])
    aqc.append(swap, [1, 2])
    assert eq_state(aqc.run(), Zero ^ One ^ bell)


#


#


def test_AQCInstructionSet_understandable_nest_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
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

    aqc1.add_condition(
        "aqc1_condition1",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [0]), One),
    )
    aqc1.add_condition(
        "aqc1_condition2",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [1]), Zero),
    )
    aqc1.add_condition(
        "aqc1_condition3",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [2]), Plus),
    )
    aqc1.add_condition(
        "aqc1_condition4",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [1]), One),
    )
    aqc1.add_condition(
        "aqc1_condition5",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [0, 2]), bell),
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

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)


def test_nest_AQCInstructionSet_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.append(XGate(), [0])
    aqc1.append(ZGate(), [1])

    qc1 = QuantumCircuit(2)
    qc1.append(CXGate(), [0, 1])
    qc1.append(XGate(), [0])
    qc1.append(ZGate(), [1])

    aqc2 = AssertQuantumCircuit(3)
    aqc2.append(HGate(), [0])
    aqc2.append(HGate(), [2])
    aqc2.append(ZGate(), [1])
    aqc2.append(XGate(), [1])
    aqc2.append(CXGate(), [2, 1])

    qc2 = QuantumCircuit(3)
    qc2.append(HGate(), [0])
    qc2.append(HGate(), [2])
    qc2.append(ZGate(), [1])
    qc2.append(XGate(), [1])
    qc2.append(CXGate(), [2, 1])

    aqc3 = AssertQuantumCircuit(4)
    aqc3.append(HGate(), [0])
    aqc3.append(aqc2, [3, 0, 2])
    aqc3.append(ZGate(), [1])
    aqc3.append(HGate(), [2])
    aqc3.append(CXGate(), [3, 2])

    qc3 = QuantumCircuit(4)
    qc3.append(HGate(), [0])
    qc3.append(qc2.to_instruction(), [3, 0, 2])
    qc3.append(ZGate(), [1])
    qc3.append(HGate(), [2])
    qc3.append(CXGate(), [3, 2])

    aqc = AssertQuantumCircuit(5)
    aqc.append(aqc1, [1, 4])
    aqc.append(YGate(), [1])
    aqc.append(HGate(), [2])
    aqc.append(XGate(), [0])
    aqc.append(CXGate(), [4, 2])
    aqc.append(CXGate(), [3, 1])
    aqc.append(aqc3, [4, 1, 2, 0])

    qc = QuantumCircuit(5)
    qc.append(qc1.to_instruction(), [1, 4])
    qc.append(YGate(), [1])
    qc.append(HGate(), [2])
    qc.append(XGate(), [0])
    qc.append(CXGate(), [4, 2])
    qc.append(CXGate(), [3, 1])
    qc.append(qc3.to_instruction(), [4, 1, 2, 0])

    state = aqc.run()
    desired_state = (
        Statevector((Zero ^ Zero ^ Zero ^ Zero ^ Zero).to_matrix())
    ).evolve(qc)

    assert eq_state(state, desired_state)


def test_nest_AQCInstructionSet_condiiton_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.append(XGate(), [0])
    aqc1.append(ZGate(), [1])

    qc1 = QuantumCircuit(2)
    qc1.append(CXGate(), [0, 1])
    qc1.append(XGate(), [0])
    qc1.append(ZGate(), [1])

    aqc1.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc1), post_state),
    )

    aqc2 = AssertQuantumCircuit(3)
    aqc2.append(HGate(), [0])
    aqc2.append(HGate(), [2])
    aqc2.append(ZGate(), [1])
    aqc2.append(XGate(), [1])
    aqc2.append(CXGate(), [2, 1])

    qc2 = QuantumCircuit(3)
    qc2.append(HGate(), [0])
    qc2.append(HGate(), [2])
    qc2.append(ZGate(), [1])
    qc2.append(XGate(), [1])
    qc2.append(CXGate(), [2, 1])

    aqc2.add_condition(
        "condition2",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc2), post_state),
    )

    aqc3 = AssertQuantumCircuit(4)
    aqc3.append(HGate(), [0])
    aqc3.append(aqc2, [3, 0, 2])
    aqc3.append(ZGate(), [1])
    aqc3.append(HGate(), [2])
    aqc3.append(CXGate(), [3, 2])

    qc3 = QuantumCircuit(4)
    qc3.append(HGate(), [0])
    qc3.append(qc2.to_instruction(), [3, 0, 2])
    qc3.append(ZGate(), [1])
    qc3.append(HGate(), [2])
    qc3.append(CXGate(), [3, 2])

    aqc3.add_condition(
        "condition3",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc3), post_state),
    )

    aqc = AssertQuantumCircuit(5)
    aqc.append(aqc1, [1, 4])
    aqc.append(YGate(), [1])
    aqc.append(HGate(), [2])
    aqc.append(XGate(), [0])
    aqc.append(CXGate(), [4, 2])
    aqc.append(CXGate(), [3, 1])
    aqc.append(aqc3, [4, 1, 2, 0])

    qc = QuantumCircuit(5)
    qc.append(qc1.to_instruction(), [1, 4])
    qc.append(YGate(), [1])
    qc.append(HGate(), [2])
    qc.append(XGate(), [0])
    qc.append(CXGate(), [4, 2])
    qc.append(CXGate(), [3, 1])
    qc.append(qc3.to_instruction(), [4, 1, 2, 0])

    aqc.add_condition(
        "condition_whole",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc), post_state),
    )

    state = aqc.run()
    desired_state = (
        Statevector((Zero ^ Zero ^ Zero ^ Zero ^ Zero).to_matrix())
    ).evolve(qc)

    assert eq_state(state, desired_state)


def test_nest_AQCInstructionSet_condiiton_2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.append(XGate(), [0])
    aqc1.append(ZGate(), [1])

    qc1 = QuantumCircuit(2)
    qc1.append(CXGate(), [0, 1])
    qc1.append(XGate(), [0])
    qc1.append(ZGate(), [1])

    aqc1.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc1), post_state),
    )

    aqc2 = AssertQuantumCircuit(3)
    aqc2.append(HGate(), [0])
    aqc2.append(HGate(), [2])
    aqc2.append(ZGate(), [1])
    aqc2.append(XGate(), [1])
    aqc2.append(CXGate(), [2, 1])

    qc2 = QuantumCircuit(3)
    qc2.append(HGate(), [0])
    qc2.append(HGate(), [2])
    qc2.append(ZGate(), [1])
    qc2.append(XGate(), [1])
    qc2.append(CXGate(), [2, 1])

    # False
    aqc2.add_condition(
        "condition2",
        lambda pre_state, post_state: eq_state(Zero ^ Zero ^ Zero, post_state),
    )

    aqc3 = AssertQuantumCircuit(4)
    aqc3.append(HGate(), [0])
    aqc3.append(aqc2, [3, 0, 2])
    aqc3.append(ZGate(), [1])
    aqc3.append(HGate(), [2])
    aqc3.append(CXGate(), [3, 2])

    qc3 = QuantumCircuit(4)
    qc3.append(HGate(), [0])
    qc3.append(qc2.to_instruction(), [3, 0, 2])
    qc3.append(ZGate(), [1])
    qc3.append(HGate(), [2])
    qc3.append(CXGate(), [3, 2])

    aqc3.add_condition(
        "condition3",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc3), post_state),
    )

    aqc = AssertQuantumCircuit(5)
    aqc.append(aqc1, [1, 4])
    aqc.append(YGate(), [1])
    aqc.append(HGate(), [2])
    aqc.append(XGate(), [0])
    aqc.append(CXGate(), [4, 2])
    aqc.append(CXGate(), [3, 1])
    aqc.append(aqc3, [4, 1, 2, 0])

    qc = QuantumCircuit(5)
    qc.append(qc1.to_instruction(), [1, 4])
    qc.append(YGate(), [1])
    qc.append(HGate(), [2])
    qc.append(XGate(), [0])
    qc.append(CXGate(), [4, 2])
    qc.append(CXGate(), [3, 1])
    qc.append(qc3.to_instruction(), [4, 1, 2, 0])

    aqc.add_condition(
        "condition_whole",
        lambda pre_state, post_state: eq_state(pre_state.evolve(qc), post_state),
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_focus_qubit_of_add_condition():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc.add_condition(
        "condition1", lambda pre, post: eq_state(post, bell), focus_qubits=[1, 2]
    )

    aqc.run()


def test_focus_qubit_of_add_condition_fail():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc.add_condition(
        "condition1", lambda pre, post: eq_state(post, Zero ^ Zero), focus_qubits=[1, 2]  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'condition1'" in str(e.value)
