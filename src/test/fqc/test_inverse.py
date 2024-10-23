import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, HGate, RXGate, RYGate
from qiskit.opflow import One, Plus, Zero

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import StateConditionError
from dbcquantum.utils import eq_state, to_Statevector

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_inverse_basic1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(RXGate(math.pi * 5 / 6), [0])
    aqc1.append(RYGate(math.pi * 5 / 6), [1])

    aqc2 = AssertQuantumCircuit(2)
    aqc2.append(RXGate(math.pi * 1 / 12), [0])
    aqc2.append(RYGate(math.pi * 1 / 12), [1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(aqc1, [0, 1])
    aqc.append(aqc2, [1, 0])

    qc = QuantumCircuit(2)
    qc.rx(math.pi * 5 / 6, 0)
    qc.ry(math.pi * 5 / 6, 1)
    qc.rx(math.pi * 1 / 12, 1)
    qc.ry(math.pi * 1 / 12, 0)

    pre_state = to_Statevector(One ^ Plus)  # type: ignore
    post_state = pre_state.evolve(qc)

    assert eq_state(pre_state, aqc.inverse().run(post_state))
    assert eq_state(post_state, aqc.run(pre_state))


def test_inverse_1():
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

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(make_bell, [2, 3])

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ Zero),  # type: ignore
    )

    aqc.add_post_condition(
        "condition2", lambda post_state: eq_state(post_state, bell ^ bell)
    )

    aqc_inverse = aqc.inverse()
    state = aqc_inverse.run(bell ^ bell)
    assert eq_state(state, Zero ^ Zero ^ Zero ^ Zero)  # type: ignore


def test_inverse_2():
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

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(make_bell, [2, 3])

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ Zero),  # type: ignore
    )

    aqc.add_post_condition(
        "condition2", lambda post_state: eq_state(post_state, bell ^ bell)
    )

    aqc_inverse = aqc.inverse()

    with pytest.raises(StateConditionError) as e:
        aqc_inverse.run(bell ^ Zero ^ Zero)

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_inverse_3():
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

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(make_bell, [2, 3])

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ One),  # type: ignore
    )

    aqc.add_post_condition(
        "condition2", lambda post_state: eq_state(post_state, bell ^ bell)
    )

    aqc_inverse = aqc.inverse()

    with pytest.raises(StateConditionError) as e:
        aqc_inverse.run(bell ^ bell)

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_inverse_4():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    # make_bell.add_condition(
    #     "bell_input_|00>",
    #     lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    # )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(make_bell, [2, 3])

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ Zero),  # type: ignore
    )

    aqc.add_post_condition(
        "condition2", lambda post_state: eq_state(post_state, bell ^ Zero ^ Zero)
    )

    aqc_inverse = aqc.inverse()

    with pytest.raises(StateConditionError) as e:
        aqc_inverse.run(bell ^ Zero ^ Zero)

    assert "Condition Error occured in 'bell_output_bell'" in str(e.value)


def test_inverse_inverse_1():
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

    aqc = AssertQuantumCircuit(4)
    aqc.append(make_bell, [0, 1])
    aqc.append(make_bell, [2, 3])

    aqc.add_pre_condition(
        "condition1",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero ^ Zero ^ Zero),  # type: ignore
    )

    aqc.add_post_condition(
        "condition2", lambda post_state: eq_state(post_state, bell ^ bell)
    )

    aqc_inverse = aqc.inverse()
    state = aqc_inverse.run(bell ^ bell)
    assert eq_state(state, Zero ^ Zero ^ Zero ^ Zero)  # type: ignore

    aqc_inverse_inverse = aqc_inverse.inverse()
    state = aqc_inverse_inverse.run()
    assert eq_state(state, bell ^ bell)  # type: ignore
