import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, HGate, RXGate, XGate, ZGate
from qiskit.opflow import One, Zero

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import StateConditionError
from dbcquantum.utils import eq_state

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_eq_circuit1():
    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(HGate(), [0])
    x.append(ZGate(), [0])
    x.append(HGate(), [0])

    x_circuit = QuantumCircuit(1)
    x_circuit.x(0)

    x.add_eq_circuit("X = HZH", x_circuit)

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

    aqc = AssertQuantumCircuit(2)
    aqc.append(make_bell, [0, 1])
    aqc.append(x, [0])

    state = aqc.run()
    eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    state = aqc.run()
    eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    state = aqc.run()
    eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore


def test_eq_circuit1_fail():
    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(HGate(), [0])
    x.append(ZGate(), [0])
    x.append(HGate(), [0])

    h_circuit = QuantumCircuit(1)
    h_circuit.h(0)

    x.add_eq_circuit("X = HZH", h_circuit)

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

    aqc = AssertQuantumCircuit(2)
    aqc.append(make_bell, [0, 1])
    aqc.append(x, [0])

    with pytest.raises(StateConditionError) as e:

        state = aqc.run()
        eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    assert "Condition Error occured in 'X = HZH'" in str(e.value)

    with pytest.raises(StateConditionError) as e:

        state = aqc.run()
        eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    assert "Condition Error occured in 'X = HZH'" in str(e.value)

    with pytest.raises(StateConditionError) as e:

        state = aqc.inverse().inverse().run()
        eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    assert "Condition Error occured in 'X = HZH'" in str(e.value)

    with pytest.raises(StateConditionError) as e:

        state = aqc.inverse().inverse().run()
        eq_state(state, 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)))  # type: ignore

    assert "Condition Error occured in 'X = HZH'" in str(e.value)


def test_eq_circuit1_inverse_complex():
    aqc1 = AssertQuantumCircuit(1)
    aqc1.append(HGate(), [0])
    aqc1.append(XGate(), [0])
    aqc1.append(HGate(), [0])
    aqc1.append(RXGate(math.pi / 6), [0])

    eq_circuit = QuantumCircuit(1)
    eq_circuit.h(0)
    eq_circuit.x(0)
    eq_circuit.h(0)
    eq_circuit.rx(math.pi / 6, 0)

    aqc1.add_eq_circuit("eq_condition1", eq_circuit)

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

    aqc = AssertQuantumCircuit(2)
    aqc.append(make_bell, [0, 1])
    aqc.append(aqc1, [0])

    aqc.inverse().run(aqc.run())
    aqc.inverse().run(aqc.run())
