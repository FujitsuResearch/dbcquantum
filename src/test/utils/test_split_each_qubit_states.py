from qiskit.opflow import Minus, One, Plus, Zero

from dbcquantum.utils import eq_state, split_each_qubit_states, to_Statevector


def test_split_each_qubit_states_1():
    state = to_Statevector(Minus ^ Plus ^ One ^ Zero)  # type:ignore
    states = split_each_qubit_states(state)
    assert eq_state(states[0], Zero)
    assert eq_state(states[1], One)
    assert eq_state(states[2], Plus)
    assert eq_state(states[3], Minus)
