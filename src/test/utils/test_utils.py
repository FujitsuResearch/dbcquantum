import math

import pytest
from qiskit.opflow import Minus, One, Plus, Zero
from qiskit.quantum_info import Statevector

from dbcquantum.err import DbCQuantumError
from dbcquantum.utils import (
    _get_order,
    _sort_statevector,
    eq_state,
    partial_state,
    to_Statevector,
)


def test_sort_statevector_1():
    vec = Statevector((Plus ^ One ^ Zero).to_matrix())
    assert eq_state(_sort_statevector(vec, [0, 1, 2]), Plus ^ One ^ Zero)  # type: ignore


def test_sort_statevector_2():
    vec = Statevector((Plus ^ One ^ Zero).to_matrix())
    assert eq_state(_sort_statevector(vec, [2, 0, 1]), One ^ Zero ^ Plus)  # type: ignore


def test_sort_statevector_3():
    vec = Statevector((Minus ^ Plus ^ One ^ Zero).to_matrix())  # type: ignore
    assert eq_state(_sort_statevector(vec, [2, 3, 0, 1]), One ^ Zero ^ Minus ^ Plus)  # type: ignore


def test_sort_statevector_error_dim():
    vec = Statevector((Plus ^ One ^ Zero).to_matrix())

    with pytest.raises(DbCQuantumError) as e:
        eq_state(_sort_statevector(vec, [1, 0]), One ^ Zero ^ Plus)  # type: ignore

    assert "The size of vec and mapping is inconsistent!" in str(e.value)


bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_partial_state1():
    state = to_Statevector(Zero ^ bell ^ One)
    assert eq_state(partial_state(state, [1, 2]), bell)


def test_partial_state2():
    state = to_Statevector(Zero ^ bell ^ One)
    assert eq_state(partial_state(state, [0, 3]), Zero ^ One)  # type: ignore


def test_partial_state3():
    state = to_Statevector(Zero ^ bell ^ One)

    with pytest.raises(DbCQuantumError) as e:
        _ = partial_state(state, [1])

    assert "The specified qubits are entangled with the other qubits!" in str(e.value)


def test_partial_state4():
    state = to_Statevector(Zero ^ bell ^ One)
    assert eq_state(partial_state(state, [3, 0]), One ^ Zero)  # type: ignore


def test_get_order1():
    assert _get_order([2, 5, 1, 6]) == [1, 2, 0, 3]


def test_get_order2():
    assert _get_order([]) == []


def test_get_order3():
    assert _get_order([7]) == [0]
