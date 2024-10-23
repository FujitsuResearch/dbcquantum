import math

import pytest
from qiskit.opflow import Minus, One, Plus, Zero
from qiskit.quantum_info import Statevector

from dbcquantum.err import DbCQuantumError
from dbcquantum.utils import (
    _merge_statevector,
    _partial_state_traceout,
    eq_state,
    partial_state,
    to_Statevector,
)

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_merge_statevector1():
    state = to_Statevector(bell ^ Minus ^ Plus ^ One ^ Zero)
    state_partial = partial_state(state, [0, 1, 2, 3, 4, 5])
    assert eq_state(state, state_partial)

    state_left = partial_state(state, [])

    state_merge = _merge_statevector(state_partial, state_left, [0, 1, 2, 3, 4, 5])
    assert eq_state(state, state_merge)


def test_merge_statevector2():
    qubits = [3, 0]

    state = to_Statevector(bell ^ Minus ^ Plus ^ One ^ Zero)
    state_partial = partial_state(state, qubits)
    assert eq_state(Zero ^ Minus, state_partial)  # type: ignore

    state_left = _partial_state_traceout(state, qubits)
    assert eq_state(bell ^ Plus ^ One, state_left)

    state_merge = _merge_statevector(state_partial, state_left, qubits)
    assert eq_state(state, state_merge)


def test_merge_statevector3():
    qubits = [3, 0]

    state: Statevector = Statevector((Minus ^ Plus ^ One ^ Zero).to_matrix())  # type: ignore
    state_partial = partial_state(state, qubits)
    assert eq_state(Zero ^ Minus, state_partial)  # type: ignore

    state_left = _partial_state_traceout(state, qubits)
    assert eq_state(Plus ^ One, state_left)  # type: ignore

    state_merge = _merge_statevector(state_partial, state_left, qubits)
    assert eq_state(state, state_merge)


def test_merge_statevector_error_dim():
    qubits = [3, 0]

    state: Statevector = Statevector((Minus ^ Plus ^ One ^ Zero).to_matrix())  # type: ignore
    state_partial = partial_state(state, qubits)
    state_left = _partial_state_traceout(state, qubits)

    with pytest.raises(DbCQuantumError) as e:
        _ = _merge_statevector(state_partial, state_left, [3, 0, 2])

    assert (
        str(e.value)
        == "The size of focus_qubits_state and focus_qubits is inconsistent!"
    )
