import pytest
from qiskit.opflow import One, Zero

from dbcquantum.err import DbCQuantumError
from dbcquantum.utils import eq_state


def test_zero_one():
    assert not (eq_state(Zero, One))


def test_zero_zero():
    assert eq_state(Zero, Zero)


def test_error_dim():
    with pytest.raises(DbCQuantumError) as e:
        _ = eq_state(Zero, Zero ^ One)  # type: ignore

    assert (
        str(e.value)
        == "The states cannot be compared with because num_qubits is not the same!"
    )
