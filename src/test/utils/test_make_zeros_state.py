from qiskit.opflow import Zero

from dbcquantum.utils import eq_state, make_zeros_state


def test_make_zeros_state_1():
    assert eq_state(Zero, make_zeros_state(1))


def test_make_zeros_state_2():
    assert eq_state(Zero ^ Zero, make_zeros_state(2))  # type:ignore


def test_make_zeros_state_3():
    assert eq_state(Zero ^ Zero ^ Zero, make_zeros_state(3))  # type:ignore


def test_make_zeros_state_4():
    assert eq_state(Zero ^ Zero ^ Zero ^ Zero, make_zeros_state(4))  # type:ignore
