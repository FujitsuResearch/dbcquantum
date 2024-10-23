import itertools
import math

from qiskit.opflow import Minus, One, Plus, Zero

from dbcquantum.utils import (
    _inverse_order_for_sort_statevector,
    _sort_statevector,
    eq_state,
    to_Statevector,
)

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_inverse_sort_statevector_1():
    order = [0, 1, 2, 3]
    s = to_Statevector(Minus ^ Plus ^ One ^ Zero)  # type: ignore
    sorted_s = _sort_statevector(s, order)
    re_sorted_s = _sort_statevector(
        sorted_s, _inverse_order_for_sort_statevector(order)
    )
    assert eq_state(sorted_s, Minus ^ Plus ^ One ^ Zero)  # type: ignore
    assert eq_state(re_sorted_s, s)  # type: ignore


def test_inverse_sort_statevector_2():
    order = [3, 0, 2, 1]
    s = to_Statevector(Minus ^ Plus ^ One ^ Zero)  # type: ignore
    sorted_s = _sort_statevector(s, order)
    re_sorted_s = _sort_statevector(
        sorted_s, _inverse_order_for_sort_statevector(order)
    )
    assert eq_state(sorted_s, One ^ Plus ^ Zero ^ Minus)  # type: ignore
    assert eq_state(re_sorted_s, s)  # type: ignore


def test_inverse_sort_statevector_3():
    order = [3, 0, 2, 1]
    s = to_Statevector(Minus ^ Plus ^ One ^ Zero)  # type: ignore
    sorted_s = _sort_statevector(s, order)
    re_sorted_s = _sort_statevector(
        sorted_s, _inverse_order_for_sort_statevector(order)
    )
    assert eq_state(sorted_s, One ^ Plus ^ Zero ^ Minus)  # type: ignore
    assert eq_state(re_sorted_s, s)  # type: ignore


def test_inverse_sort_statevector_4():
    s = to_Statevector(Minus ^ Plus ^ bell ^ One ^ Zero)  # type: ignore

    for _order in itertools.permutations(range(6)):
        order = list(_order)
        sorted_s = _sort_statevector(s, order)
        re_sorted_s = _sort_statevector(
            sorted_s, _inverse_order_for_sort_statevector(order)
        )
        assert eq_state(re_sorted_s, s)  # type: ignore
