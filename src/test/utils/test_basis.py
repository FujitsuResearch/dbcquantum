import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils


def test_is_basis_true():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j])),
    ]

    assert utils.is_basis(basis)

    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 1 + 6j, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 1 + 7j, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 1 + 8j, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 0, 1 + 9j])),
    ]

    assert utils.is_basis(basis)


def test_is_basis_false():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0])),
        Statevector(np.array([0, 0, 2 + 8j, 0])),
    ]

    assert not utils.is_basis(basis)

    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 1 + 6j, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 1 + 7j, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 1 + 8j, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 0, 0])),
    ]

    assert not utils.is_basis(basis)


def test_is_basis_fail_1():
    basis = []

    with pytest.raises(RuntimeError) as e:
        utils.is_basis(basis)

    assert "basis is empty." in str(e.value)


def test_is_basis_fail_2():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 1 + 6j, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 1 + 7j, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 1 + 8j, 0])),
        Statevector(np.array([0, 0, 0, 0])),
    ]

    with pytest.raises(RuntimeError) as e:
        utils.is_basis(basis)

    assert "inconsistent" in str(e.value)


def test_is_basis_fail_3():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j, 0, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 1 + 6j, 0, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 1 + 7j, 0, 0])),
        Statevector(np.array([0, 0, 0, 0, 0, 0, 1 + 8j, 0])),
    ]

    with pytest.raises(RuntimeError) as e:
        utils.is_basis(basis)

    assert "small" in str(e.value)


def test_coeff_of_basis():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j])),
    ]
    assert utils.is_basis(basis)

    vec = Statevector(np.array([2 + 5j, 3 + 6j, 4 + 7j, 5 + 8j]))
    coeff = utils.coeff_of_basis(vec, basis)
    assert np.allclose(np.dot(basis, coeff), vec.data)


def test_coeff_of_basis_fail():
    basis = [
        Statevector(np.array([1 + 2j, 0, 0, 0])),
        Statevector(np.array([0, 1 + 3j, 0, 0])),
        Statevector(np.array([0, 0, 1 + 4j, 0])),
        Statevector(np.array([0, 0, 0, 1 + 5j])),
    ]
    assert utils.is_basis(basis)

    vec = Statevector(np.array([2 + 5j, 3 + 6j]))
    with pytest.raises(RuntimeError) as e:
        utils.coeff_of_basis(vec, basis)

    assert "inconsistent" in str(e.value)


def test_binary_basis():
    basis = utils.binary_basis(3)
    assert utils.is_basis(basis)

    assert basis[0] == Statevector([1, 0, 0, 0, 0, 0, 0, 0])
    assert basis[1] == Statevector([0, 1, 0, 0, 0, 0, 0, 0])
    assert basis[7] == Statevector([0, 0, 0, 0, 0, 0, 0, 1])
