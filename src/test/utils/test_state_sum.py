import math

from qiskit.opflow import One, Plus, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils

zero: Statevector = utils.to_Statevector(Zero)
one: Statevector = utils.to_Statevector(One)
jone: Statevector = Statevector(utils.to_Statevector(One).data / (1j))


def test_state_sum_1():
    plus = utils.state_sum(
        [(1 / math.sqrt(2), zero), (1 / math.sqrt(2), one)], num_qubits=1
    )

    assert utils.eq_state(Plus, plus)


def test_state_sum_2():
    plus = utils.state_sum(
        [(1 / math.sqrt(2), zero), (1 / math.sqrt(2) * (1j), jone)], num_qubits=1
    )

    assert utils.eq_state(Plus, plus)
