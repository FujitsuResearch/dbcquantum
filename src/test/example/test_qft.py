import pytest
from qiskit.opflow import One, Plus, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.err as err
import dbcquantum.utils as utils

from .qft import make_qft


def test_qft_1():
    state: Statevector = utils.to_Statevector(One ^ Zero ^ Zero ^ One ^ Zero)  # type: ignore
    qft_circuit = make_qft(state.num_qubits)
    qft_circuit.run(state)


def test_qft_2():
    state: Statevector = utils.to_Statevector(One ^ Zero ^ One ^ One)  # type: ignore
    qft_circuit = make_qft(state.num_qubits)
    qft_circuit.run(state)


def test_qft_3():
    state: Statevector = utils.to_Statevector(One ^ Zero ^ Zero ^ Plus ^ Zero)  # type: ignore
    qft_circuit = make_qft(state.num_qubits)

    with pytest.raises(err.StateConditionError) as e:
        qft_circuit.run(state)
    assert (
        "Condition Error occured in 'all qft input states must be |0> or |1>'"
        in str(e.value)
    )
