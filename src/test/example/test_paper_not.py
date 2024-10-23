import math

import pytest
from qiskit.circuit.library.standard_gates import HGate, XGate
from qiskit.opflow import One, Zero

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import StateConditionError
from dbcquantum.utils import eq_state


def test_superposition_example_1():
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])

    x.add_condition(
        "X_condition",
        lambda pre_state, post_state: (
            eq_state(pre_state, Zero) and eq_state(post_state, One)
        )
        or (eq_state(pre_state, One) and eq_state(post_state, Zero)),
    )

    aqc = AssertQuantumCircuit(1)

    aqc.append(HGate(), [0])
    aqc.append(x, [0])

    with pytest.raises(StateConditionError) as e:
        aqc.run(init_state=Zero)

    assert "Condition Error occured in 'X_condition'" in str(e.value)


def test_superposition_example_2():
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])

    x.add_condition(
        "X_condition",
        lambda pre_state, post_state: (
            eq_state(pre_state, Zero) and eq_state(post_state, One)
        )
        or (eq_state(pre_state, One) and eq_state(post_state, Zero)),
    )

    aqc = AssertQuantumCircuit(1)
    aqc.append(HGate(), [0])

    aqc.append_superposition(
        "x_superposition",
        x,
        pre_superposition_states=[(1 / math.sqrt(2), Zero), (1 / math.sqrt(2), One)],
        post_superposition_states=None,
        qargs=[0],
    )

    aqc.run(init_state=Zero)
