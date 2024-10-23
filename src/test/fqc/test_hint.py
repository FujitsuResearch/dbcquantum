import math

import pytest
from qiskit.circuit.library.standard_gates import XGate
from qiskit.opflow import One, Zero
from qiskit.quantum_info import Statevector

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import (
    DbCQuantumError,
    StateConditionError,
    SuperpositionStateConditionError,
)
from dbcquantum.utils import eq_state, partial_state, to_Statevector

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_param_basic_1():

    aqc: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc.add_condition_use_param(
        "condition1", lambda pre_state, post_state, param: param == 1
    )

    aqc.run(param=1)


def test_param_basic_1_fail():
    aqc: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc.add_condition_use_param(
        "condition1", lambda pre_state, post_state, param: param == 1
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run(param=2)

    assert "Condition Error occured in 'condition1'" in str(e.value)
    assert str(e.value.info.param) == "2"


def test_param_basic_2():

    aqc: AssertQuantumCircuit[Statevector] = AssertQuantumCircuit(2)
    aqc.append(XGate(), [0])
    aqc.add_condition(
        "condition1",
        lambda pre_state, post_state: eq_state(
            partial_state(post_state, [1]), partial_state(pre_state, [1])
        ),
    )

    aqc.add_condition_use_param(
        "condition2",
        lambda pre_state, post_state, param: eq_state(
            partial_state(post_state, [0]), param
        ),
    )

    state = aqc.run(param=to_Statevector(One))
    assert eq_state(state, Zero ^ One)  # type: ignore

    with pytest.raises(StateConditionError) as e:
        state = aqc.run(param=to_Statevector(Zero))

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_param_basic_3():

    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc1.add_condition_use_param(
        "condition1", lambda pre_state, post_state, param: param == 1
    )

    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc2.append(aqc1, [0, 1])

    aqc2.run(param=1)

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=2)

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_param_basic_4():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 1
    )

    aqc2: AssertQuantumCircuit[tuple[str, int]] = AssertQuantumCircuit(2)

    aqc2.append(aqc1, [0, 1], param_converter=lambda str_int: str_int[1])

    aqc2.add_pre_condition_use_param(
        "condition2", lambda pre_state, param: param[0] == "a"
    )
    aqc2.run(param=("a", 1))

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=("a", 2))

    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=("b", 2))

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_param_basic_5():
    aqc1: AssertQuantumCircuit[Statevector] = AssertQuantumCircuit(1)
    aqc1.append(XGate(), [0])
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: eq_state(param, post_state)
    )

    aqc2: AssertQuantumCircuit[tuple[Statevector, Statevector]] = AssertQuantumCircuit(
        2
    )
    aqc2.append(XGate(), [1])

    aqc2.append(aqc1, [0], param_converter=lambda param: param[1])

    aqc2.add_pre_condition_use_param(
        "condition2",
        lambda pre_state, param: eq_state(param[0], partial_state(pre_state, [0])),
    )
    aqc2.run(
        param=(
            to_Statevector(Zero),
            to_Statevector(One),
        ),
    )

    aqc2.run(
        init_state=One ^ One,
        param=(
            to_Statevector(One),
            to_Statevector(Zero),
        ),
    )


def test_param_inverse_1():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 1
    )

    aqc2: AssertQuantumCircuit[tuple[str, int]] = AssertQuantumCircuit(2)

    aqc2.append(aqc1, [0, 1], param_converter=lambda str_int: str_int[1])

    aqc2.add_pre_condition_use_param(
        "condition2", lambda pre_state, param: param[0] == "a"
    )
    aqc2.inverse().run(param=("a", 1))

    with pytest.raises(StateConditionError) as e:
        aqc2.inverse().run(param=("b", 2))

    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_param_inverse_2():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 1
    )

    aqc2: AssertQuantumCircuit[tuple[str, int]] = AssertQuantumCircuit(2)

    aqc2.append(aqc1, [0, 1], param_converter=lambda str_int: str_int[1])

    aqc2.add_post_condition_use_param(
        "condition2", lambda pre_state, param: param[0] == "a"
    )
    aqc2.inverse().run(param=("a", 1))

    with pytest.raises(StateConditionError) as e:
        aqc2.inverse().run(param=("b", 2))

    assert "Condition Error occured in 'condition2'" in str(e.value)


def test_param_superposition_1():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc1.append(XGate(), [0])
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 1 or param == 2
    )

    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2.append_superposition(
        "superposition",
        aqc1,
        pre_superposition_states=[
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states=[
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
        qargs=[0],
    )

    aqc2.run(param=1)
    aqc2.inverse().inverse().inverse().run(init_state=One, param=1)

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=3)
    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc2.inverse().inverse().inverse().run(init_state=One, param=3)
    assert "Condition Error occured in 'condition1'" in str(e.value)


def test_param_superposition_2():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc1.append(XGate(), [0])
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 0 or param == 1
    )

    aqc2: AssertQuantumCircuit[tuple[Statevector, int, int]] = AssertQuantumCircuit(1)

    aqc2.append_superposition_gen(
        "superposition1",
        aqc1,
        pre_superposition_states_gen=lambda p, h: [
            (1 + 0j, h[0]),
            (0 + 0j, One),
        ],
        post_superposition_states_gen=lambda p, h: [
            (1 + 0j, One),
            (0 + 0j, h[0]),
        ],
        qargs=[0],
        param_converter=lambda h: [h[1], h[2]],
    )

    aqc2.run(param=(to_Statevector(Zero), 0, 1))

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=(to_Statevector(Zero), 0, 100))
    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc2.run(param=(to_Statevector(Zero), 100, 1))
    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc2.run(param=(to_Statevector(One), 0, 100))
    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )


def test_param_superposition_3():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc1.append(XGate(), [0])
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 0 or param == 1
    )

    aqc2: AssertQuantumCircuit[tuple[Statevector, int, int]] = AssertQuantumCircuit(1)

    aqc2.append_superposition_gen(
        "superposition1",
        aqc1,
        pre_superposition_states_gen=lambda p, h: [
            (1 + 0j, Zero),
            (0 + 0j, h[0]),
        ],
        post_superposition_states_gen=lambda p, h: [
            (1 + 0j, h[0]),
            (0 + 0j, Zero),
        ],
        qargs=[0],
        param_converter=lambda h: [h[1], h[2]],
    )

    aqc2.inverse().run(init_state=One, param=(to_Statevector(One), 0, 1))

    with pytest.raises(StateConditionError) as e:
        aqc2.inverse().run(init_state=One, param=(to_Statevector(One), 0, 100))
    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc2.inverse().run(init_state=One, param=(to_Statevector(One), 100, 1))
    assert "Condition Error occured in 'condition1'" in str(e.value)

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc2.inverse().run(init_state=One, param=(to_Statevector(Zero), 100, 1))
    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )


def test_param_superposition_length_param_fail():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc1.append(XGate(), [0])
    aqc1.add_post_condition_use_param(
        "condition1", lambda post_state, param: param == 1 or param == 2
    )

    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2.append_superposition(
        "superposition",
        aqc1,
        pre_superposition_states=[
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states=[
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
        qargs=[0],
        param_converter=lambda h: [h],
    )

    with pytest.raises(DbCQuantumError) as e:
        aqc2.run(param=1)
    assert (
        "The length of pre_superposition_states is inconsistent with param_converter."
        in str(e.value)
    )

    with pytest.raises(DbCQuantumError) as e:
        aqc2.inverse().run(init_state=One, param=1)
    assert (
        "The length of pre_superposition_states is inconsistent with param_converter."
        in str(e.value)
    )
