import math

import pytest
from qiskit.circuit.library.standard_gates import CXGate, HGate, XGate
from qiskit.opflow import Minus, One, Plus, Zero

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import (
    DbCQuantumError,
    StateConditionError,
    SuperpositionStateConditionError,
)
from dbcquantum.utils import binary_basis, eq_state, partial_state

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_superposition_basic1():
    cx = AssertQuantumCircuit(2)
    cx.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        cx,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)


def test_superposition_basic2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.append(XGate(), [1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), One ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    final_state = 1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One))  # type: ignore

    state = aqc.run()
    assert eq_state(state, final_state)

    state = aqc.run()
    assert eq_state(state, final_state)


def test_superposition_basic3():
    aqc1 = AssertQuantumCircuit(3)
    aqc1.append(CXGate(), [0, 1])
    aqc1.append(XGate(), [1])
    aqc1.append(HGate(), [2])

    aqc2 = AssertQuantumCircuit(4)
    aqc2.append(HGate(), [3])
    aqc2.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Plus ^ One ^ Zero),
            (1 / math.sqrt(2), Plus ^ Zero ^ One),  # type: ignore
        ],
        qargs=[3, 1, 0],
    )

    aqc = AssertQuantumCircuit(5)
    aqc.append(aqc2, [1, 2, 3, 4])
    aqc.append(XGate(), [0])
    aqc.append(HGate(), [0])

    final_state = (
        (1 / math.sqrt(2))
        * ((Zero ^ Zero ^ One ^ Plus) + (One ^ Zero ^ Zero ^ Plus))  # type: ignore
    ) ^ Minus

    state = aqc.run()
    assert eq_state(partial_state(state, [0]), Minus)
    assert eq_state(partial_state(state, [1]), Plus)
    assert eq_state(partial_state(state, [3]), Zero)
    assert eq_state(
        partial_state(state, [2, 4]),
        1 / math.sqrt(2) * ((One ^ Zero) + (Zero ^ One)),  # type: ignore
    )
    assert eq_state(state, final_state)

    state = aqc.run()
    assert eq_state(state, final_state)


def test_superposition_matching_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    state = aqc.run()
    assert eq_state(state, bell)


def test_superposition_matching_2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), One ^ One),
            (1 / math.sqrt(2), Zero ^ Zero),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )


def test_superposition_matching_3():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), One ^ One),
            (1 / math.sqrt(2), Zero ^ Zero),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )


def test_superposition_matching_num_err_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
            (0, One ^ Minus),
        ],
        qargs=[0, 1],
    )

    with pytest.raises(DbCQuantumError) as e:
        aqc.run()

    assert (
        str(e.value) == "The length of pre_superposition_states and "
        "post_superposition_states is inconsistent!"
    )

    with pytest.raises(DbCQuantumError) as e:
        aqc.run()

    assert (
        "The length of pre_superposition_states and "
        "post_superposition_states is inconsistent!" in str(e.value)
    )


# def test_superposition_matching_num_err_2():
#     aqc1 = AssertQuantumCircuit(2)
#     aqc1.append(CXGate(), [0, 1])

#     aqc = AssertQuantumCircuit(2)
#     aqc.append(HGate(), [0])
#     aqc.append_superposition(
#         "superposition1",
#         aqc1,
#         pre_superposition_states=[
#             (1 / math.sqrt(2), Zero ^ Zero),
#             (1 / math.sqrt(2), Zero ^ One),  # type: ignore
#         ],
#         post_superposition_states=[
#             (1 / math.sqrt(2), Zero ^ Zero),
#             (1 / math.sqrt(4), One ^ Plus),  # type: ignore
#             (-1 / math.sqrt(4), One ^ Minus),
#         ],
#         qargs=[0, 1],
#         check_match=False,
#     )

#     state = aqc.run()
#     assert eq_state(state, bell)

#     state = aqc.run()
#     assert eq_state(state, bell)


def test_superposition_pre_not_match():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), One ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )


def test_superposition_post_not_match_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )


def test_superposition_nest_condition_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    state = aqc.run()
    assert eq_state(state, bell)


def test_superposition_nest_condition_1_meaning():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(aqc1, [0, 1])

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'pre_condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'pre_condition1'" in str(e.value)


def test_superposition_nest_condition_2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1", lambda pre_condition: eq_state(pre_condition, Plus ^ Zero)  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1", lambda post_condition: eq_state(post_condition, bell)
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'pre_condition1'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'pre_condition1'" in str(e.value)


def test_superposition_post_None_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=None,
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    state = aqc.run()
    assert eq_state(state, bell)


def test_superposition_pre_None_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=None,
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    with pytest.raises(DbCQuantumError) as e:
        aqc.run()

    assert (
        "The state vector cannot be decomposed because "
        "it is not specified in 'superposition1'."
    ) in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.run()

    assert (
        "The state vector cannot be decomposed because "
        "it is not specified in 'superposition1'."
    ) in str(e.value)


def test_superposition_pre_None_2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=None,
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    aqc.inverse().run(init_state=bell)


# def test_superposition_pre_None_2():
#     aqc1 = AssertQuantumCircuit(2)
#     aqc1.append(CXGate(), [0, 1])
#     aqc1.add_pre_condition(
#         "pre_condition1", lambda pre_condition: eq_state(pre_condition, Zero ^ Plus)  # type: ignore
#     )
#     aqc1.add_post_condition(
#         "post_condition1", lambda post_condition: eq_state(post_condition, bell)
#     )

#     aqc = AssertQuantumCircuit(2)
#     aqc.append(HGate(), [0])
#     aqc.append_superposition(
#         "superposition1",
#         aqc1,
#         pre_superposition_states=None,
#         post_superposition_states=[
#             (1 / math.sqrt(2), Zero ^ Zero),
#             (1 / math.sqrt(2), One ^ Zero),  # type: ignore
#         ],
#         qargs=[0, 1],
#         check_match=False,
#     )

#     with pytest.raises(SuperpositionStateConditionError) as e:
#         aqc.run()

#     assert (
#         "The post_state and post_superposition_states don't match in 'superposition1'"
#         in str(e.value)
#     )

#     with pytest.raises(SuperpositionStateConditionError) as e:
#         aqc.run()

#     assert (
#         "The post_state and post_superposition_states don't match in 'superposition1'"
#         in str(e.value)
#     )


def test_inverse_test_superposition_1():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    inv_aqc = aqc.inverse()
    inv_state = inv_aqc.run(bell)
    assert eq_state(inv_state, Zero ^ Zero)  # type: ignore


def test_inverse_test_superposition_2():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), One ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    inv_aqc = aqc.inverse()

    with pytest.raises(SuperpositionStateConditionError) as e:
        inv_aqc.run(bell)

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        inv_aqc.run(bell)

    assert (
        "The evolved pre_superposition_states and post_superposition_states "
        "don't match in 'superposition1'" in str(e.value)
    )


def test_inverse_test_superposition_3():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ Zero),  # type: ignore
        ],
        qargs=[0, 1],
    )

    inv_aqc = aqc.inverse()

    with pytest.raises(SuperpositionStateConditionError) as e:
        inv_aqc.run(bell)

    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )

    assert e.value.info.condition_name == "superposition1"

    with pytest.raises(SuperpositionStateConditionError) as e:
        inv_aqc.run(bell)

    assert (
        "The pre_state and pre_superposition_states don't match in 'superposition1'"
        in str(e.value)
    )


def test_inverse_inverse_test_superposition():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    inv_aqc = aqc.inverse()
    inv_state = inv_aqc.run(bell)
    assert eq_state(inv_state, Zero ^ Zero)  # type: ignore

    inv_inv_aqc = inv_aqc.inverse()
    inv_inv_state = inv_inv_aqc.run()
    assert eq_state(inv_inv_state, bell)  # type: ignore


def test_append_superposition_basis():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.x(1)
    aqc1.add_pre_condition(
        "condition1",
        lambda pre: eq_state(pre, Zero ^ Zero)  # type: ignore
        or eq_state(pre, Zero ^ One)  # type: ignore
        or eq_state(pre, One ^ Zero)  # type: ignore
        or eq_state(pre, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.h(0)
    aqc.h(1)
    aqc.append_superposition_basis(
        "sup1", aqc1, [0, 1], pre_state_basis=binary_basis(2)
    )
    aqc.run()


def test_append_superposition_basis_assert_fail():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.x(1)
    aqc1.add_pre_condition(
        "condition1",
        lambda pre: eq_state(pre, Zero ^ Zero)  # type: ignore
        or eq_state(pre, Zero ^ One)  # type: ignore
        or eq_state(pre, One ^ Zero),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.h(0)
    aqc.h(1)
    aqc.append_superposition_basis(
        "sup1", aqc1, [0, 1], pre_state_basis=binary_basis(2)
    )
    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "condition1" in str(e.value)


# def test_append_superposition_basis_fail():
#     aqc1 = AssertQuantumCircuit(2)
#     aqc1.x(1)
#     aqc1.add_pre_condition(
#         "condition1",
#         lambda pre: eq_state(pre, Zero ^ Zero)  # type: ignore
#         or eq_state(pre, Zero ^ One)  # type: ignore
#         or eq_state(pre, One ^ Zero)  # type: ignore
#         or eq_state(pre, One ^ One),  # type: ignore
#     )

#     aqc = AssertQuantumCircuit(2)
#     aqc.h(0)
#     aqc.h(1)

#     with pytest.raises(RuntimeError) as e:
#         aqc.append_superposition_basis(
#             "sup1", aqc1, [0, 1], pre_state_basis=binary_basis(3)
#         )

#     assert "The size of pre_state_basis is inconsistent with qargs." in str(e.value)


def test_append_superposition_post_basis():
    aqc1 = AssertQuantumCircuit(1)
    aqc1.x(0)
    aqc1.add_pre_condition(
        "condition1",
        lambda pre: eq_state(pre, Zero)  # type: ignore
        or eq_state(pre, One),  # type: ignore
    )

    aqc1.add_post_condition(
        "condition2",
        lambda pre: eq_state(pre, Zero)  # type: ignore
        or eq_state(pre, One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(1)
    aqc.h(0)
    aqc.append_superposition_basis(
        "sup1",
        aqc1,
        [0],
        pre_state_basis=binary_basis(1),
        post_state_basis=reversed(binary_basis(1)),
    )

    aqc.run()


def test_append_superposition_post_basis_basis_fail():
    aqc1 = AssertQuantumCircuit(1)
    aqc1.x(0)
    aqc1.add_pre_condition(
        "condition1",
        lambda pre: eq_state(pre, Zero)  # type: ignore
        or eq_state(pre, One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(1)
    aqc.h(0)
    aqc.append_superposition_basis(
        "sup1",
        aqc1,
        pre_state_basis=binary_basis(1),
        post_state_basis=binary_basis(1),
        qargs=[0],
    )

    with pytest.raises(SuperpositionStateConditionError) as e:
        aqc.run()

    assert "sup1" in str(e.value)


# def test_append_superposition_post_basis_fail():
#     aqc1 = AssertQuantumCircuit(2)
#     aqc1.x(1)
#     aqc1.add_pre_condition(
#         "condition1",
#         lambda pre: eq_state(pre, Zero ^ Zero)  # type: ignore
#         or eq_state(pre, Zero ^ One)  # type: ignore
#         or eq_state(pre, One ^ Zero)  # type: ignore
#         or eq_state(pre, One ^ One),  # type: ignore
#     )

#     aqc = AssertQuantumCircuit(2)
#     aqc.h(0)
#     aqc.h(1)

#     with pytest.raises(RuntimeError) as e:
#         aqc.append_superposition_basis(
#             "sup1",
#             aqc1,
#             [0, 1],
#             pre_state_basis=binary_basis(2),
#             post_state_basis=binary_basis(3),
#         )

#     assert "The size of post_state_basis is inconsistent with qargs." in str(e.value)
