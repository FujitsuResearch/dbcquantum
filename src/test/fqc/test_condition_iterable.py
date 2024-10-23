import math

import pytest
from qiskit.circuit.library.standard_gates import CXGate, HGate
from qiskit.opflow import One, Zero

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import StateConditionError
from dbcquantum.utils import eq_state

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_condition_iterator_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    state = make_bell.run()
    assert eq_state(state, bell)

    state = make_bell.run()
    assert eq_state(state, bell)


def test_condition_iterator_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)


def test_inverse_condition_iterator_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "always true1",
        lambda pre_state, post_state: True,
    )

    make_bell.add_condition(
        "always true2",
        lambda pre_state, post_state: True,
    )

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    state = make_bell.run()
    assert eq_state(state, bell)

    inv_make_bell = make_bell.inverse()

    state = make_bell.run()
    assert eq_state(state, bell)

    state = inv_make_bell.run(bell)
    assert eq_state(state, Zero ^ Zero)  # type: ignore

    state = inv_make_bell.run(bell)
    assert eq_state(state, Zero ^ Zero)  # type: ignore


def test_inverse_condition_iterator_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "always true1",
        lambda pre_state, post_state: True,
    )

    make_bell.add_condition(
        "always true2",
        lambda pre_state, post_state: True,
    )

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    inv_make_bell = make_bell.inverse()

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        inv_make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        inv_make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)


# def test_inverse_condition_iterator_3():
#     make_bell = AssertQuantumCircuit(2)
#     make_bell.append(HGate(), [0])
#     make_bell.append(CXGate(), [0, 1])

#     make_bell.add_condition(
#         "always true",
#         lambda pre_state, post_state: True,
#     )

#     make_bell.add_condition(
#         "bell_output_bell",
#         lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
#     )

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     inv_make_bell = make_bell.inverse()

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)
#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)


def test_pre_post_condition_iterator_inverse_1():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "always true",
        lambda _: True,
    )

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "always true",
        lambda _: True,
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    state = make_bell.run()
    assert eq_state(state, bell)

    inv_make_bell = make_bell.inverse()
    state = inv_make_bell.run(bell)
    assert eq_state(state, Zero ^ Zero)  # type: ignore

    state = make_bell.run()
    assert eq_state(state, bell)

    inv_make_bell = make_bell.inverse()
    state = inv_make_bell.run(bell)
    assert eq_state(state, Zero ^ Zero)  # type: ignore


def test_pre_post_condition_iterator_inverse_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "always true",
        lambda _: True,
    )

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "always true",
        lambda _: True,
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    inv_make_bell = make_bell.inverse()

    with pytest.raises(StateConditionError) as e:
        inv_make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

    with pytest.raises(StateConditionError) as e:
        inv_make_bell.run(Zero ^ One)

    assert "Condition Error occured in 'bell_output_bell'" in str(e.value)


# def test_condition_iterator_infinite_1():
#     make_bell = AssertQuantumCircuit(2)
#     make_bell.append(HGate(), [0])
#     make_bell.append(CXGate(), [0, 1])

#     def pre_condition(pre_state: Statevector, post_state: Statevector, param) -> bool:
#         return eq_state(pre_state, Zero ^ Zero)  # type: ignore

#     def post_condition(pre_state: Statevector, post_state: Statevector, param) -> bool:
#         return eq_state(post_state, bell)

#     make_bell.add_conditions(itertools.repeat(("bell_input_|00>", pre_condition)))
#     make_bell.add_conditions(itertools.repeat(("bell_output_bell", post_condition)))

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

#     inv_make_bell = make_bell.inverse()

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)


# def test_condition_iterator_infinite_2():
#     make_bell = AssertQuantumCircuit(2)
#     make_bell.append(HGate(), [0])
#     make_bell.append(CXGate(), [0, 1])

#     def pre_condition(pre_state: Statevector, param) -> bool:
#         return eq_state(pre_state, Zero ^ Zero)  # type: ignore

#     def post_condition(post_state: Statevector, param) -> bool:
#         return eq_state(post_state, bell)

#     make_bell.add_pre_conditions(itertools.repeat(("bell_input_|00>", pre_condition)))
#     make_bell.add_post_conditions(
#         itertools.repeat(("bell_output_bell", post_condition))
#     )

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

#     inv_make_bell = make_bell.inverse()

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         inv_make_bell.run(Zero ^ One)

#     assert "Condition Error occured in 'bell_output_bell'" in str(e.value)
