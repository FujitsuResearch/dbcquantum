import math

import pytest
import qiskit
from qiskit.circuit.library.standard_gates import CXGate, HGate, XGate
from qiskit.opflow import Minus, One, Plus, Zero

import dbcquantum.utils as utils
from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import DbCQuantumError, StateConditionError
from dbcquantum.utils import eq_state


def test_knowledge_1():
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])

    assert eq_state(x.run(Zero), One)

    assert eq_state(x.run(One), Zero)

    h = AssertQuantumCircuit(1)
    h.append(HGate(), [0])

    assert eq_state(h.run(Zero), Plus)
    assert eq_state(Plus, 1 / math.sqrt(2) * (One + Zero))  # type: ignore

    assert eq_state(h.run(One), Minus)
    assert eq_state(Minus, 1 / math.sqrt(2) * (One - Zero))  # type: ignore


# Bell state: 1/√2 * (|00> + |11>)
bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_example_basic_1_success():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    state = make_bell.run()
    assert eq_state(state, bell)


def test_example_basic_1_fail():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(XGate(), [0])
    aqc.append(make_bell, [0, 1])

    with pytest.raises(StateConditionError) as e:
        state = aqc.run()
        assert eq_state(state, bell)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)
    assert "condition_name='bell_input_|00>'" in str(e.value)
    assert (
        "pre_state=\n"
        "  Statevector([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],\n"
        "              dims=(2, 2))" in str(e.value)
    )
    assert "post_state=None" in str(e.value)
    assert "param=None" in str(e.value)


def test_example_basic_2():
    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition_1",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [0]), utils.partial_state(post_state, [1])
        ),
    )

    swap.add_condition(
        "swap_condition_2",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [1]), utils.partial_state(post_state, [0])
        ),
    )

    state = swap.run(Zero ^ One)
    assert eq_state(state, One ^ Zero)  # type:ignore


def test_example_basic_2_fail():
    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    # swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition_1",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [0]), utils.partial_state(post_state, [1])
        ),
    )

    swap.add_condition(
        "swap_condition_2",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [1]), utils.partial_state(post_state, [0])
        ),
    )

    with pytest.raises(StateConditionError) as e:
        state = swap.run(Zero ^ One)
        assert eq_state(state, One ^ Zero)  # type:ignore

    assert "Condition Error occured in 'swap_condition_1'" in str(e.value)


def test_difference_between_conditions():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(make_bell, [0, 1])

    aqc.add_condition(
        "aqc_input_condition",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    with pytest.raises(StateConditionError) as e:
        state = aqc.run(Zero ^ One)
        assert eq_state(state, bell)

    assert "Condition Error occured in 'bell_input_|00>'" in str(e.value)


def test_example_inverse():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    inv_make_bell: AssertQuantumCircuit = make_bell.inverse()

    state = inv_make_bell.run(bell)
    assert eq_state(state, Zero ^ Zero)  # type: ignore


def test_example_inverse_fail():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    inv_make_bell = make_bell.inverse()

    with pytest.raises(StateConditionError) as e:
        state = inv_make_bell.run(One ^ One)
        assert eq_state(state, Zero ^ Zero)  # type: ignore

    assert "Condition Error occured in 'bell_output_bell'" in str(e.value)


# def test_condition_lazy():
#     aqc = AssertQuantumCircuit(2)

#     condition = ("always False", lambda pre_state, post_state, param: False)
#     inf_condition = itertools.repeat(condition)
#     aqc.add_conditions(inf_condition)

#     with pytest.raises(StateConditionError) as e:
#         aqc.run()

#     assert "Condition Error occured in 'always False'" in str(e.value)

#     with pytest.raises(StateConditionError) as e:
#         aqc.inverse().inverse().inverse().run()

#     assert "Condition Error occured in 'always False'" in str(e.value)


def test_qiskit_integrate():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    aqc = AssertQuantumCircuit(2)

    aqc.append(qc.to_gate(), [0, 1])

    aqc.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    aqc.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),
    )

    assert eq_state(aqc.run(), bell)

    inv_aqc = aqc.inverse()
    assert eq_state(inv_aqc.run(bell), Zero ^ Zero)  # type: ignore

    qc_inv_aqc: qiskit.QuantumCircuit = inv_aqc.remove_assertions()
    bell_statevector = utils.to_Statevector(bell)
    assert eq_state(bell_statevector.evolve(qc_inv_aqc), Zero ^ Zero)  # type: ignore


def test_limitation():
    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition_1",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [0]), utils.partial_state(post_state, [1])
        ),
    )

    swap.add_condition(
        "swap_condition_2",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(pre_state, [1]), utils.partial_state(post_state, [0])
        ),
    )

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    aqc = AssertQuantumCircuit(3)
    aqc.append(make_bell, [0, 1])
    aqc.append(HGate(), [2])
    aqc.append(swap, [1, 2])

    with pytest.raises(DbCQuantumError) as e:
        state = aqc.run()
        assert eq_state(utils.partial_state(state, [0, 2]), bell)
        assert eq_state(utils.partial_state(state, [1]), Plus)

    assert "The specified qubits are entangled with the other qubits!" in str(e.value)


def test_limitation_no_condition():
    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    # swap.add_condition(
    #     "swap_condition_1",
    #     lambda pre_state, post_state: eq_state(
    #         utils.partial_state(pre_state, [0]), utils.partial_state(post_state, [1])
    #     ),
    # )

    # swap.add_condition(
    #     "swap_condition_2",
    #     lambda pre_state, post_state: eq_state(
    #         utils.partial_state(pre_state, [1]), utils.partial_state(post_state, [0])
    #     ),
    # )

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_pre_condition(
        "bell_input_|00>",
        lambda pre_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_post_condition(
        "bell_output_bell",
        lambda post_state: eq_state(post_state, bell),  # type: ignore
    )

    aqc = AssertQuantumCircuit(3)
    aqc.append(make_bell, [0, 1])
    aqc.append(HGate(), [2])
    aqc.append(swap, [1, 2])

    state = aqc.run()
    assert eq_state(utils.partial_state(state, [0, 2]), bell)
    assert eq_state(utils.partial_state(state, [1]), Plus)


def test_superposition_example_1():
    cx = AssertQuantumCircuit(2)
    cx.append(CXGate(), [0, 1])

    cx.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    cx.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(cx, [0, 1])

    with pytest.raises(StateConditionError) as e:
        aqc.run()

    assert "Condition Error occured in 'pre_condition1'" in str(e.value)


def test_superposition_example_2():
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
